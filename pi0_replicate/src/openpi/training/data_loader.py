from collections.abc import Iterator, Sequence
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
import random
try:
    import zarr  # type: ignore
except ImportError:  # pragma: no cover
    zarr = None  # noqa: N816  # Allow upper-case alias to match import style

try:
    import nvidia.dali as dali  # type: ignore
    from nvidia.dali.plugin.pytorch import DALIGenericIterator  # type: ignore
except ImportError:  # pragma: no cover
    dali = None  # type: ignore
    DALIGenericIterator = None  # type: ignore

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class ZarrDataset(Dataset):
    """Simple Zarr-backed random-access dataset.

    The dataset expects a Zarr *group* that contains one or more N-dimensional
    arrays with a common leading sample dimension.  Nested groups are traversed
    recursively – arrays are flattened into a single dict using ``/``-separated
    keys (e.g. ``"obs/images"``).

    Notes
    -----
    • The dataset works entirely on NumPy ndarrays (no GPU memory pressure).
    • You *must* provide an ``actions`` array – it will be used as the target.
    """

    def __init__(self, zarr_path: str):
        if zarr is None:
            raise ImportError(
                "`zarr` is not installed.  Install it with `pip install zarr` to "
                "use ZarrDataset."
            )

        self._root = zarr.open_group(zarr_path, mode="r")
        self._arrays = self._collect_arrays(self._root)
        if not self._arrays:
            raise ValueError(f"No arrays found in Zarr group at '{zarr_path}'.")

        # All arrays must share the leading dimension – use the smallest for safety
        self._len = min(arr.shape[0] for arr in self._arrays.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_arrays(group: "zarr.Group", prefix: str = "") -> dict[str, "zarr.Array"]:
        """Recursively gather all arrays inside *group* (depth-first)."""
        arrays: dict[str, "zarr.Array"] = {}
        for key, item in group.items():
            if isinstance(item, zarr.Array):
                arrays[prefix + key] = item
            else:  # subgroup → recurse
                arrays.update(ZarrDataset._collect_arrays(item, prefix + key + "/"))
        return arrays

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __getitem__(self, index: SupportsIndex) -> dict:
        i = index.__index__()
        if i >= self._len:
            raise IndexError(i)
        # Convert to NumPy eagerly – JAX can map later.
        return {k: np.asarray(arr[i]) for k, arr in self._arrays.items()}

    def __len__(self) -> int:  # noqa: D401  (simple function)
        return self._len


class ZarrDALILoader:
    """High-performance batched loader that bridges ZarrDataset → JAX via NVIDIA DALI.

    We rely on ``external_source`` to stream samples from Python into the
    pipeline.  Even though this means the Python GIL remains a limiting factor,
    we still benefit from GPU transfer overlap, built-in shuffling, and batched
    collation performed by DALI.
    """

    def __init__(
        self,
        dataset: ZarrDataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_threads: int = 4,
        device_id: int = 0,
    ) -> None:
        if dali is None or DALIGenericIterator is None:
            raise ImportError(
                "`nvidia-dali` is not installed.  Install it with the CUDA-specific "
                "package, e.g. `pip install nvidia-dali-cuda12`."
            )
        if jax.process_count() > 1:
            raise NotImplementedError("Multi-process data loading is not yet supported.")
        if len(dataset) < local_batch_size:
            raise ValueError(
                f"Local batch size ({local_batch_size}) is larger than dataset size "
                f"({len(dataset)})."
            )

        if sharding is None:
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)), jax.sharding.PartitionSpec("B")
            )
        self._sharding = sharding
        self._num_batches = num_batches

        # Keep an explicit reference so that DALI's pipeline (running in a
        # separate thread) can access the Python object.
        self._dataset = dataset
        self._keys = list(dataset._arrays.keys())

        # ------------------------------------------------------------------
        # DALI pipeline definition
        # ------------------------------------------------------------------
        class _ZarrPipeline(dali.pipeline.Pipeline):
            def __init__(self, dataset: ZarrDataset, keys: list[str]):
                super().__init__(
                    batch_size=local_batch_size,
                    num_threads=num_threads,
                    device_id=device_id,
                    seed=42,
                )
                self._dataset = dataset
                self._keys = keys
                self._shuffle = shuffle
                self._order = list(range(len(dataset)))
                if self._shuffle:
                    random.shuffle(self._order)
                self._pos = 0

                def _sample(_):  # signature must accept sample_info (ignored)
                    if self._pos >= len(self._order):
                        self._pos = 0
                        if self._shuffle:
                            random.shuffle(self._order)
                    idx = self._order[self._pos]
                    self._pos += 1
                    sample = self._dataset[idx]
                    # External source expects a *sequence* when num_outputs > 1
                    return [sample[k] for k in self._keys]

                # Produce one output per key (all on the CPU).
                self._outs = dali.fn.external_source(
                    source=_sample,
                    batch=False,
                    num_outputs=len(self._keys),
                    no_copy=True,
                )

            def define_graph(self):  # noqa: D401 (simple)
                return self._outs

        self._pipeline = _ZarrPipeline(dataset, self._keys)
        self._pipeline.build()

        self._iterator = DALIGenericIterator(
            pipelines=[self._pipeline],
            output_map=self._keys,
            reader_name=None,
            last_batch_policy=dali.plugin.base_iterator.LastBatchPolicy.DROP,
            auto_reset=True,
        )

    # ------------------------------------------------------------------
    # Python iterator interface
    # ------------------------------------------------------------------
    def __iter__(self):  # noqa: D401 (simple)
        produced = 0
        while True:
            if self._num_batches is not None and produced >= self._num_batches:
                return
            try:
                dali_batch = next(self._iterator)
            except StopIteration:
                return
            # DALIGenericIterator returns a list with length == num pipelines.
            batch_dict = {k: v[0].cpu().numpy() for k, v in dali_batch[0].items()}
            produced += 1
            yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch_dict)


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=data_config.dataset_root)

    # ------------------------------------------------------------------
    # Allow users to override video backend via the ``LEROBOT_VIDEO_BACKEND``
    # environment variable.  LeRobot currently supports only the
    # back-ends below.  If the user requests an unsupported alias we fall
    # back to the default ("pyav") instead of crashing with
    # ``ValueError: Unsupported video backend``.
    # ------------------------------------------------------------------

    video_backend = os.getenv("LEROBOT_VIDEO_BACKEND", "pyav")
    _SUPPORTED_BACKENDS = {"pyav", "video_reader", "torchcodec"}

    # Common synonyms that users might try. Map them to a valid backend.
    _ALIASES = {
        "torchvision": "video_reader",  # torchvision's accelerated decoder
        "pil": "pyav",  # no PIL backend – use default
        "image": "pyav",  # ditto (was available in older LeRobot revs)
    }

    video_backend = _ALIASES.get(video_backend, video_backend)

    if video_backend not in _SUPPORTED_BACKENDS:
        import warnings

        warnings.warn(
            f"LEROBOT_VIDEO_BACKEND='{video_backend}' is not supported; "
            "falling back to 'pyav'.  Supported options: pyav, video_reader, torchcodec.",
            RuntimeWarning,
        )
        video_backend = "pyav"

    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=data_config.dataset_root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        video_backend=video_backend,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training."""
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
        )

    # ------------------------------------------------------------------
    # Zarr+DALI fast-path – detect via `zarr_data_dir` attribute or `.zarr` suffix
    # ------------------------------------------------------------------
    zarr_dir: str | None = None
    if hasattr(data_config, "zarr_data_dir"):
        zarr_dir = typing.cast(str | None, getattr(data_config, "zarr_data_dir"))
    # Fallback heuristic: repo_id or dataset_root directory ending with ".zarr"
    if zarr_dir is None and data_config.repo_id and str(data_config.repo_id).endswith(".zarr"):
        zarr_dir = typing.cast(str, data_config.repo_id)
    if zarr_dir is None and data_config.dataset_root and str(data_config.dataset_root).endswith(".zarr"):
        zarr_dir = typing.cast(str, data_config.dataset_root)

    if zarr_dir is not None:
        return create_zarr_dali_data_loader(
            zarr_dir,
            data_config,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
        )

    # ------------------------------------------------------------------
    # Default (PyTorch) loader
    # ------------------------------------------------------------------
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]


# -----------------------------------------------------------------------------
# Backwards-compat helper – exposed for scripts.compute_norm_stats
# -----------------------------------------------------------------------------

def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Return a *raw* (unbatched) dataset given a DataConfig.

    Older utilities (e.g. `scripts.compute_norm_stats`) expect
    `openpi.training.data_loader.create_dataset`.  It was removed during
    refactor; we restore a thin wrapper here that delegates to the current
    helper functions.
    """

    if data_config.rlds_data_dir is not None:
        # Use RLDS dataset (e.g. DROID). We return the underlying dataset
        # without any transforms; stats script will wrap it in
        # `TransformedDataset` afterwards.
        return create_rlds_dataset(
            data_config,
            action_horizon=model_config.action_horizon,
            batch_size=1,
            shuffle=False,
        )

    # Standard LeRobot dataset.
    return create_torch_dataset(
        data_config,
        action_horizon=model_config.action_horizon,
        model_config=model_config,
    )


def create_zarr_dali_data_loader(
    zarr_path: str,
    data_config: _config.DataConfig,
    *,
    batch_size: int,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Factory that wires *ZarrDataset* + *ZarrDALILoader* together and wraps
    everything in the standard ``DataLoaderImpl`` used throughout the codebase.
    """

    dataset: Dataset = ZarrDataset(zarr_path)
    # Apply standard transforms (incl. normalization) – they operate on
    # *individual* samples, hence we wrap before batching.
    dataset = transform_dataset(dataset, data_config)

    # Note: we divide global batch by process count to mirror behaviour of the
    # existing TorchDataLoader.  Multi-process JAX not supported yet.
    local_bs = batch_size // jax.process_count()
    data_loader = ZarrDALILoader(
        dataset=dataset,
        local_batch_size=local_bs,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return DataLoaderImpl(data_config, data_loader)
