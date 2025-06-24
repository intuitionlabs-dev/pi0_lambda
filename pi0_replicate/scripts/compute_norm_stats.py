"""Compute normalization statistics for a config.

This script automatically falls back to the *CPU* JAX backend if no
CUDA-visible GPU is detected.  That avoids the common "No visible GPU
devices" crash when running on CPU-only machines.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

# -----------------------------------------------------------------------------
# Fallback to CPU when no GPU is present *before* JAX gets imported.
# -----------------------------------------------------------------------------

import os

try:
    import pynvml  # type: ignore

    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
except ModuleNotFoundError:
    # pynvml not installed – assume CPU-only environment.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

# -----------------------------------------------------------------------------

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.transforms import RemoveStrings
import openpi.transforms as transforms


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    # Disable video decoding to speed up statistics computation. We only need
    # numeric state/actions.  Provide lightweight dummy images instead of
    # actually decoding frames so later transforms that expect image keys do
    # not fail with KeyError.
    if hasattr(dataset, "_query_videos"):
        shapes = getattr(dataset.meta, "shapes", {})

        def _dummy_videos(query_timestamps, ep_idx):  # noqa: D401, ANN001
            out = {}
            for key in query_timestamps:
                shape = shapes.get(key, (64, 64, 3))
                out[key] = np.zeros(shape, dtype=np.uint8)
            return out

        dataset._query_videos = _dummy_videos
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    shuffle = False
    
    # import ipdb; ipdb.set_trace()
    
    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=1,
        num_workers=0,
        shuffle=shuffle,
        num_batches=num_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_frames, desc="Computing stats"):
        for key in keys:
            # import ipdb; ipdb.set_trace()
            values = np.asarray(batch[key][0])
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
