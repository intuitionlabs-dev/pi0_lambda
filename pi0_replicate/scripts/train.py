"""Training entry-point.

Automatically falls back to CPU JAX backend when no GPU is present to
prevent runtime crashes on CPU-only machines.
"""

# -----------------------------------------------------------------------------
# CPU fallback – must run before the first `jax` import in this file.
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
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb
import concurrent.futures as futures  # For background checkpoint uploads

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

_HF_EXECUTOR = futures.ThreadPoolExecutor(max_workers=1)


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def _async_upload(repo_id: str, ckpt_path: epath.Path, exp_name: str, step: int) -> None:
    """Background task to upload a checkpoint directory to HuggingFace Hub."""
    try:
        from huggingface_hub import upload_folder  # Imported here to avoid global dependency at startup.

        logging.info("Uploading checkpoint %s to HF repo %s (async)", ckpt_path, repo_id)
        upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(ckpt_path),
            path_in_repo=f"{exp_name}/{step}",
            commit_message=f"Add checkpoint step {step}",
            token=os.getenv("HF_TOKEN"),
        )
    except Exception:
        logging.exception("Failed to upload checkpoint to HF")


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # ------------------------------------------------------------------
    # Optional: speed-up during training by bypassing MP4 decoding.
    # If the dataset directory contains extracted PNG frames at
    #   images/{cam_key}/episode_{episode_index}/frame_{frame_index}.png
    # then we override LeRobotDataset._query_videos to load those PNGs
    # directly instead of seeking through MP4 files.
    # ------------------------------------------------------------------

    import numpy as _np
    from PIL import Image as _PIL

    try:
        import lerobot.common.datasets.lerobot_dataset as _ld

        _orig_query_videos = _ld.LeRobotDataset._query_videos  # save original

        # Fast path: if the user sets BYPASS_IMAGES=1 we will ignore all video
        # decoding entirely and feed dummy zero images.  This guarantees no
        # torch-vision video back-end is invoked and removes any I/O stall.
        _BYPASS = bool(os.getenv("BYPASS_IMAGES", "0") == "1")

        # ------------------------------------------------------------------
        # Notify the user which image-loading path will be taken. This prints
        # only *once* at startup so that the log is not flooded during
        # training.
        # ------------------------------------------------------------------
        if _BYPASS:
            logging.info(
                "BYPASS_IMAGES=1 → Will *not* decode MP4 videos. If extracted PNG "
                "frames are found they will be loaded; otherwise the loader will "
                "return black dummy frames (images masked out downstream)."
            )
        else:
            logging.info(
                "BYPASS_IMAGES=0 → Will load extracted PNG frames when present, "
                "falling back to on-the-fly MP4 decoding if PNGs are absent."
            )

        def _png_query(self, query_ts, ep_idx):  # type: ignore[override]
            fps = self.meta.fps
            out = {}

            # We print an informative message the *first* time this helper is
            # called for each data-loading mode to clarify whether we are
            # pulling PNGs, black dummy frames, or falling back to MP4 decode.
            # A simple function attribute is used as a one-time flag.
            def _log_once(msg: str, level: str = "INFO") -> None:
                """Print *msg* once per worker/process.

                We avoid the standard logging module here because this helper
                runs inside PyTorch DataLoader *worker* processes – their log
                records are not automatically forwarded to the main process.
                A direct `print(..., flush=True)` ensures the note appears in
                the parent terminal/stdout regardless of worker isolation.  A
                function attribute tracks messages we have already printed so
                that each distinct line is emitted only once per process.
                """

                if not hasattr(_png_query, "_printed_messages"):
                    _png_query._printed_messages = set()  # type: ignore[attr-defined]
                if msg not in _png_query._printed_messages:  # type: ignore[attr-defined]
                    print(f"[{level}] {msg}", flush=True)
                    _png_query._printed_messages.add(msg)  # type: ignore[attr-defined]

            for key, ts_list in query_ts.items():
                imgs = []
                for ts in ts_list:
                    frame_idx = round(ts * fps)
                    img_path = self._get_image_file_path(ep_idx, key, frame_idx)
                    if not img_path.is_file():
                        if _BYPASS:
                            # Return zeros in the expected shape, no decode
                            shape = self.meta.shapes.get(key, (224, 224, 3))
                            dummy = _np.zeros(shape, _np.float32)
                            imgs.append(dummy)
                            _log_once(
                                "PNG frame missing → returning dummy zeros because BYPASS_IMAGES=1. No videos will be read.",
                                level="WARNING",
                            )
                            continue
                        # Otherwise fallback to original video decoding
                        _log_once(
                            "PNG frame missing → decoding MP4 video on-the-fly.",
                            level="INFO",
                        )
                        return _orig_query_videos(self, query_ts, ep_idx)
                    arr = _np.asarray(_PIL.open(img_path).convert("RGB"), dtype=_np.uint8)  # HWC
                    imgs.append(arr)
                    _log_once(
                        "Decoding pre-extracted PNG image frames.",
                        level="INFO",
                    )
                stack = imgs[0] if len(imgs) == 1 else _np.stack(imgs)
                if stack.ndim == 3:
                    stack = _np.expand_dims(stack, 0)
                out[key] = stack
            return out

        _ld.LeRobotDataset._query_videos = _png_query  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        pass
    
    # import ipdb; ipdb.set_trace()

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

            # Ensure Orbax finished writing so the directory has been renamed from
            # *.orbax-checkpoint-tmp-* → <step>/ before we start uploading.
            checkpoint_manager.wait_until_finished()

            # ------------------------------------------------------------------
            # Push checkpoint to HuggingFace Hub in a background thread so that
            # training does not block on network I/O.
            # ------------------------------------------------------------------
            if (
                config.hf_repo_id is not None
                and jax.process_index() == 0
            ):
                ckpt_path = checkpoint_manager.directory / str(step)
                _HF_EXECUTOR.submit(_async_upload, config.hf_repo_id, ckpt_path, config.exp_name, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()

    # Wait for any outstanding uploads before exiting.
    logging.info("Waiting for HuggingFace uploads to finish")
    _HF_EXECUTOR.shutdown(wait=True)


if __name__ == "__main__":
    main(_config.cli())
