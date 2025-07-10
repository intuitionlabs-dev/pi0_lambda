import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# NOTE: This transform targets *dual-arm* (14-DoF) Piper recordings.
# It expects two wrist cameras (left & right) plus a workspace camera.
# The dataset keys below should be mapped upstream via `RepackTransform` so
# that the flattened sample contains:
#   observation/image               – workspace RGB
#   observation/left_wrist_image    – left wrist RGB
#   observation/right_wrist_image   – right wrist RGB
#   observation/state               – 14-dim proprio
#   actions                         – (T,14) delta actions (optional)
#   prompt                          – str (optional)
#
# The transform packages these into π₀'s canonical structure.


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


class PiperInputs(transforms.DataTransformFn):
    """Convert dual-arm Piper datapoint → π₀ model input."""

    # Model's padded action dimension (π₀ default 32).
    action_dim: int

    # Model type to decide padding vs masking behaviour.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        # ------------------------------------------------------------------
        # Proprioception (14-DoF) – pad to model action_dim (32)
        # ------------------------------------------------------------------
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # ------------------------------------------------------------------
        # Images – workspace + two wrists. Fallback to zeros if missing.
        # ------------------------------------------------------------------
        base_img = _parse_image(data["observation/image"])

        # *Left* wrist camera may be absent in some episodes – handle gracefully.
        left_img = data.get("observation/left_wrist_image")
        if left_img is not None:
            left_img = _parse_image(left_img)
        else:
            left_img = np.zeros_like(base_img)

        right_img = data.get("observation/right_wrist_image")
        if right_img is not None:
            right_img = _parse_image(right_img)
        else:
            right_img = np.zeros_like(base_img)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_img,
                "left_wrist_0_rgb": left_img,
                "right_wrist_0_rgb": right_img,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # ------------------------------------------------------------------
        # Actions (training-time only)
        # ------------------------------------------------------------------
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim, axis=-1)
            inputs["actions"] = actions

        # Prompt (if present)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


class PiperOutputs(transforms.DataTransformFn):
    """Truncate model action tensor to dual-arm dimensions (14)."""

    action_dim: int = 14

    def __call__(self, data: dict) -> dict:  # type: ignore[override]
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
