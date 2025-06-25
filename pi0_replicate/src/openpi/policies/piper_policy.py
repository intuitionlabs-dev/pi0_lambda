"""Piper dual-arm policy I/O transforms for π₀ models.

These transforms map raw Piper LeRobot dataset keys to the inputs/outputs expected
by π₀ (and π₀-FAST) models.

Input expectations (before transform):
    workspace_image        – RGB uint8 (H,W,3)          third-person view
    left_wrist_image       – RGB uint8 (H,W,3)
    right_wrist_image      – RGB uint8 (H,W,3)
    state                  – float32 (14,)              concatenated left+right joint angles
    actions                – float32 (T,14)             Δ joint actions (optional, training only)
    prompt                 – str                         natural-language task (optional if injected via DataConfig)

The transforms will package these into π₀'s canonical dict structure:
    image / image_mask      – dict of 3 cams (base + two wrists)
    state                  – zero-padded to `action_dim`
    actions                – zero-padded to `action_dim` (if present)
    prompt                 – unchanged

Outputs simply truncate the first 14 dimensions of the predicted action tensor.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Format Piper observations for π₀."""

    # Target action dimension of the model (π₀ default 32). The transform pads
    # state / action arrays with zeros if they are shorter than this number.
    action_dim: int

    # Camera mapping from dataset → model input names
    CAM_MAP: ClassVar[dict[str, str]] = {
        # dest_key         : src_key
        "base_0_rgb": "workspace_image",
        "left_wrist_0_rgb": "left_wrist_image",
        "right_wrist_0_rgb": "right_wrist_image",
    }

    def __call__(self, data: dict) -> dict:
        # ------------------------------------------------------------------
        # Proprioception
        # ------------------------------------------------------------------
        state = np.asarray(data["state"], dtype=np.float32)
        state = transforms.pad_to_dim(state, self.action_dim)

        # ------------------------------------------------------------------
        # Images
        # Convert to dict of expected keys + masks.
        # Images are already (H,W,3) uint8 from recorder.
        # ------------------------------------------------------------------
        images = {}
        image_masks = {}

        # Loop over the expected destination → source key mapping. For the standard dual-arm
        # dataset the mapping will resolve directly. For *single-arm* recordings we often only
        # have a single wrist camera named ``wrist_image`` instead of separate left / right
        # wrist topics. In that case we fall back to using ``wrist_image`` for the left wrist
        # view and treat the (missing) right wrist view as padding.

        for dst_key, src_key in self.CAM_MAP.items():
            img = data.get(src_key)

            # Fallback: if the left wrist image is missing but we have a generic "wrist_image"
            # (common for single-arm Piper setups) use that instead.
            if img is None and src_key == "left_wrist_image":
                img = data.get("wrist_image")

            # No fallback for the right wrist – if it's absent we will pad with zeros below.

            # ----------------------------------------------------------------
            # Ensure we work with NumPy arrays throughout the transform.  The
            # LeRobot video decoders may yield `torch.Tensor`s which do *not*
            # satisfy the `isinstance(v, np.ndarray)` check used later when we
            # look for a reference image to pad missing views.  Converting the
            # tensor to a NumPy array *and* writing the result back to
            # `data[src_key]` guarantees that the reference lookup succeeds
            # regardless of the backend that produced the frames.
            # ----------------------------------------------------------------
            if img is not None and not isinstance(img, np.ndarray):
                img = np.asarray(img)

            # Make the converted image visible to subsequent iterations /
            # fallback logic.
            if img is not None:
                data[src_key] = img

            # ----------------------------------------------------------------
            # If the video decoder returned channel-first (C,H,W) arrays, convert
            # them to channel-last (H,W,C).  We heuristically decide that the
            # frame is channel-first when *both* of the following hold:
            #   • the first dimension is tiny (≤4) – plausible channel count
            #   • the last dimension is large (>4) – plausible spatial extent
            # This avoids mistakenly transposing already channel-last RGB frames
            # where height also happens to be ≤4 (e.g. 1×1 thumbnails).
            # ----------------------------------------------------------------
            if img is not None and getattr(img, "ndim", 0) == 3:
                c, h, w = img.shape[0], img.shape[1], img.shape[2]
                if c <= 4 and img.shape[-1] > 4:
                    img = np.moveaxis(img, 0, -1)

                # Handle single-channel images returned as (H,W,1); replicate
                # to RGB so that downstream PIL routines accept them.
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)

            if img is None:
                # Fill missing view with black image matching the first available view
                ref = next(v for v in data.values() if isinstance(v, np.ndarray) and v.ndim == 3)
                img = np.zeros_like(ref)
                image_masks[dst_key] = np.False_
            else:
                # Ensure uint8
                img = np.asarray(img)
                if img.dtype != np.uint8:
                    img = (255 * img).astype(np.uint8)
                image_masks[dst_key] = np.True_
            images[dst_key] = img

        # ------------------------------------------------------------------
        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        # ------------------------------------------------------------------
        # Actions (only during training)
        # ------------------------------------------------------------------
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            actions = transforms.pad_to_dim(actions, self.action_dim, axis=-1)
            inputs["actions"] = actions

        # Prompt (if present)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Reduce model output to Piper action dimensions."""

    action_dim: int = 14  # first 14 dims correspond to Piper joints

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, : self.action_dim]} 