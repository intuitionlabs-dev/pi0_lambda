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
        for dst_key, src_key in self.CAM_MAP.items():
            img = data.get(src_key)
            # ----------------------------------------------------------------
            # Convert to NumPy early. This ensures we avoid calling PyTorch's
            # `transpose` when the decoded frames are `torch.Tensor`s (e.g. if
            # the LeRobot dataset video decoder returns tensors in CHW
            # format). Converting first lets `np.moveaxis` operate on a real
            # ndarray and prevents the "transpose() received an invalid
            # combination of arguments - got (list)" TypeError raised by
            # PyTorch when given a Python list of axes.
            # ----------------------------------------------------------------
            if img is not None and not isinstance(img, np.ndarray):
                img = np.asarray(img)

            # ----------------------------------------------------------------
            # If the video decoder returned channel-first (C,H,W) tensors or
            # arrays, flip them to standard (H,W,C) so downstream PIL resize
            # works. We check `ndim` after the possible conversion above.
            # ----------------------------------------------------------------
            if img is not None and getattr(img, "ndim", 0) == 3 and img.shape[0] in (1, 3, 4):
                # Assume channel-first; move to channel-last
                img = np.moveaxis(img, 0, -1)

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