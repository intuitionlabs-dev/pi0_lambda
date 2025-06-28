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
import functools

from openpi import transforms


# -----------------------------------------------------------------------------
# Piper joint sign conventions
#   Indices needing sign flip (per single arm):
#     1 – elbow lift
#     2 – shoulder forward/back
#     3 – wrist roll/twist
#   Gripper (6) is unchanged.
# -----------------------------------------------------------------------------


def _joint_flip_mask(action_dim: int) -> np.ndarray:
    """Return a sign mask (+1 or -1) of length *action_dim*.

    For single-arm (7-DoF) Piper recordings the mask is
        [ 1, -1, -1, 1, 1, 1, 1 ]
    and for dual-arm (14-DoF) we concatenate the mask twice.
    """

    single = np.array([1, -1, -1, 1, 1, 1, 1], dtype=np.float32)
    if action_dim == 7:
        return single
    if action_dim == 14:
        return np.concatenate([single, single], axis=0)
    raise ValueError(f"Unsupported Piper action_dim {action_dim}; expected 7 or 14")


# -----------------------------------------------------------------------------
# Gripper conversion helpers (reuse constants from Aloha)
# -----------------------------------------------------------------------------


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    """Map linear gripper position (Aloha convention) to angular radians (π₀)."""
    # These numbers are copied from Aloha policy – adjust if Piper differs.
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    def linear_to_radian(linear_position, arm_length, horn_radius):
        val = (horn_radius ** 2 + linear_position ** 2 - arm_length ** 2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(val, -1.0, 1.0))

    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    """Inverse of _gripper_to_angular for outputs."""
    value = _unnormalize(value, min_val=0.4, max_val=1.5)
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)


# ----------------------------------------------------------------------------
# Input transform (dataset -> model)
# ----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Format Piper observations for π₀."""

    # Target action dimension of the model (π₀ default 32). The transform pads
    # state / action arrays with zeros if they are shorter than this number.
    action_dim: int

    # Whether to flip joint signs to match π₀ internal convention.
    adapt_to_pi: bool = True

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
        # print("using new piper_policy!!")
        state = np.asarray(data["state"], dtype=np.float32).copy()

        # Apply sign correction if requested
        if self.adapt_to_pi:
            mask = _joint_flip_mask(min(len(state), 14))
            state[: len(mask)] = mask * state[: len(mask)]
            # Gripper indices 6 and 13 (if present)
            if len(state) >= 7:
                state[6] = _gripper_to_angular(state[6])
            if len(state) >= 14:
                state[13] = _gripper_to_angular(state[13])

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
            actions = np.asarray(data["actions"], dtype=np.float32).copy()
            if self.adapt_to_pi:
                mask = _joint_flip_mask(min(actions.shape[-1], 14))
                actions[..., : len(mask)] = actions[..., : len(mask)] * mask
                # gripper columns
                if actions.shape[-1] >= 7:
                    actions[..., 6] = _gripper_from_angular_inv(actions[..., 6])
                if actions.shape[-1] >= 14:
                    actions[..., 13] = _gripper_from_angular_inv(actions[..., 13])
            actions = transforms.pad_to_dim(actions, self.action_dim, axis=-1)
            inputs["actions"] = actions

        # Prompt (if present)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


# ----------------------------------------------------------------------------
# Output transform (model -> robot)
# ----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Reduce model output to Piper action dimensions."""

    action_dim: int = 14  # first 14 dims correspond to Piper joints
    adapt_to_pi: bool = True  # convert back to firmware convention

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        acts = actions[:, : self.action_dim]
        if self.adapt_to_pi:
            mask = _joint_flip_mask(self.action_dim)
            acts = acts * mask
            # Convert gripper back
            if self.action_dim >= 7:
                acts[:, 6] = _gripper_from_angular(acts[:, 6])
            if self.action_dim >= 14:
                acts[:, 13] = _gripper_from_angular(acts[:, 13])
        return {"actions": acts} 