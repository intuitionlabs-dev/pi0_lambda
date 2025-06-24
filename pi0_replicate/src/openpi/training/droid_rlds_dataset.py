from __future__ import annotations

"""Minimal stub for DROID RLDS dataset support.

The real `DroidRldsDataset` implementation is not needed when running Piper
configs but `openpi.training.config` imports the symbol unconditionally.  To
avoid an ImportError when that module is missing we provide a lightweight
fallback that raises a clear error if someone actually tries to instantiate
it.
"""

import enum
from typing import Iterator, Any


class DroidActionSpace(str, enum.Enum):
    """Enumeration of action space types expected by the full implementation."""

    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"


class DroidRldsDataset:  # noqa: D101 – stub only
    def __init__(self, *args: Any, **kwargs: Any):
        msg = (
            "DroidRldsDataset stub: the full DROID RLDS reader is not bundled in this "
            "repository checkout.  If you actually need it install the extra "
            "dependencies and copy the full implementation from upstream openpi."
        )
        raise RuntimeError(msg)

    # Provide the iterable/dataset interface to avoid AttributeError if the
    # class gets inspected somewhere without being instantiated.
    def __iter__(self) -> Iterator:
        raise StopIteration

    def __len__(self) -> int:  # pragma: no cover
        return 0 