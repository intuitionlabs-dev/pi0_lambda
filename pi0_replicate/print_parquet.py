#!/usr/bin/env python3
"""print_parquet_episode.py

Utility script to quickly inspect the contents of a LeRobot/OpenPI episode
*.parquet* file from the command line.

Example
-------
    python print_parquet_episode.py /path/to/episode_000123.parquet

It will:
• Load the file with pandas (fallback to PyArrow if pandas is unavailable).
• Print the first few rows (configurable with --rows).
• If a "state" column exists, show its shape and whether the left- and right-arm
  halves are identical (helps diagnose recording issues).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _load_parquet(path: Path) -> Any:
    """Return pandas.DataFrame or pyarrow.Table."""
    try:
        import pandas as pd  # noqa: F401 – optional dependency
    except ImportError:
        pd = None  # type: ignore[assignment]

    if pd is not None:
        try:
            return pd.read_parquet(path)
        except Exception as e:  # pragma: no cover
            print(f"[pandas backend] Failed: {e}. Falling back to PyArrow…", file=sys.stderr)

    # --- fallback: pyarrow ---
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path)
    except Exception as e:  # pragma: no cover
        sys.exit(f"Unable to load parquet file (no pandas + PyArrow error): {e}")


def _print_state_info(first_row: Any):
    """Pretty-print state vector diagnostics, if present."""
    import numpy as np

    if "state" not in first_row:
        print("[state] column not found – skipped state diagnostics.")
        return

    state_vec = first_row["state"]
    if hasattr(state_vec, "as_py"):
        # pyarrow scalar
        state_vec = state_vec.as_py()

    arr = np.asarray(state_vec, dtype=float)
    print(f"state shape: {arr.shape}")
    if arr.size == 14:
        left, right = arr[:7], arr[7:]
        print("left arm :", np.round(left, 4))
        print("right arm:", np.round(right, 4))
        print("identical halves:", bool(np.allclose(left, right)))
    elif arr.size == 7:
        print("Single-arm recording (7-DoF).")
    else:
        print("Unexpected state dimension:", arr.size)


def main():
    parser = argparse.ArgumentParser(description="Print basic information about a LeRobot episode parquet file.")
    parser.add_argument("parquet", type=Path, help="Path to the *.parquet episode file")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows to display")
    args = parser.parse_args()

    if not args.parquet.is_file():
        sys.exit(f"File not found: {args.parquet}")

    data = _load_parquet(args.parquet)

    # ------------------------------------------------------------------
    # If pandas DataFrame ------------------------------------------------
    # ------------------------------------------------------------------
    if "pandas" in str(type(data)):
        import pandas as pd  # type: ignore

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 180)

        print("\n=== First rows ===")
        print(data.head(args.rows).to_string(index=False))

        print("\n=== Columns ===")
        print(list(data.columns))

        print("\n=== State diagnostics ===")
        _print_state_info(data.iloc[0])
        return

    # ------------------------------------------------------------------
    # If PyArrow Table ---------------------------------------------------
    # ------------------------------------------------------------------
    print("\n=== Schema ===")
    print(data.schema)

    import pyarrow as pa  # type: ignore

    slice_table = data.slice(0, args.rows)
    print("\n=== First rows (converted to pandas) ===")
    print(slice_table.to_pandas().to_string(index=False))

    first_row = slice_table.slice(0, 1)[0] if hasattr(slice_table, "column") else None
    if first_row is not None:
        print("\n=== State diagnostics ===")
        _print_state_info(first_row)


if __name__ == "__main__":
    main() 