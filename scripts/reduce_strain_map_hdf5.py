#!/usr/bin/env python3
"""Create a deterministic reduced HDF5 fixture from a full strain_map file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _copy_attrs(src: h5py.Dataset | h5py.Group | h5py.File, dst: h5py.Dataset | h5py.Group | h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _reduced_maxshape(maxshape: tuple[int | None, ...], new_first_dim: int) -> tuple[int | None, ...]:
    if not maxshape:
        return maxshape
    first = maxshape[0]
    if first is None:
        new_first = None
    else:
        new_first = min(first, new_first_dim)
    return (new_first, *maxshape[1:])


def _copy_group(src_group: h5py.Group, dst_group: h5py.Group, max_rows: int) -> None:
    for name, obj in src_group.items():
        if isinstance(obj, h5py.Group):
            child = dst_group.create_group(name)
            _copy_attrs(obj, child)
            _copy_group(obj, child, max_rows)
            continue

        if obj.shape and obj.ndim >= 1 and obj.shape[0] > max_rows:
            data = obj[0:max_rows, ...]
        else:
            data = obj[...]

        create_kwargs: dict[str, object] = {
            "dtype": obj.dtype,
        }

        if obj.shape:
            create_kwargs["maxshape"] = _reduced_maxshape(obj.maxshape, data.shape[0])
            if obj.chunks is not None:
                chunks = list(obj.chunks)
                chunks[0] = min(chunks[0], max(1, data.shape[0]))
                create_kwargs["chunks"] = tuple(chunks)

        compression = obj.compression
        if compression:
            create_kwargs["compression"] = compression
            if obj.compression_opts is not None:
                create_kwargs["compression_opts"] = obj.compression_opts
        else:
            # Use gzip for fixture compactness when the source was uncompressed.
            if obj.shape:
                create_kwargs["compression"] = "gzip"
                create_kwargs["compression_opts"] = 4

        if obj.fillvalue is not None:
            create_kwargs["fillvalue"] = obj.fillvalue

        ds = dst_group.create_dataset(name, data=data, **create_kwargs)
        _copy_attrs(obj, ds)


def reduce_file(input_file: Path, output_file: Path, max_rows: int) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_file, "r") as src, h5py.File(output_file, "w", libver="latest") as dst:
        _copy_attrs(src, dst)
        _copy_group(src, dst, max_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("strain_map.nxs"),
        help="Input full-size strain_map Nexus/HDF5 path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/strain_map.small.nxs"),
        help="Output reduced fixture path",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=64,
        help="Maximum number of rows to keep for datasets where axis 0 is sample-indexed",
    )
    args = parser.parse_args()

    if args.max_rows <= 0:
        raise ValueError("--max-rows must be a positive integer")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    reduce_file(args.input, args.output, args.max_rows)
    print(f"Wrote reduced fixture to {args.output}")


if __name__ == "__main__":
    main()
