#!/usr/bin/env python3
"""Export selected arrays from strain_map.nxs into flat JSON stream-results shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np


class FileLayout(NamedTuple):
    """Describes the HDF5 layout for a specific file type."""

    entry: str
    strain_entry: str
    detector: str
    hkl: str
    uniform_strain_field: str
    unconstrained_strain_field: str


# X-ray AMBench layout (strain_map_ambench.nxs)
AMBENCH_LAYOUT = FileLayout(
    entry="v8-p3-10s-0deg_dataset1",
    strain_entry="v8-p3-10s-0deg_dataset1_strainanalysis",
    detector="0",
    hkl="2_2_2",
    uniform_strain_field="uniform_microstrain",
    unconstrained_strain_field="unconstrained_microstrain",
)

# Neutron AMMDF/VULCAN layout (strain_map_ammdf.nxs)
AMMDF_LAYOUT = FileLayout(
    entry="d1-2_dataset1",
    strain_entry="d1-2_dataset1_strainanalysis",
    detector="0",
    hkl="2_2_2",
    uniform_strain_field="uniform_strain",
    unconstrained_strain_field="unconstrained_strain",
)


def _detect_layout(h5: h5py.File) -> FileLayout:
    """Auto-detect the file layout based on top-level group names."""
    keys = set(h5.keys())
    if AMMDF_LAYOUT.entry in keys:
        return AMMDF_LAYOUT
    if AMBENCH_LAYOUT.entry in keys:
        return AMBENCH_LAYOUT
    raise ValueError(f"Unknown file layout. Top-level groups: {list(keys)}")


def _to_json_numbers(arr: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in np.asarray(arr, dtype=np.float64).tolist():
        if value is None or np.isnan(value):
            out.append(None)
        else:
            out.append(float(value))
    return out


def _resolve_default_input() -> Path:
    candidates = [Path("datasets/strain_map.small.nxs"), Path("strain_map.nxs")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def export(input_file: Path, output_file: Path) -> None:
    with h5py.File(input_file, "r") as h5:
        layout = _detect_layout(h5)

        labx = _to_json_numbers(h5[f"{layout.entry}/data/labx"][...])
        labz = _to_json_numbers(h5[f"{layout.entry}/data/labz"][...])
        uniform_strain = _to_json_numbers(
            h5[f"{layout.strain_entry}/{layout.detector}/data/{layout.uniform_strain_field}"][...]
        )
        unconstrained_strain = _to_json_numbers(
            h5[f"{layout.strain_entry}/{layout.detector}/data/{layout.unconstrained_strain_field}"][...]
        )
        unconstrained_centers = _to_json_numbers(
            h5[f"{layout.strain_entry}/{layout.detector}/unconstrained_fit/{layout.hkl}/centers/values"][...]
        )

    payload = {
        "labx": labx,
        "labz": labz,
        # Alias strain arrays to the keys expected by JSON monitor defaults/tests.
        "0/data/uniform_strain": uniform_strain,
        "0/data/unconstrained_strain": unconstrained_strain,
        f"0/unconstrained_fit/{layout.hkl}/centers/values": unconstrained_centers,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=_resolve_default_input(),
        help="Input Nexus/HDF5 file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/strain_map.reduced.json"),
        help="Output JSON file",
    )
    args = parser.parse_args()

    export(args.input, args.output)
    print(f"Wrote {args.output}")
