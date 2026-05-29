#!/usr/bin/env python3
"""Export selected arrays from strain_map.nxs into flat JSON stream-results shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ENTRY = "v8-p3-10s-0deg_dataset1"
STRAIN_ENTRY = "v8-p3-10s-0deg_dataset1_strainanalysis"
DETECTOR = "0"
HKL = "2_2_2"


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
        labx = _to_json_numbers(h5[f"{ENTRY}/data/labx"][...])
        labz = _to_json_numbers(h5[f"{ENTRY}/data/labz"][...])
        uniform_microstrain = _to_json_numbers(
            h5[f"{STRAIN_ENTRY}/{DETECTOR}/data/uniform_microstrain"][...]
        )
        unconstrained_microstrain = _to_json_numbers(
            h5[f"{STRAIN_ENTRY}/{DETECTOR}/data/unconstrained_microstrain"][...]
        )
        unconstrained_centers = _to_json_numbers(
            h5[f"{STRAIN_ENTRY}/{DETECTOR}/unconstrained_fit/{HKL}/centers/values"][...]
        )

    payload = {
        "labx": labx,
        "labz": labz,
        # Alias microstrain arrays to the keys expected by JSON monitor defaults/tests.
        "0/data/uniform_strain": uniform_microstrain,
        "0/data/unconstrained_strain": unconstrained_microstrain,
        "0/unconstrained_fit/2_2_2/centers/values": unconstrained_centers,
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
