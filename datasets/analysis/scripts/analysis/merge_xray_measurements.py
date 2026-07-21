#!/usr/bin/env python3
"""
Merge individual XRay measurements into complete sample datasets.

Combines multiple measurement files for each sample (D1_2 + D1_3, D2_2 + legacy)
into single merged datasets (D1_full.json, D2_full.json).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Adjust paths for scripts/analysis/ subdirectory
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "xray"


def load_json(filepath: Path) -> Dict:
    """Load JSON data file."""
    with open(filepath) as f:
        return json.load(f)


def save_json(data: Dict, filepath: Path) -> None:
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def merge_measurements(files: List[Path], sample_name: str) -> Dict:
    """
    Merge multiple measurement files into a single dataset.
    
    Combines coordinates and detector data from all measurements.
    """
    print(f"\nMerging {sample_name}:")
    
    all_data = []
    for filepath in files:
        data = load_json(filepath)
        all_data.append(data)
        print(f"  • {filepath.name}: {len(data['labx'])} points")
    
    # Get detector IDs from first file
    detector_ids = [k for k in all_data[0].keys() if k not in ['labx', 'labz', 'strain_rosette']]
    detector_ids = sorted([int(d) for d in detector_ids if d.isdigit()])
    
    print(f"  Found {len(detector_ids)} detectors: {detector_ids}")
    
    # Combine coordinates
    all_labx = []
    all_labz = []
    file_boundaries = [0]  # Track where each file's data starts
    
    for data in all_data:
        all_labx.extend(data['labx'])
        all_labz.extend(data['labz'])
        file_boundaries.append(len(all_labx))
    
    total_points = len(all_labx)
    print(f"  Total points: {total_points}")
    
    # Initialize merged output
    merged = {
        "labx": all_labx,
        "labz": all_labz,
    }
    
    # Merge detector data
    for det_id in detector_ids:
        det_id_str = str(det_id)
        merged[det_id_str] = {}
        
        # Initialize arrays for this detector
        det_unconstrained_strain = []
        det_uniform_strain = []
        det_unconstrained_stdev = []
        det_success_list = []
        
        # Collect data from each file
        for i, data in enumerate(all_data):
            if det_id_str not in data:
                continue
            
            det_data = data[det_id_str]
            
            if "data" in det_data:
                det_unconstrained_strain.extend(det_data["data"]["unconstrained_strain"])
                det_uniform_strain.extend(det_data["data"]["uniform_strain"])
                det_unconstrained_stdev.extend(det_data["data"]["unconstrained_strain_stdev"])
            
            if "unconstrained_fit" in det_data and "results/success" in det_data["unconstrained_fit"]:
                det_success_list.extend(det_data["unconstrained_fit"]["results/success"])
        
        # Store merged data for this detector
        if det_unconstrained_strain:
            merged[det_id_str]["data"] = {
                "unconstrained_strain": det_unconstrained_strain,
                "uniform_strain": det_uniform_strain,
                "unconstrained_strain_stdev": det_unconstrained_stdev,
            }
        
        # Store fit results
        if det_success_list:
            if "unconstrained_fit" not in merged[det_id_str]:
                merged[det_id_str]["unconstrained_fit"] = {}
            merged[det_id_str]["unconstrained_fit"]["results/success"] = det_success_list
        
        # Merge Miller indices fits (all HKL values from unconstrained_fit)
        # Note: HKL keys like "3_1_1/strains/values" map directly to lists in individual files
        for data in all_data:
            if det_id_str not in data or "unconstrained_fit" not in data[det_id_str]:
                continue
            
            fit_data = data[det_id_str]["unconstrained_fit"]
            for hkl_key, hkl_value in fit_data.items():
                if hkl_key == "results/success":
                    continue  # Already handled above
                
                # HKL keys are like "3_1_1/strains/values" and map directly to lists
                if isinstance(hkl_value, list):
                    if hkl_key not in merged[det_id_str]["unconstrained_fit"]:
                        merged[det_id_str]["unconstrained_fit"][hkl_key] = []
                    # Extend the list with values from this file
                    merged[det_id_str]["unconstrained_fit"][hkl_key].extend(hkl_value)
    
    # Merge rosette analysis if present
    if "strain_rosette" in all_data[0]:
        print(f"  Merging rosette analysis...")
        merged["strain_rosette"] = {
            "e_xx": {"values": [], "errors": []},
            "e_yy": {"values": [], "errors": []},
            "e_xy": {"values": [], "errors": []},
            "detector_ids": all_data[0]["strain_rosette"].get("detector_ids", []),
            "detector_angles": all_data[0]["strain_rosette"].get("detector_angles", []),
        }
        
        for data in all_data:
            if "strain_rosette" in data:
                rosette = data["strain_rosette"]
                for component in ["e_xx", "e_yy", "e_xy"]:
                    if component in rosette:
                        merged["strain_rosette"][component]["values"].extend(
                            rosette[component].get("values", [])
                        )
                        merged["strain_rosette"][component]["errors"].extend(
                            rosette[component].get("errors", [])
                        )
    
    return merged


def main():
    """Merge all XRay measurements."""
    print("=" * 70)
    print("XRay Measurement Merging")
    print("=" * 70)
    
    # Define merge groups
    merge_groups = {
        "D1_full.json": ["D1_2.json", "D1_3.json"],
        "D2_full.json": ["D2_2.json"],  # Just D2_2; D2_xray is legacy
    }
    
    for output_name, input_files in merge_groups.items():
        input_paths = [DATA_DIR / fname for fname in input_files]
        
        # Check all files exist
        missing = [p for p in input_paths if not p.exists()]
        if missing:
            print(f"\n✗ Missing files for {output_name}:")
            for p in missing:
                print(f"  {p.name}")
            continue
        
        # Merge
        merged_data = merge_measurements(input_paths, output_name)
        
        # Save
        output_path = DATA_DIR / output_name
        save_json(merged_data, output_path)
        
        size_kb = output_path.stat().st_size / 1024
        print(f"  → Saved: {output_name} ({size_kb:.1f} KB)")
    
    print("\n" + "=" * 70)
    print("✓ Merging complete!")
    print("\nGenerated merged files:")
    for fname in ["D1_full.json", "D2_full.json"]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"  {fname:20s} ({size_kb:7.1f} KB)")
    
    return 0


if __name__ == "__main__":
    exit(main())
