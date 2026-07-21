#!/usr/bin/env python3
"""
Extract XRay HDF5 data from CHESS reduced data and convert to JSON format.

Processes strain_full.nxs files from reduced_data/fancher-4630-b/{sample}/output/
Extracts coordinates (labx, labz), strain values, and fit results for all detectors.
Includes optional strain rosette analysis when 3+ detectors available.
"""

import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import scipy.optimize as optimize

# Configuration
REDUCED_DATA_BASE = Path("/home/ntm/projects/nsdf-fabric/chess-experiment/reduced_data/fancher-4630-b")
# Adjust OUTPUT_DIR for scripts/analysis/ subdirectory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "xray"

# Detector angles (in degrees) - from beamline geometry
DETECTOR_ANGLES = {
    0: 0.0,
    1: 10.0,
    5: 50.0,
    8: 80.0,
    11: 110.0,
    12: 120.0,
    13: 130.0,
    14: 140.0,
    16: 160.0,
    17: 170.0,
    18: 180.0,
    19: 190.0,
    21: 210.0,
    22: 220.0,
}


def fit_strain_rosette(
    normal_strains: np.ndarray,
    detector_ids: List[int],
    detector_angles: Dict[int, float]
) -> Optional[Dict]:
    """
    Fit strain rosette (e_xx, e_yy, e_xy) from multi-detector measurements.
    
    Args:
        normal_strains: Array of normal strains from each detector (N_detectors, N_points)
        detector_ids: List of detector IDs with valid data
        detector_angles: Dict mapping detector ID to angle in degrees
        
    Returns:
        Dict with e_xx, e_yy, e_xy arrays (N_points each), or None if fit failed
    """
    if len(detector_ids) < 3:
        return None  # Need at least 3 detectors for rosette
    
    # Extract angles for available detectors
    angles = np.array([detector_angles[det_id] for det_id in detector_ids])
    angles_rad = np.radians(angles)
    
    n_points = normal_strains.shape[1]
    e_xx_values = []
    e_yy_values = []
    e_xy_values = []
    e_xx_errors = []
    e_yy_errors = []
    e_xy_errors = []
    
    for i in range(n_points):
        strains = normal_strains[:, i]
        
        try:
            # Initial guesses
            e_xx_guess = strains[np.argmin(np.abs(angles - 0))]
            e_yy_guess = strains[np.argmin(np.abs(angles - 90))]
            e_xy_guess = 0.0
            
            bounds_scale = 100 * np.max(np.abs(strains))
            bounds = (
                [-bounds_scale, -bounds_scale, -bounds_scale],
                [bounds_scale, bounds_scale, bounds_scale]
            )
            
            # Fit
            popt, pcov = optimize.curve_fit(
                _strain_rosette_calc,
                angles_rad,
                strains,
                p0=(e_xx_guess, e_yy_guess, e_xy_guess),
                bounds=bounds,
                max_nfev=10000
            )
            
            e_xx_values.append(popt[0])
            e_yy_values.append(popt[1])
            e_xy_values.append(popt[2])
            
            perr = np.sqrt(np.diag(pcov))
            e_xx_errors.append(perr[0])
            e_yy_errors.append(perr[1])
            e_xy_errors.append(perr[2])
        except Exception:
            # Fit failed at this point
            e_xx_values.append(np.nan)
            e_yy_values.append(np.nan)
            e_xy_values.append(np.nan)
            e_xx_errors.append(np.nan)
            e_yy_errors.append(np.nan)
            e_xy_errors.append(np.nan)
    
    return {
        "e_xx": {"values": e_xx_values, "errors": e_xx_errors},
        "e_yy": {"values": e_yy_values, "errors": e_yy_errors},
        "e_xy": {"values": e_xy_values, "errors": e_xy_errors},
    }


def _strain_rosette_calc(angle: np.ndarray, e_xx: float, e_yy: float, e_xy: float) -> np.ndarray:
    """Normal strain at angle (radians) given e_xx, e_yy, e_xy."""
    c = np.cos(angle)
    s = np.sin(angle)
    return e_xx * c**2 + e_yy * s**2 + 2.0 * e_xy * c * s


def extract_detector_data(hdf5_file: Path, det_id: int, det_name: str) -> Optional[Dict]:
    """Extract data for a single detector from HDF5."""
    try:
        with h5py.File(hdf5_file, 'r') as f:
            # Find the dataset keys (d1-2_dataset1_strainanalysis, etc.)
            strain_analysis_key = None
            for key in f.keys():
                if key.endswith("_strainanalysis") and str(det_id) in f[key]:
                    strain_analysis_key = key
                    break
            
            if not strain_analysis_key or str(det_id) not in f[strain_analysis_key]:
                return None
            
            det_group = f[strain_analysis_key][str(det_id)]
            
            # Extract data
            data = {
                "data": {
                    "unconstrained_strain": det_group["data/unconstrained_strain"][:].tolist(),
                    "uniform_strain": det_group["data/uniform_strain"][:].tolist(),
                    "unconstrained_strain_stdev": det_group["data/unconstrained_strain_stdev"][:].tolist(),
                },
                "unconstrained_fit": {},
                "uniform_fit": {},
            }
            
            # Extract all Miller indices from unconstrained fits
            if "unconstrained_fit" in det_group:
                for hkl_str in det_group["unconstrained_fit"].keys():
                    if hkl_str == "results":
                        continue
                    hkl_key = hkl_str.replace("_", ",")  # Convert back if needed
                    if "strains" in det_group[f"unconstrained_fit/{hkl_str}"]:
                        strain_vals = det_group[f"unconstrained_fit/{hkl_str}/strains/values"][:]
                        data["unconstrained_fit"][f"{hkl_str}/strains/values"] = strain_vals.tolist()
                
                # Extract fit results
                if "results" in det_group["unconstrained_fit"]:
                    data["unconstrained_fit"]["results/success"] = det_group["unconstrained_fit/results/success"][:].tolist()
            
            # Extract uniform fit info if available
            if "uniform_fit" in det_group:
                if "results" in det_group["uniform_fit"]:
                    data["uniform_fit"]["results/success"] = det_group["uniform_fit/results/success"][:].tolist()
            
            return data
    except Exception as e:
        print(f"  ✗ Error extracting detector {det_id}: {e}")
        return None


def extract_sample(hdf5_file: Path, sample_name: str) -> Optional[Dict]:
    """Extract all data from a sample HDF5 file."""
    print(f"\nProcessing {sample_name}...")
    
    try:
        with h5py.File(hdf5_file, 'r') as f:
            # Find the strain analysis key first
            strain_analysis_key = None
            for key in f.keys():
                if key.endswith("_strainanalysis"):
                    strain_analysis_key = key
                    break
            
            if not strain_analysis_key:
                print(f"  ✗ No strain analysis key found")
                return None
            
            # Find the dataset key (the companion to strainanalysis)
            dataset_key = strain_analysis_key.replace("_strainanalysis", "")
            
            if dataset_key not in f or "data" not in f[dataset_key]:
                print(f"  ✗ Could not find data in dataset key {dataset_key}")
                print(f"    Available keys: {list(f.keys())}")
                return None
            
            # Get coordinates
            dataset_group = f[dataset_key]
            if "data" not in dataset_group:
                print(f"  ✗ No data group found")
                return None
            
            labx = dataset_group["data/labx"][:]
            # Use labz if available, otherwise try fly_labz
            if "labz" in dataset_group["data"]:
                labz = dataset_group["data/labz"][:]
            elif "fly_labz" in dataset_group["data"]:
                labz = dataset_group["data/fly_labz"][:]
            else:
                print(f"  ✗ No labz or fly_labz found")
                return None
            
            n_points = len(labx)
            
            print(f"  Found {n_points} measurement points")
            
            # Initialize output with coordinates
            output = {
                "labx": labx.tolist(),
                "labz": labz.tolist(),
            }
            
            # Extract data from all detectors
            detector_ids = f[strain_analysis_key].attrs.get("detector_ids", [])
            if not detector_ids:
                # Try to find detector IDs from group keys
                detector_ids = [int(k) for k in f[strain_analysis_key].keys() 
                               if k.isdigit()]
            
            print(f"  Found {len(detector_ids)} detectors: {detector_ids}")
            
            # Extract each detector's data
            all_unconstrained_strains = []
            valid_detector_ids = []
            
            for det_id in sorted(detector_ids):
                det_data = extract_detector_data(hdf5_file, det_id, f"det{det_id}")
                
                if det_data:
                    output[f"{det_id}"] = det_data
                    all_unconstrained_strains.append(det_data["data"]["unconstrained_strain"])
                    valid_detector_ids.append(det_id)
                    print(f"  ✓ Detector {det_id}: {len(det_data['data']['unconstrained_strain'])} points")
            
            # Perform rosette analysis if we have 3+ detectors
            if len(valid_detector_ids) >= 3:
                print(f"  Performing strain rosette analysis...")
                strains_array = np.array(all_unconstrained_strains)
                
                rosette_result = fit_strain_rosette(
                    strains_array,
                    valid_detector_ids,
                    DETECTOR_ANGLES
                )
                
                if rosette_result:
                    output["strain_rosette"] = {
                        "e_xx": rosette_result["e_xx"],
                        "e_yy": rosette_result["e_yy"],
                        "e_xy": rosette_result["e_xy"],
                        "detector_ids": valid_detector_ids,
                        "detector_angles": [DETECTOR_ANGLES[det_id] for det_id in valid_detector_ids],
                    }
                    print(f"  ✓ Rosette analysis complete")
            
            return output
            
    except Exception as e:
        print(f"  ✗ Error processing {hdf5_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Extract all xray HDF5 files and save as JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CHESS XRay HDF5 Extraction")
    print("=" * 70)
    
    # Find all strain_full.nxs files
    hdf5_files = sorted(REDUCED_DATA_BASE.glob("*/output*/strain_full.nxs"))
    
    if not hdf5_files:
        print("✗ No HDF5 files found!")
        return 1
    
    print(f"\nFound {len(hdf5_files)} HDF5 files to process:")
    for f in hdf5_files:
        print(f"  {f.relative_to(REDUCED_DATA_BASE.parent)}")
    
    # Extract and save each file
    samples_by_name = {}
    for hdf5_file in hdf5_files:
        # Determine sample name
        sample_dir = hdf5_file.parent.parent.name  # e.g., "d1-2", "d0_2-1"
        
        # Group by base sample name (e.g., d1, d0_2)
        base_name = sample_dir.rsplit("-", 1)[0]  # e.g., "d1" or "d0_2"
        measurement_num = sample_dir.rsplit("-", 1)[1]  # e.g., "2" or "1"
        
        if base_name not in samples_by_name:
            samples_by_name[base_name] = []
        samples_by_name[base_name].append((measurement_num, hdf5_file))
    
    # Process each sample
    extracted_files = {}
    for base_name in sorted(samples_by_name.keys()):
        measurements = sorted(samples_by_name[base_name], key=lambda x: int(x[0]))
        
        for meas_num, hdf5_file in measurements:
            # Extract data
            data = extract_sample(hdf5_file, f"{base_name}-{meas_num}")
            
            if data:
                # Save with numbering
                upper_name = base_name.upper()
                output_file = OUTPUT_DIR / f"{upper_name}_{meas_num}.json"
                
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"  → Saved to {output_file.name}")
                extracted_files[f"{upper_name}_{meas_num}"] = output_file
    
    print("\n" + "=" * 70)
    print(f"✓ Extraction complete! Generated {len(extracted_files)} JSON files")
    for name, path in sorted(extracted_files.items()):
        size_kb = path.stat().st_size / 1024
        print(f"  {name:20s} ({size_kb:6.1f} KB)")
    
    return 0


if __name__ == "__main__":
    exit(main())
