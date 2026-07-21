#!/usr/bin/env python3
"""
Extract CHESS neutron data from HDF5 files and convert to JSON format.

Reads strain_full.nxs files, extracts detector measurements, performs
strain rosette analysis, and outputs unified JSON files compatible with
plotting and analysis scripts.
"""

import json
import h5py
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
from scipy.optimize import curve_fit

# Add parent directory to path to import rosette_math from reduced_data
SCRIPT_DIR = Path(__file__).parent
REDUCED_DATA_DIR = SCRIPT_DIR.parent / "reduced_data" / "fancher-4630-b"


def load_detector_config(config_path: Path) -> Dict:
    """Load detector configuration with angles."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def strain_rosette_calc(angle, e_xx, e_yy, e_xy):
    """Calculate normal strain at angle (radians) given strain components."""
    a = np.asarray(angle)
    c = np.cos(a)
    s = np.sin(a)
    return e_xx * c**2 + e_yy * s**2 + 2.0 * e_xy * c * s


def fit_strain_rosette(normal_strain, det_angles):
    """Fit strain rosette to measured normal strains at detector angles."""
    normal_strain = np.asarray(normal_strain)
    det_angles = np.asarray(det_angles)
    det_angles_rad = np.radians(det_angles)

    # Initial guesses from closest detectors
    e_xx_guess = normal_strain[np.argmin(np.abs(det_angles - 0))]
    e_yy_guess = normal_strain[np.argmin(np.abs(det_angles - 90))]
    e_xy_guess = 0.0

    bounds_guess = 100 * np.max(np.abs(normal_strain))
    bounds = (
        [-bounds_guess, -bounds_guess, -bounds_guess],
        [ bounds_guess,  bounds_guess,  bounds_guess]
    )

    try:
        popt, pcov = curve_fit(
            strain_rosette_calc,
            det_angles_rad,
            normal_strain,
            p0=(e_xx_guess, e_yy_guess, e_xy_guess),
            bounds=bounds,
            max_nfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, True
    except Exception as e:
        print(f"  Warning: Rosette fit failed: {e}")
        return np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), False


def extract_hdf5_to_dict(
    h5_file: Path,
    detector_config: Dict,
    sample_name: str,
    include_rosette: bool = True
) -> Dict:
    """Extract data from HDF5 file and return as dictionary."""
    
    data_dict = {}
    
    with h5py.File(h5_file, 'r') as f:
        # Find the strainanalysis group
        strain_group_key = None
        for key in f.keys():
            if 'strainanalysis' in key:
                strain_group_key = key
                break
        
        if not strain_group_key:
            raise ValueError(f"No strainanalysis group found in {h5_file}")
        
        strain_analysis = f[strain_group_key]
        
        # Get detector IDs present in this file
        det_ids = sorted([int(k) for k in strain_analysis.keys() if k.isdigit()])
        
        if not det_ids:
            raise ValueError(f"No detectors found in {strain_group_key}")
        
        # Extract coordinates from first detector
        first_det = str(det_ids[0])
        lab_x = np.array(strain_analysis[first_det]['data']['labx'])
        lab_z = np.array(strain_analysis[first_det]['data']['labz'])
        
        data_dict['labx'] = lab_x.tolist()
        data_dict['labz'] = lab_z.tolist()
        
        n_points = len(lab_x)
        
        # Extract data from each detector
        det_angles = []
        det_strains = []
        det_strains_err = []
        
        for det_id in det_ids:
            det_key = str(det_id)
            if det_key not in strain_analysis:
                print(f"  Warning: Detector {det_id} not found")
                continue
            
            det_group = strain_analysis[det_key]
            
            # Get detector angle from config
            if det_id in detector_config['detectors']:
                eta = detector_config['detectors'][det_id]['eta']
                det_angles.append(eta)
            else:
                print(f"  Warning: No angle config for detector {det_id}")
                continue
            
            # Extract strain measurements
            unconstrained_strain = np.array(det_group['data']['unconstrained_strain'])
            unconstrained_error = np.array(det_group['data']['unconstrained_strain_stdev'])
            uniform_strain = np.array(det_group['data']['uniform_strain'])
            
            det_strains.append(unconstrained_strain)
            det_strains_err.append(unconstrained_error)
            
            # Store detector data
            det_key_path = f'{det_id}/data/unconstrained_strain'
            data_dict[det_key_path] = unconstrained_strain.tolist()
            
            det_key_path = f'{det_id}/data/unconstrained_strain_stdev'
            data_dict[det_key_path] = unconstrained_error.tolist()
            
            det_key_path = f'{det_id}/data/uniform_strain'
            data_dict[det_key_path] = uniform_strain.tolist()
            
            # Extract all available fit reflections
            if 'unconstrained_fit' in det_group:
                for hkl_key in det_group['unconstrained_fit'].keys():
                    if hkl_key != 'results':
                        try:
                            hkl_group = det_group['unconstrained_fit'][hkl_key]
                            if 'strains' in hkl_group and 'values' in hkl_group['strains']:
                                strains = np.array(hkl_group['strains']['values'])
                                fit_key = f'{det_id}/unconstrained_fit/{hkl_key}/strains/values'
                                data_dict[fit_key] = strains.tolist()
                        except Exception as e:
                            print(f"  Warning: Could not extract {hkl_key} for detector {det_id}: {e}")
        
        # Perform rosette analysis if requested and we have enough detectors
        if include_rosette and len(det_angles) >= 3:
            print(f"  Performing strain rosette analysis with {len(det_angles)} detectors...")
            
            det_angles = np.array(det_angles)
            det_strains = np.array(det_strains)
            
            e_xx = np.zeros(n_points)
            e_yy = np.zeros(n_points)
            e_xy = np.zeros(n_points)
            e_xx_err = np.zeros(n_points)
            e_yy_err = np.zeros(n_points)
            e_xy_err = np.zeros(n_points)
            fit_success = np.zeros(n_points, dtype=bool)
            
            for i in range(n_points):
                popt, perr, success = fit_strain_rosette(det_strains[:, i], det_angles)
                e_xx[i] = popt[0]
                e_yy[i] = popt[1]
                e_xy[i] = popt[2]
                e_xx_err[i] = perr[0]
                e_yy_err[i] = perr[1]
                e_xy_err[i] = perr[2]
                fit_success[i] = success
            
            # Store rosette results
            data_dict['strain_rosette/e_xx/values'] = e_xx.tolist()
            data_dict['strain_rosette/e_yy/values'] = e_yy.tolist()
            data_dict['strain_rosette/e_xy/values'] = e_xy.tolist()
            data_dict['strain_rosette/e_xx/errors'] = e_xx_err.tolist()
            data_dict['strain_rosette/e_yy/errors'] = e_yy_err.tolist()
            data_dict['strain_rosette/e_xy/errors'] = e_xy_err.tolist()
            data_dict['strain_rosette/fit_success'] = fit_success.tolist()
    
    return data_dict


def process_sample(
    sample_name: str,
    reduced_data_dir: Path,
    output_dir: Path,
    detector_config: Dict,
    include_rosette: bool = True
) -> bool:
    """Process a single sample and save to JSON."""
    
    h5_file = reduced_data_dir / sample_name / "output" / "strain_full.nxs"
    
    if not h5_file.exists():
        print(f"✗ File not found: {h5_file}")
        return False
    
    print(f"\nProcessing: {sample_name}")
    print(f"  Reading: {h5_file}")
    
    try:
        data = extract_hdf5_to_dict(h5_file, detector_config, sample_name, include_rosette)
        
        # Save to JSON
        output_file = output_dir / f"{sample_name}.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  ✓ Saved: {output_file}")
        print(f"    Data points: {len(data['labx'])}")
        print(f"    Keys: {len(data)}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error processing {sample_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Extract all neutron samples to JSON."""
    
    # Configuration
    config_file = SCRIPT_DIR.parent / "detector_config.yaml"
    output_dir = SCRIPT_DIR.parent / "data" / "neutron"
    
    # Samples to process (auto-detect from reduced_data directory)
    samples = []
    for sample_dir in REDUCED_DATA_DIR.iterdir():
        if sample_dir.is_dir():
            output_dir_path = sample_dir / "output"
            if (output_dir_path / "strain_full.nxs").exists():
                samples.append(sample_dir.name)
    
    samples = sorted(samples)
    
    print("=" * 70)
    print("CHESS HDF5 to JSON Extraction")
    print("=" * 70)
    
    # Load detector configuration
    if not config_file.exists():
        print(f"✗ Config file not found: {config_file}")
        return 1
    
    print(f"Loading detector config: {config_file}")
    detector_config = load_detector_config(config_file)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Process samples
    successful = 0
    failed = 0
    
    for sample in samples:
        if process_sample(sample, REDUCED_DATA_DIR, output_dir, detector_config):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Extraction complete: {successful} successful, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
