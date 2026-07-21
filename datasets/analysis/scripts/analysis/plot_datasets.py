#!/usr/bin/env python3
"""
Plot CHESS experiment datasets with multi-detector support.

Comprehensive visualization tool supporting:
- Single detector (neutron) scatter plots
- Multi-detector (xray) with error bars and rosette analysis
- Legacy deployment datasets and new extracted XRay data
- Interactive matplotlib plots (zoom, pan, inspect values)
- Selectable HKL/Miller indices for XRay data

Usage:
    python plot_datasets.py                           # Auto-detect & plot all available data
    python plot_datasets.py --hkl 3_1_1              # Plot XRay with 3,1,1 reflection (matches neutron)
    python plot_datasets.py --hkl 2_2_2              # Plot XRay with 2,2,2 reflection (default)
    python plot_datasets.py --plot                   # Interactive matplotlib windows
    python plot_datasets.py --legacy                 # Plot legacy deployment datasets only
    python plot_datasets.py --enhanced --hkl 3_1_1   # Enhanced mode with custom HKL
    python plot_datasets.py --help                   # Show all options
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Dataset paths - analysis/data (new extracted data)
ANALYSIS_DATA_DIR = Path(__file__).parent.parent.parent / "data"
ANALYSIS_NEUTRON_DIR = ANALYSIS_DATA_DIR / "neutron"
ANALYSIS_XRAY_DIR = ANALYSIS_DATA_DIR / "xray"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"

# Dataset paths - deployment (legacy data)
DEPLOYMENT_DATA_DIR = Path(__file__).parent.parent.parent / "intersect-chess-deployment" / "datasets"

# Neutron files (both sources have same names)
NEUTRON_FILES = {
    "A2": ANALYSIS_NEUTRON_DIR / "A2_full.json",
    "D1": ANALYSIS_NEUTRON_DIR / "D1_full.json",
    "D2": ANALYSIS_NEUTRON_DIR / "D2_full.json",
    "D4": ANALYSIS_NEUTRON_DIR / "D4_full.json",
}

# New extracted XRay files (multi-detector)
XRAY_MULTIDET_FILES = {
    "D1_full": ANALYSIS_XRAY_DIR / "D1_full.json",
    "D2_full": ANALYSIS_XRAY_DIR / "D2_full.json",
}

# Legacy deployment files
DEPLOYMENT_NEUTRON_FILES = {
    "A2": DEPLOYMENT_DATA_DIR / "A2_2p1_30layers_VDriveSPF_LD.json",
    "D1": DEPLOYMENT_DATA_DIR / "D1_1p8_30layers_VDriveSPF_LD.json",
    "D2": DEPLOYMENT_DATA_DIR / "D2_2p3_30layers_VDriveSPF_LD.json",
    "D4": DEPLOYMENT_DATA_DIR / "D4_4p4_30layers_VDriveSPF_LD.json",
}
DEPLOYMENT_XRAY_FILE = DEPLOYMENT_DATA_DIR / "strain_map_xray_ammdf_d2_smb.reduced.json"

# Global flag for interactive display
INTERACTIVE_PLOT = False

# Global flag for HKL selection (Miller indices)
XRAY_HKL = "3_1_1"  # Default to 3,1,1 to match neutron data

# Global flag for autoscaling plots (True = use global min/max, False = individual plot ranges)
AUTO_SCALE_PLOTS = True

# Global min/max for consistent scaling across plots
PLOT_VMIN = None
PLOT_VMAX = None


def format_hkl(hkl_str):
    """Convert HKL notation from underscore to comma format (e.g., '3_1_1' -> '3,1,1')."""
    return hkl_str.replace("_", ",") if hkl_str else hkl_str


def load_json(filepath):
    """Load JSON data file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def plot_neutron_data(files_dict, label_prefix="", vmin=None, vmax=None):
    """Plot neutron datasets (single detector).
    
    Args:
        files_dict: Dictionary of sample -> filepath
        label_prefix: Prefix for plot title
        vmin, vmax: Color scale limits. If None and AUTO_SCALE_PLOTS=False, uses individual plot ranges.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{label_prefix}Neutron Data: Strain Measurements (3,1,1 reflection)", 
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    # If autoscaling disabled but limits not provided, compute them
    if not AUTO_SCALE_PLOTS and vmin is None and vmax is None:
        all_strains = []
        for filepath in files_dict.values():
            data = load_json(filepath)
            if data:
                all_strains.extend(np.array(data.get("0/unconstrained_fit/3_1_1/strains/values", [])).flatten())
        if all_strains:
            vmin = np.min(all_strains)
            vmax = np.max(all_strains)

    for idx, (sample_label, filepath) in enumerate(files_dict.items()):
        data = load_json(filepath)
        if data is None:
            axes[idx].text(0.5, 0.5, f"{sample_label}\nFile not found", ha='center', va='center')
            continue
        
        x = np.array(data["labx"])
        z = np.array(data["labz"])
        strains = np.array(data["0/unconstrained_fit/3_1_1/strains/values"])
        
        ax = axes[idx]
        # Use provided vmin/vmax if autoscaling enabled, otherwise None (individual plot range)
        scatter_vmin = vmin if AUTO_SCALE_PLOTS or vmin is not None else None
        scatter_vmax = vmax if AUTO_SCALE_PLOTS or vmax is not None else None
        scatter = ax.scatter(x, z, c=strains, cmap="RdYlBu_r", s=50, alpha=0.7, edgecolors='k', linewidth=0.5,
                            vmin=scatter_vmin, vmax=scatter_vmax)
        ax.set_xlabel("Lab X (mm)")
        ax.set_ylabel("Lab Z (mm)")
        ax.set_title(f"{sample_label} Sample ({len(x)} pts)")
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Strain (×10⁻³)")

    plt.tight_layout()
    if INTERACTIVE_PLOT:
        plt.show()
    else:
        output_file = OUTPUT_DIR / "neutron_strains.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_xray_legacy():
    """Plot legacy xray strain map (single detector)."""
    data = load_json(DEPLOYMENT_XRAY_FILE)
    if data is None:
        print("  ✗ Legacy XRay file not found")
        return
    
    x = np.array(data["labx"])
    z = np.array(data["labz"])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("XRay Data: D2 Sample Strain Maps (Legacy - 2,2,2 reflection)", fontsize=14, fontweight='bold')

    # Uniform strain
    uniform_strain = np.array(data["0/data/uniform_strain"])
    scatter1 = axes[0].scatter(x, z, c=uniform_strain, cmap="RdYlBu_r", s=100, alpha=0.8)
    axes[0].set_xlabel("Lab X (mm)")
    axes[0].set_ylabel("Lab Z (mm)")
    axes[0].set_title("Uniform Strain")
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label="Strain (×10⁻³)")

    # Unconstrained strain
    unconstrained = np.array(data["0/data/unconstrained_strain"])
    scatter2 = axes[1].scatter(x, z, c=unconstrained, cmap="RdYlBu_r", s=100, alpha=0.8)
    axes[1].set_xlabel("Lab X (mm)")
    axes[1].set_ylabel("Lab Z (mm)")
    axes[1].set_title("Unconstrained Strain")
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label="Strain (×10⁻³)")

    plt.tight_layout()
    if INTERACTIVE_PLOT:
        plt.show()
    else:
        output_file = OUTPUT_DIR / "xray_legacy.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_d2_legacy_comparison():
    """Plot legacy neutron vs xray comparison (D2 sample)."""
    neutron_data = load_json(DEPLOYMENT_NEUTRON_FILES["D2"])
    xray_data = load_json(DEPLOYMENT_XRAY_FILE)

    if neutron_data is None or xray_data is None:
        print("  ✗ Missing data for comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("D2 Sample: Neutron vs XRay Comparison (Legacy)", fontsize=14, fontweight='bold')

    # Neutron D2
    x_n = np.array(neutron_data["labx"])
    z_n = np.array(neutron_data["labz"])
    strain_n = np.array(neutron_data["0/unconstrained_fit/3_1_1/strains/values"])
    scatter1 = axes[0].scatter(x_n, z_n, c=strain_n, cmap="RdYlBu_r", s=30, alpha=0.6)
    axes[0].set_xlabel("Lab X (mm)")
    axes[0].set_ylabel("Lab Z (mm)")
    axes[0].set_title(f"Neutron D2 (273 points, 3,1,1)")
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label="Strain")

    # XRay D2
    x_x = np.array(xray_data["labx"])
    z_x = np.array(xray_data["labz"])
    strain_x = np.array(xray_data["0/data/uniform_strain"])
    scatter2 = axes[1].scatter(x_x, z_x, c=strain_x, cmap="RdYlBu_r", s=100, alpha=0.8)
    axes[1].set_xlabel("Lab X (mm)")
    axes[1].set_ylabel("Lab Z (mm)")
    axes[1].set_title(f"XRay D2 (64 points, 2,2,2)")
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label="Strain")

    plt.tight_layout()
    if INTERACTIVE_PLOT:
        plt.show()
    else:
        output_file = OUTPUT_DIR / "d2_comparison_legacy.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_xray_multidetector(sample_name, filepath, output_name, vmin=None, vmax=None):
    """Plot XRay multi-detector data with configurable color scale.
    
    Uses strain data from selected HKL (Miller indices) if available,
    otherwise falls back to uniform/unconstrained strain data.
    
    Args:
        vmin, vmax: Color scale limits. If None and AUTO_SCALE_PLOTS=True, uses global scale.
                    If None and AUTO_SCALE_PLOTS=False, computes individual per-detector scales.
    """
    global XRAY_HKL, AUTO_SCALE_PLOTS
    data = load_json(filepath)
    if data is None:
        print(f"  ✗ File not found: {output_name}")
        return
    
    x = np.array(data["labx"])
    z = np.array(data["labz"])
    n_points = len(x)
    
    # Find detector IDs
    det_ids = sorted([int(k) for k in data.keys() if k.isdigit()])
    
    if not det_ids:
        print(f"  ✗ No detectors found in {output_name}")
        return
    
    print(f"  {sample_name}: {n_points} points, {len(det_ids)} detectors (HKL: {format_hkl(XRAY_HKL)})")
    
    # FIRST PASS: Collect strain data for all detectors
    all_strains = []
    detector_data_cache = {}  # Cache for second pass, stores individual min/max too
    
    for det_id in det_ids:
        det_key = str(det_id)
        if det_key not in data:
            continue
        
        det_info = data[det_key]
        strains_list = None
        errors_list = None
        
        # Try to get strains from HKL first
        hkl_values_key = f"{XRAY_HKL}/strains/values"
        hkl_errors_key = f"{XRAY_HKL}/strains/errors"
        
        if "unconstrained_fit" in det_info and hkl_values_key in det_info["unconstrained_fit"]:
            hkl_value = det_info["unconstrained_fit"][hkl_values_key]
            if isinstance(hkl_value, list):
                strains_list = hkl_value
                # Try to get errors if available
                if hkl_errors_key in det_info["unconstrained_fit"]:
                    errors_val = det_info["unconstrained_fit"][hkl_errors_key]
                    if isinstance(errors_val, list):
                        errors_list = errors_val
        
        # Fallback to simple unconstrained_strain data
        if strains_list is None and "data" in det_info:
            det_data = det_info["data"]
            strains_list = det_data.get("unconstrained_strain", [])
            errors_list = det_data.get("unconstrained_strain_stdev", [])
        
        if strains_list:
            all_strains.extend(strains_list)
            # Compute individual detector min/max for later use
            strains_array = np.array(strains_list)
            det_min = np.min(strains_array)
            det_max = np.max(strains_array)
            detector_data_cache[det_key] = (strains_list, errors_list, det_min, det_max)
    
    # Determine scale based on AUTO_SCALE_PLOTS and parameters
    if vmin is not None and vmax is not None:
        # External scale provided (from main, e.g., joint neutron/rosette scale)
        scale_type = "Joint"
    elif AUTO_SCALE_PLOTS:
        # Use global scale: compute from all detectors
        if all_strains:
            global_min = np.min(all_strains)
            global_max = np.max(all_strains)
            margin = (global_max - global_min) * 0.05 if (global_max - global_min) != 0 else 0.01
            vmin = global_min - margin
            vmax = global_max + margin
        else:
            vmin, vmax = -0.01, 0.01
        scale_type = "Global"
    else:
        # Individual scales: will be computed per detector
        vmin, vmax = None, None
        scale_type = "Individual"
    
    # Create figure with subplots for each detector
    ncols = 3
    nrows = int(np.ceil(len(det_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
    hkl_formatted = format_hkl(XRAY_HKL)
    scale_info = f"{scale_type} Scale" if scale_type == "Individual" else f"{scale_type} Scale: [{vmin:.4f}, {vmax:.4f}]"
    fig.suptitle(f"XRay {sample_name}: Multi-Detector Strains ({n_points} pts, {len(det_ids)} dets, {hkl_formatted})\n{scale_info}", 
                 fontsize=14, fontweight='bold')
    
    if nrows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # SECOND PASS: Plot each detector
    for idx, det_id in enumerate(det_ids):
        ax = axes[idx]
        det_key = str(det_id)
        
        if det_key not in data:
            ax.text(0.5, 0.5, f"Det {det_id}\nNo data", ha='center', va='center')
            continue
        
        if det_key not in detector_data_cache:
            ax.text(0.5, 0.5, f"Det {det_id}\nNo strain data", ha='center', va='center')
            continue
        
        strains_list, errors_list, det_min, det_max = detector_data_cache[det_key]
        
        # Handle both merged (different point counts) and single-measurement data
        n_det_points = len(strains_list)
        if n_det_points != n_points:
            # This is merged data with different point counts per detector
            x_det = x[:n_det_points]
            z_det = z[:n_det_points]
            title_suffix = f" ({n_det_points} pts)"
        else:
            x_det = x
            z_det = z
            title_suffix = ""
        
        strains = np.array(strains_list)
        errors = np.array(errors_list) if errors_list else np.zeros_like(strains)
        
        # Determine scale for this detector
        if scale_type == "Individual":
            # Use individual per-detector scale
            det_vmin = det_min - (det_max - det_min) * 0.05 if (det_max - det_min) != 0 else det_min - 0.01
            det_vmax = det_max + (det_max - det_min) * 0.05 if (det_max - det_min) != 0 else det_max + 0.01
            scatter_vmin = det_vmin
            scatter_vmax = det_vmax
        else:
            # Use global or joint scale
            scatter_vmin = vmin
            scatter_vmax = vmax
        
        # Scatter plot - USE APPROPRIATE SCALE (global or individual)
        scatter = ax.scatter(x_det, z_det, c=strains, cmap="RdYlBu_r", s=40, alpha=0.7, 
                            edgecolors='k', linewidth=0.3, vmin=scatter_vmin, vmax=scatter_vmax)
        
        # Add error visualization (semi-transparent circles around points)
        for i in range(min(len(x_det), len(errors))):
            circle = plt.Circle((x_det[i], z_det[i]), errors[i]*2, color='gray', 
                              alpha=0.1, linewidth=0)
            ax.add_patch(circle)
        
        ax.set_xlabel("Lab X (mm)", fontsize=9)
        ax.set_ylabel("Lab Z (mm)", fontsize=9)
        ax.set_title(f"Detector {det_id}{title_suffix}", fontsize=10)
        ax.grid(True, alpha=0.2)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Strain", fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(det_ids), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    if INTERACTIVE_PLOT:
        plt.show()
    else:
        output_file = OUTPUT_DIR / f"{output_name}_multidet.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  → Saved: {output_file.name}")
    plt.close()


def plot_xray_rosette(sample_name, filepath, output_name, vmin=None, vmax=None):
    """Plot XRay strain rosette analysis (εxx, εyy, εxy).
    
    Args:
        sample_name: Sample identifier for title
        filepath: Path to data file
        output_name: Name for output PNG file
        vmin, vmax: Color scale limits. If None and AUTO_SCALE_PLOTS=False, computes from data.
    """
    data = load_json(filepath)
    if data is None or "strain_rosette" not in data:
        return
    
    x = np.array(data["labx"])
    z = np.array(data["labz"])
    rosette = data["strain_rosette"]
    
    # If autoscaling disabled but limits not provided, compute them
    if not AUTO_SCALE_PLOTS and vmin is None and vmax is None:
        all_values = []
        for comp_name in ["e_xx", "e_yy", "e_xy"]:
            if comp_name in rosette:
                all_values.extend(rosette[comp_name].get("values", []))
        if all_values:
            vmin = np.min(all_values)
            vmax = np.max(all_values)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"XRay {sample_name}: Strain Rosette Analysis", fontsize=14, fontweight='bold')
    
    components = [
        ("e_xx", "$\\varepsilon_{xx}$ (Longitudinal Strain)", 0),
        ("e_yy", "$\\varepsilon_{yy}$ (Transverse Strain)", 1),
        ("e_xy", "$\\varepsilon_{xy}$ (Shear Strain)", 2),
    ]
    
    for comp_name, comp_label, ax_idx in components:
        if comp_name not in rosette:
            continue
        
        values_list = rosette[comp_name].get("values", [])
        errors_list = rosette[comp_name].get("errors", [])
        
        # Handle both merged and single-measurement data
        n_rosette_points = len(values_list)
        if n_rosette_points != len(x):
            # Use only coordinates matching rosette data
            x_use = x[:n_rosette_points]
            z_use = z[:n_rosette_points]
        else:
            x_use = x
            z_use = z
        
        values = np.array(values_list)
        errors = np.array(errors_list) if errors_list else np.zeros_like(values)
        
        ax = axes[ax_idx]
        # Use provided vmin/vmax if autoscaling enabled, otherwise None (individual plot range)
        scatter_vmin = vmin if AUTO_SCALE_PLOTS or vmin is not None else None
        scatter_vmax = vmax if AUTO_SCALE_PLOTS or vmax is not None else None
        scatter = ax.scatter(x_use, z_use, c=values, cmap="RdYlBu_r", s=50, alpha=0.7, 
                            edgecolors='k', linewidth=0.3, vmin=scatter_vmin, vmax=scatter_vmax)
        ax.set_xlabel("Lab X (mm)")
        ax.set_ylabel("Lab Z (mm)")
        ax.set_title(comp_label)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Strain (×10⁻³)")
    
    plt.tight_layout()
    if INTERACTIVE_PLOT:
        plt.show()
    else:
        output_file = OUTPUT_DIR / f"{output_name}_rosette.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  → Saved: {output_file.name}")
    plt.close()


def get_detector_strains(detector_dict, det_id, use_hkl=None):
    """
    Extract strain values from a detector, trying HKL first, then fallback to simple data.
    
    Args:
        detector_dict: The detector data dictionary
        det_id: Detector ID string
        use_hkl: If provided, try to get strains from this HKL (e.g., "3_1_1")
    
    Returns:
        List of strain values or empty list if not found
    """
    det_key = str(det_id)
    
    if det_key not in detector_dict:
        return []
    
    det_data = detector_dict[det_key]
    
    # Try HKL first if specified - key is like "3_1_1/strains/values" and maps directly to a list
    if use_hkl and "unconstrained_fit" in det_data:
        hkl_key = f"{use_hkl}/strains/values"
        if hkl_key in det_data["unconstrained_fit"]:
            hkl_value = det_data["unconstrained_fit"][hkl_key]
            if isinstance(hkl_value, list):
                return hkl_value
    
    # Fallback to simple strain data
    if "data" in det_data:
        return det_data["data"].get("unconstrained_strain", [])
    
    return []


def plot_sample_comparison():
    """Plot D1 and D2 neutron vs xray comparison (multi-detector average).
    
    Uses HKL-matched data when available (3,1,1 for both).
    Respects AUTO_SCALE_PLOTS flag:
      - True: global scale across all plots for direct comparison
      - False: individual scales per plot to maximize detail visibility
    """
    global XRAY_HKL, AUTO_SCALE_PLOTS
    
    # Load all data first
    neutron_d1 = load_json(NEUTRON_FILES["D1"])
    neutron_d2 = load_json(NEUTRON_FILES["D2"])
    xray_d1 = load_json(XRAY_MULTIDET_FILES["D1_full"]) if Path(XRAY_MULTIDET_FILES["D1_full"]).exists() else None
    xray_d2 = load_json(XRAY_MULTIDET_FILES["D2_full"]) if Path(XRAY_MULTIDET_FILES["D2_full"]).exists() else None
    
    # Determine scale mode
    if AUTO_SCALE_PLOTS:
        # GLOBAL SCALE: Collect all strain data across all plots
        all_strains = []
        
        if neutron_d1:
            all_strains.extend(np.array(neutron_d1.get("0/unconstrained_fit/3_1_1/strains/values", [])).flatten())
        if neutron_d2:
            all_strains.extend(np.array(neutron_d2.get("0/unconstrained_fit/3_1_1/strains/values", [])).flatten())
        
        if xray_d1:
            det_ids = sorted([int(k) for k in xray_d1.keys() if k.isdigit()])
            for det_id in det_ids:
                strain_data = get_detector_strains(xray_d1, det_id, use_hkl=XRAY_HKL)
                if strain_data:
                    all_strains.extend(strain_data)
        
        if xray_d2:
            det_ids = sorted([int(k) for k in xray_d2.keys() if k.isdigit()])
            for det_id in det_ids:
                strain_data = get_detector_strains(xray_d2, det_id, use_hkl=XRAY_HKL)
                if strain_data:
                    all_strains.extend(strain_data)
        
        if all_strains:
            global_min = np.min(all_strains)
            global_max = np.max(all_strains)
            margin = (global_max - global_min) * 0.05 if (global_max - global_min) != 0 else 0.01
            vmin_global = global_min - margin
            vmax_global = global_max + margin
        else:
            vmin_global, vmax_global = -0.01, 0.01
        
        scale_info = f"Global Scale: [{global_min:.4f}, {global_max:.4f}]"
    else:
        # INDIVIDUAL SCALES: Will compute per plot
        vmin_global, vmax_global = None, None
        scale_info = "Individual Scales"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    hkl_formatted = format_hkl(XRAY_HKL)
    fig.suptitle(f"Sample Comparison: Neutron (3,1,1) vs XRay Multi-Detector Avg ({hkl_formatted})\n{scale_info}", 
                 fontsize=14, fontweight='bold')
    
    # D1 Neutron
    if neutron_d1:
        x_n = np.array(neutron_d1["labx"])
        z_n = np.array(neutron_d1["labz"])
        strain_n = np.array(neutron_d1["0/unconstrained_fit/3_1_1/strains/values"])
        
        if AUTO_SCALE_PLOTS:
            vmin_plot, vmax_plot = vmin_global, vmax_global
        else:
            vmin_plot = np.min(strain_n)
            vmax_plot = np.max(strain_n)
            margin = (vmax_plot - vmin_plot) * 0.05 if (vmax_plot - vmin_plot) != 0 else 0.01
            vmin_plot -= margin
            vmax_plot += margin
        
        scatter = axes[0, 0].scatter(x_n, z_n, c=strain_n, cmap="RdYlBu_r", s=30, alpha=0.6, edgecolors='k', linewidth=0.3, vmin=vmin_plot, vmax=vmax_plot)
        axes[0, 0].set_title("D1 Neutron (3,1,1, 273 pts)")
        axes[0, 0].set_xlabel("Lab X (mm)")
        axes[0, 0].set_ylabel("Lab Z (mm)")
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0])
    
    # D1 XRay multi-detector average
    if xray_d1:
        x_x = np.array(xray_d1["labx"])
        z_x = np.array(xray_d1["labz"])
        
        # Get detector data
        det_ids = sorted([int(k) for k in xray_d1.keys() if k.isdigit()])
        strain_list = []
        for det_id in det_ids:
            strain_data = get_detector_strains(xray_d1, det_id, use_hkl=XRAY_HKL)
            if strain_data:
                strain_list.append(strain_data)
        
        if strain_list:
            # Pad to same length if needed
            max_len = max(len(s) for s in strain_list)
            strain_array = np.full((len(strain_list), max_len), np.nan)
            for i, s in enumerate(strain_list):
                strain_array[i, :len(s)] = s
            
            # Average, ignoring NaN
            strain_x = np.nanmean(strain_array, axis=0)
            
            # Use only valid points
            valid_mask = ~np.isnan(strain_x)
            x_plot = x_x[:len(strain_x)][valid_mask]
            z_plot = z_x[:len(strain_x)][valid_mask]
            strain_plot = strain_x[valid_mask]
            
            if AUTO_SCALE_PLOTS:
                vmin_plot, vmax_plot = vmin_global, vmax_global
            else:
                vmin_plot = np.min(strain_plot)
                vmax_plot = np.max(strain_plot)
                margin = (vmax_plot - vmin_plot) * 0.05 if (vmax_plot - vmin_plot) != 0 else 0.01
                vmin_plot -= margin
                vmax_plot += margin
            
            scatter = axes[0, 1].scatter(x_plot, z_plot, c=strain_plot, cmap="RdYlBu_r", s=50, alpha=0.6, edgecolors='k', linewidth=0.3, vmin=vmin_plot, vmax=vmax_plot)
            axes[0, 1].set_title(f"D1 XRay (multi-det avg {format_hkl(XRAY_HKL)}, {len(strain_plot)} pts)")
            axes[0, 1].set_xlabel("Lab X (mm)")
            axes[0, 1].set_ylabel("Lab Z (mm)")
            axes[0, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 1])
    
    # D2 Neutron
    if neutron_d2:
        x_n = np.array(neutron_d2["labx"])
        z_n = np.array(neutron_d2["labz"])
        strain_n = np.array(neutron_d2["0/unconstrained_fit/3_1_1/strains/values"])
        
        if AUTO_SCALE_PLOTS:
            vmin_plot, vmax_plot = vmin_global, vmax_global
        else:
            vmin_plot = np.min(strain_n)
            vmax_plot = np.max(strain_n)
            margin = (vmax_plot - vmin_plot) * 0.05 if (vmax_plot - vmin_plot) != 0 else 0.01
            vmin_plot -= margin
            vmax_plot += margin
        
        scatter = axes[1, 0].scatter(x_n, z_n, c=strain_n, cmap="RdYlBu_r", s=30, alpha=0.6, edgecolors='k', linewidth=0.3, vmin=vmin_plot, vmax=vmax_plot)
        axes[1, 0].set_title("D2 Neutron (3,1,1, 273 pts)")
        axes[1, 0].set_xlabel("Lab X (mm)")
        axes[1, 0].set_ylabel("Lab Z (mm)")
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0])
    
    # D2 XRay multi-detector average
    if xray_d2:
        x_x = np.array(xray_d2["labx"])
        z_x = np.array(xray_d2["labz"])
        
        # Get detector data
        det_ids = sorted([int(k) for k in xray_d2.keys() if k.isdigit()])
        strain_list = []
        for det_id in det_ids:
            strain_data = get_detector_strains(xray_d2, det_id, use_hkl=XRAY_HKL)
            if strain_data:
                strain_list.append(strain_data)
        
        if strain_list:
            # Pad to same length if needed
            max_len = max(len(s) for s in strain_list)
            strain_array = np.full((len(strain_list), max_len), np.nan)
            for i, s in enumerate(strain_list):
                strain_array[i, :len(s)] = s
            
            # Average, ignoring NaN
            strain_x = np.nanmean(strain_array, axis=0)
            
            # Use only valid points
            valid_mask = ~np.isnan(strain_x)
            x_plot = x_x[:len(strain_x)][valid_mask]
            z_plot = z_x[:len(strain_x)][valid_mask]
            strain_plot = strain_x[valid_mask]
            
            if AUTO_SCALE_PLOTS:
                vmin_plot, vmax_plot = vmin_global, vmax_global
            else:
                vmin_plot = np.min(strain_plot)
                vmax_plot = np.max(strain_plot)
                margin = (vmax_plot - vmin_plot) * 0.05 if (vmax_plot - vmin_plot) != 0 else 0.01
                vmin_plot -= margin
                vmax_plot += margin
            
            scatter = axes[1, 1].scatter(x_plot, z_plot, c=strain_plot, cmap="RdYlBu_r", s=50, alpha=0.6, edgecolors='k', linewidth=0.3, vmin=vmin_plot, vmax=vmax_plot)
            axes[1, 1].set_title(f"D2 XRay (multi-det avg {format_hkl(XRAY_HKL)}, {len(strain_plot)} pts)")
            axes[1, 1].set_xlabel("Lab X (mm)")
            axes[1, 1].set_ylabel("Lab Z (mm)")
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    if INTERACTIVE_PLOT:
        plt.show()
    else:
        output_file = OUTPUT_DIR / "sample_comparison.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def detect_available_data():
    """Auto-detect which data sources are available."""
    available = {
        "neutron_extracted": all(f.exists() for f in NEUTRON_FILES.values()),
        "xray_multidet": all(f.exists() for f in XRAY_MULTIDET_FILES.values()),
        "neutron_legacy": all(f.exists() for f in DEPLOYMENT_NEUTRON_FILES.values()),
        "xray_legacy": DEPLOYMENT_XRAY_FILE.exists(),
    }
    return available


def main(interactive=False, mode="auto", hkl="3_1_1", autoscale_plot=True):
    """Generate plots based on available data.
    
    Args:
        interactive: If True, display interactive plots. If False, save PNGs.
        mode: "auto" (all available), "enhanced" (new multi-det), "legacy" (deployment)
        hkl: Miller indices for XRay data (e.g., "3_1_1", "2_2_2"). Default "3_1_1" to match neutron.
        autoscale_plot: If True, use global min/max across all plots. If False, individual plot ranges.
    """
    global INTERACTIVE_PLOT, XRAY_HKL, AUTO_SCALE_PLOTS
    INTERACTIVE_PLOT = interactive
    XRAY_HKL = hkl
    AUTO_SCALE_PLOTS = autoscale_plot
    
    available = detect_available_data()
    plot_mode = "Interactive" if interactive else "Saved to outputs/"
    
    print(f"Plotting CHESS datasets ({plot_mode})...")
    print(f"Available data sources:")
    print(f"  Neutron extracted: {available['neutron_extracted']}")
    print(f"  XRay multi-detector: {available['xray_multidet']}")
    print(f"  Neutron legacy: {available['neutron_legacy']}")
    print(f"  XRay legacy: {available['xray_legacy']}")
    print()
    
    try:
        if mode == "auto":
            # Plot enhanced data if available, otherwise fall back to legacy
            if available["neutron_extracted"]:
                # Compute joint scale for neutron and rosette if autoscaling
                vmin_nr, vmax_nr = None, None
                if AUTO_SCALE_PLOTS:
                    all_strain_data = []
                    # Collect neutron strains
                    for filepath in NEUTRON_FILES.values():
                        data = load_json(filepath)
                        if data:
                            all_strain_data.extend(np.array(data.get("0/unconstrained_fit/3_1_1/strains/values", [])).flatten())
                    # Collect rosette strains
                    for filepath in XRAY_MULTIDET_FILES.values():
                        data = load_json(filepath)
                        if data and "strain_rosette" in data:
                            for comp_name in ["e_xx", "e_yy", "e_xy"]:
                                all_strain_data.extend(data["strain_rosette"].get(comp_name, {}).get("values", []))
                    
                    if all_strain_data:
                        vmin_nr = np.min(all_strain_data)
                        vmax_nr = np.max(all_strain_data)
                
                print("Neutron Data (Extracted):")
                plot_neutron_data(NEUTRON_FILES, vmin=vmin_nr, vmax=vmax_nr)
                
                print("\nXRay Data (Multi-Detector):")
                for name, filepath in XRAY_MULTIDET_FILES.items():
                    if filepath.exists():
                        plot_xray_multidetector(name, filepath, name)
                        plot_xray_rosette(name, filepath, name, vmin=vmin_nr, vmax=vmax_nr)
                
                print("\nComparison (Multi-Detector Average):")
                plot_sample_comparison()
            elif available["neutron_legacy"]:
                print("Neutron Data (Legacy):")
                plot_neutron_data(DEPLOYMENT_NEUTRON_FILES, label_prefix="[Legacy] ")
                
                if available["xray_legacy"]:
                    print("\nXRay Data (Legacy):")
                    plot_xray_legacy()
                    
                    print("\nComparison (Legacy):")
                    plot_d2_legacy_comparison()
            else:
                print("✗ No data sources found")
                return 1
        
        elif mode == "enhanced":
            if available["neutron_extracted"] and available["xray_multidet"]:
                # Compute joint scale for neutron and rosette if autoscaling
                vmin_nr, vmax_nr = None, None
                if AUTO_SCALE_PLOTS:
                    all_strain_data = []
                    # Collect neutron strains
                    for filepath in NEUTRON_FILES.values():
                        data = load_json(filepath)
                        if data:
                            all_strain_data.extend(np.array(data.get("0/unconstrained_fit/3_1_1/strains/values", [])).flatten())
                    # Collect rosette strains
                    for filepath in XRAY_MULTIDET_FILES.values():
                        data = load_json(filepath)
                        if data and "strain_rosette" in data:
                            for comp_name in ["e_xx", "e_yy", "e_xy"]:
                                all_strain_data.extend(data["strain_rosette"].get(comp_name, {}).get("values", []))
                    
                    if all_strain_data:
                        vmin_nr = np.min(all_strain_data)
                        vmax_nr = np.max(all_strain_data)
                
                print("Neutron Data (Extracted):")
                plot_neutron_data(NEUTRON_FILES, vmin=vmin_nr, vmax=vmax_nr)
                
                print("\nXRay Data (Multi-Detector):")
                for name, filepath in XRAY_MULTIDET_FILES.items():
                    if filepath.exists():
                        plot_xray_multidetector(name, filepath, name)
                        plot_xray_rosette(name, filepath, name, vmin=vmin_nr, vmax=vmax_nr)
                
                print("\nComparison (Multi-Detector Average):")
                plot_sample_comparison()
            else:
                print("✗ Enhanced data (extracted multi-detector) not available")
                print("  Run: python scripts/extract_xray_hdf5.py")
                print("  Then: python scripts/merge_xray_measurements.py")
                return 1
        
        elif mode == "legacy":
            if available["neutron_legacy"]:
                print("Neutron Data (Legacy):")
                plot_neutron_data(DEPLOYMENT_NEUTRON_FILES, label_prefix="[Legacy] ")
                
                if available["xray_legacy"]:
                    print("\nXRay Data (Legacy):")
                    plot_xray_legacy()
                    
                    print("\nComparison (Legacy):")
                    plot_d2_legacy_comparison()
                else:
                    print("✗ Legacy XRay data not found")
                    return 1
            else:
                print("✗ Legacy neutron data not found")
                return 1
        
        if not INTERACTIVE_PLOT:
            print("\n✓ All plots generated successfully!")
        
    except Exception as e:
        print(f"✗ Error during plotting: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot CHESS experiment datasets with multi-detector support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_datasets.py                                      # Auto-detect & plot all available data
  python plot_datasets.py --plot                               # Show interactive matplotlib plots
  python plot_datasets.py --hkl 3_1_1                         # Plot XRay with 3,1,1 (matches neutron)
  python plot_datasets.py --hkl 2_2_2                         # Plot XRay with 2,2,2
  python plot_datasets.py --enhanced --hkl 3_1_1              # Enhanced mode with custom HKL
  python plot_datasets.py --autoscale-plot false              # Individual plot scales (no global scaling)
  python plot_datasets.py --hkl 3_1_1 --autoscale-plot false  # Custom HKL with individual scales
  python plot_datasets.py --legacy                            # Plot legacy deployment datasets only
        """,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show interactive matplotlib plots instead of saving PNGs",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "enhanced", "legacy"],
        default="auto",
        help="Which data source to plot: auto-detect (default), enhanced (multi-det), or legacy (deployment)",
    )
    parser.add_argument(
        "--hkl",
        default="3_1_1",
        help="Miller indices for XRay strain data (e.g., 3_1_1, 2_2_2, 1_1_1). Default: 3_1_1 (matches neutron)",
    )
    parser.add_argument(
        "--autoscale-plot",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="Use global min/max scale across all plots (True/False). Default: True",
    )
    
    args = parser.parse_args()
    exit(main(interactive=args.plot, mode=args.mode, hkl=args.hkl, autoscale_plot=args.autoscale_plot))
