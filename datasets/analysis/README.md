# CHESS Data Analysis

Multi-detector XRay and neutron strain analysis with visualization tools.

## Quick Start

### 1. Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. Run Plots
```bash
# Default: Save PNGs to outputs/ with global color scaling
python scripts/analysis/plot_datasets.py

# Interactive matplotlib (zoom, pan, inspect)
python scripts/analysis/plot_datasets.py --plot

# Data source modes
python scripts/analysis/plot_datasets.py --mode auto      # Auto-detect (default)
python scripts/analysis/plot_datasets.py --mode enhanced  # Force extracted multi-detector data

# Select XRay Miller indices (default: 3_1_1 to match neutron)
python scripts/analysis/plot_datasets.py --hkl 2_2_2
python scripts/analysis/plot_datasets.py --hkl 4_0_0

# Control color scale mode
python scripts/analysis/plot_datasets.py --autoscale-plot true   # Global scale (default)
python scripts/analysis/plot_datasets.py --autoscale-plot false  # Individual scales per plot

# Combine options
python scripts/analysis/plot_datasets.py --hkl 3_1_1 --autoscale-plot false
```

### 3. Generated Outputs
- `neutron_strains.png` - Neutron single-detector measurements (2×2 grid)
- `D{1,2}_full_multidet.png` - 14 XRay detectors in 3×5 grid with error visualization
- `D{1,2}_full_rosette.png` - Strain tensor components (εxx, εyy, εxy) fitted from multi-detector data
- `sample_comparison.png` - 2×2 comparison grid (Neutron D1, XRay D1, Neutron D2, XRay D2)

## Features

### Color Scaling Modes

The `--autoscale-plot` flag controls how color scales are computed across plots:

#### Global Scaling (default: `--autoscale-plot true`)
- **All plots share one color scale** computed from min/max across all data
- **Use case**: Direct visual comparison between samples/detectors
- **Benefit**: Same color = same strain value across all plots
- **Trade-off**: Small variations may be hard to see if ranges differ widely

Example: With neutron (-400 to +600) and XRay (-0.002 to +0.002), global scale makes XRay look featureless.

#### Individual Scaling (`--autoscale-plot false`)
- **Each plot gets its own color scale** based on its data range
- **Use case**: Maximize detail visibility in each plot independently
- **Benefit**: Reveals spatial patterns and fine structure in each detector/sample
- **Trade-off**: Colors don't directly compare between plots

**Recommended Usage**:
- `--autoscale-plot true` (default) for neutron/XRay comparisons
- `--autoscale-plot false` for detailed analysis of individual datasets

### Miller Indices Selection

The `--hkl` flag selects which Miller indices to plot for XRay data:

```bash
# Default (matches neutron 3,1,1)
python scripts/analysis/plot_datasets.py --hkl 3_1_1

# Other available indices (from 9 fitted planes)
python scripts/analysis/plot_datasets.py --hkl 2_0_0
python scripts/analysis/plot_datasets.py --hkl 2_2_0
python scripts/analysis/plot_datasets.py --hkl 2_2_2
python scripts/analysis/plot_datasets.py --hkl 3_3_1
python scripts/analysis/plot_datasets.py --hkl 4_0_0
python scripts/analysis/plot_datasets.py --hkl 4_2_0
python scripts/analysis/plot_datasets.py --hkl 4_2_2
python scripts/analysis/plot_datasets.py --hkl 1_1_1
```

**Note**: XRay data contains all 9 Miller indices. Neutron uses only 3,1,1. When comparing, use `--hkl 3_1_1`.

### Plot Details

**Neutron Data** (`neutron_strains.png`)
- Single detector measurement per sample
- 2×2 grid showing strain distributions
- ~273 measurement points per sample

**Multi-Detector XRay** (`D{1,2}_full_multidet.png`)
- All 14 detectors arranged in 3×5 grid
- Each detector shows strain field at different angular position
- Semi-transparent circles show measurement uncertainty (2σ)
- Global scale (by default) enables detector-to-detector comparison

**Rosette Strain** (`D{1,2}_full_rosette.png`)
- Three subplots: εxx (longitudinal), εyy (transverse), εxy (shear)
- Fitted from multi-detector measurements using least-squares
- Shows complete strain tensor components

**Sample Comparison** (`sample_comparison.png`)
- 2×2 grid: Neutron and XRay averaged maps side-by-side
- D1 (top row), D2 (bottom row)
- Direct visual comparison when using global scaling
- Individual color scales reveal fine detail when using `--autoscale-plot false`

## Data

Located in `data/`:
- **Neutron**: A2, D1, D2, D4 (single detector, ~273 pts each, 3,1,1 reflection)
- **XRay**: D1, D2 only (multi-detector, 14 detectors, 3,790-4,485 pts, all 9 Miller indices)
  - Includes rosette fitted components (εxx, εyy, εxy)
  - A2/D4 require HDF5 reduction (see [docs/DATA.md](docs/DATA.md))

## Scripts

### `plot_datasets.py` - Main Visualization Tool

Generate plots from extracted and merged datasets.

```bash
python scripts/analysis/plot_datasets.py --help
```

**Common arguments:**
- `--plot` - Show interactive matplotlib plots (instead of saving PNG)
- `--mode {auto, enhanced}` - Data source (default: auto-detect)
  - `auto`: Auto-detect available data and plot all (recommended)
  - `enhanced`: Force use of extracted multi-detector data only
- `--hkl HKL` - Miller indices for XRay (default: 3_1_1)
- `--autoscale-plot {true, false}` - Global vs individual color scales (default: true)

### `extract_xray_hdf5.py` - Extract HDF5 → JSON

Convert CHESS beamline HDF5 files to JSON format.

```bash
python scripts/analysis/extract_xray_hdf5.py
```

**Input**: CHESS reduced HDF5 files at `/reduced_data/fancher-4630-b/{sample}/output/strain_full.nxs`
**Output**: `data/xray/{sample}.json` with all detector data and 9 Miller indices

### `merge_xray_measurements.py` - Merge Individual Measurements

Combine multi-measurement samples into complete datasets.

```bash
python scripts/analysis/merge_xray_measurements.py
```

**Input**: Individual measurement JSON files from extraction
**Output**: `data/xray/{sample}_full.json` with merged strain data

## Extracting New Data

To process HDF5 files from CHESS beamline:

1. Download reduced files: `/reduced_data/fancher-4630-b/{sample}/output/strain_full.nxs`
2. Extract to JSON:
   ```bash
   python scripts/analysis/extract_xray_hdf5.py
   ```
3. Merge measurements (if multiple scans per sample):
   ```bash
   python scripts/analysis/merge_xray_measurements.py
   ```
4. Generate plots:
   ```bash
   python scripts/analysis/plot_datasets.py
   ```

## Help & Reference

For detailed command options:
```bash
python scripts/analysis/plot_datasets.py --help
```

For data structure and CHESS filesystem paths:
- [docs/DATA.md](docs/DATA.md) - Datasets, derivation, CHESS paths

