# Data Overview

## Included Datasets

### Neutron Data (Single Detector, 3,1,1 Reflection)
- **Files**: `data/neutron/{A2,D1,D2,D4}_full.json`
- **Points**: ~273 per sample
- **Source**: Legacy deployment datasets (pre-processed)
- **Schema**: `{labx, labz, 0/unconstrained_fit/3_1_1/strains/values}`

### XRay Data (Multi-Detector, 14 Detectors)
- **Files**: `data/xray/{D1,D2}_full.json`
- **Points**: D1=3,790, D2=4,485
- **Source**: Extracted from CHESS HDF5 files, includes rosette fitting
- **Schema**: Coordinates + all 14 detectors + rosette components (εxx, εyy, εxy)
- **Status**: A2 and D4 XRay data not available (HDF5 reduction incomplete at beamline)

### Strain Tensor Components (Rosette Analysis)
- **εxx**: Longitudinal strain (lab X-direction)
- **εyy**: Transverse strain (lab Y-direction)
- **εxy**: Shear strain (lab XY-plane)

## How Data Was Derived

### Neutron Data
Pre-processed and provided in deployment package. No processing required.
Performed by instrument scientist
POC: Chris Fancher

### XRay Data (D1, D2 only)
```
Raw CHESS data → CHESS beamline reduction → HDF5 (NeXus) → JSON extraction
```

**Raw CHESS data to reduction to HDF5**
Performed by instrument scientist + CHESS staff
POC: Amlan Das

**Extraction steps**:
1. Read HDF5 files from `/reduced_data/fancher-4630-b/{sample}/output/strain_full.nxs`
2. Extract coordinates (labx, labz) from dataset group
3. Extract strain values from all 14 detector groups
4. Fit strain rosette (εxx, εyy, εxy) from multi-detector measurements
5. Save to JSON with complete detector and rosette data

**Script**: `scripts/analysis/extract_xray_hdf5.py`

## CHESS Filesystem Paths (For Team Reference)

If you download raw and reduced data from CHESS:

```
Raw XRay Data (SPEC logs, detector scans) at CHESS:
  /nfs/chess/id1a3/2026-2/fancher-4630-b/a2-1/
  /nfs/chess/id1a3/2026-2/fancher-4630-b/d1-1/
  /nfs/chess/id1a3/2026-2/fancher-4630-b/d1-2/
  /nfs/chess/id1a3/2026-2/fancher-4630-b/d1-3/
  /nfs/chess/id1a3/2026-2/fancher-4630-b/d2-1/
  /nfs/chess/id1a3/2026-2/fancher-4630-b/d2-2/
  /nfs/chess/id1a3/2026-2/fancher-4630-b/d4-1/
  /nfs/chess/id1a3/2026-2/fancher-4630-b/d4-2/
  /nfs/chess/id1a3/2026-2/fancher-4630-b/d4-3/

Reduced HDF5 Files (NeXus strain data):
  Available: d1-2, d1-3, d2-2

  /nfs/chess/auxiliary/reduced_data/cycles/2026-2/id1a3/fancher-4630-b/d1-2/output/strain_full.nxs
  /nfs/chess/auxiliary/reduced_data/cycles/2026-2/id1a3/fancher-4630-b/d1-3/output/strain_full.nxs
  /nfs/chess/auxiliary/reduced_data/cycles/2026-2/id1a3/fancher-4630-b/d2-2/output_nopar/strain_full.nxs

  Missing: a2-1, d2-1, d4-1, d4-2, d4-3
  NOTES: d1-1 seems sparse, only 3 detector directories so can probably skip? (question for Amlan)
  UPDATED: Contacted Amlan at CHESS for beamline for reduction on July 21st, 2026; plan to meet July 29th
```

## Detector Configuration

- **14 detectors** with IDs: 0, 1, 5, 8, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22
- **Angles**: 0° to 220° (circumferential coverage)
- **Setup**: Rosette detector configuration for complete strain tensor extraction
- **Config file**: `detector_config.yaml`

## Notes

- Neutron and XRay use different lattice constants (3,1,1 vs 2,2,2)
- XRay data includes measurement uncertainty (unconstrained_strain_stdev)
- Rosette fitting requires ≥3 detectors; fully sampled with 14
- D1/D2 complete; A2/D4 require CHESS beamline HDF5 reduction (external step)
