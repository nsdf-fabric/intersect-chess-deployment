# Intersect CHESS Deployment

This repository is organized around self-contained scenario folders.
Each scenario includes its own compose file and campaign payload.

## Quick Start

Run any scenario directly:

```bash
docker compose -f scenarios/xray_ambench_smb_json/docker-compose.yml up -d --force-recreate
```

For neutron scenarios, choose one of the dataset files with `NEUTRON_DATASET_JSON`:

- `A2_2p1_30layers_VDriveSPF_LD.json`
- `D1_1p8_30layers_VDriveSPF_LD.json`
- `D2_2p3_30layers_VDriveSPF_LD.json`
- `D4_4p4_30layers_VDriveSPF_LD.json`

Example:

```bash
NEUTRON_DATASET_JSON=D2_2p3_30layers_VDriveSPF_LD.json \
  docker compose -f scenarios/neutron_ammdf_vulcan_json/docker-compose.yml up -d --force-recreate
```

## Scenarios

| Scenario | Dataset | Services | Compose | Campaign | Notes |
|---|---|---|---|---|---|
| xray_ambench_smb_hdf5 | Xray HDF5 reduced fixture (`strain_map.small.nxs`) | Core stack (broker, mongodb, informer, data service, dial, instrument-control, orchestrator, campaign-ui) | [scenarios/xray_ambench_smb_hdf5/docker-compose.yml](scenarios/xray_ambench_smb_hdf5/docker-compose.yml) | [scenarios/xray_ambench_smb_hdf5/campaign.json](scenarios/xray_ambench_smb_hdf5/campaign.json) | Default HDF5 flow |
| xray_ambench_smb_hdf5-full | Xray full HDF5 (`.local-data/strain_map.nxs`) | Core stack | [scenarios/xray_ambench_smb_hdf5-full/docker-compose.yml](scenarios/xray_ambench_smb_hdf5-full/docker-compose.yml) | [scenarios/xray_ambench_smb_hdf5-full/campaign.json](scenarios/xray_ambench_smb_hdf5-full/campaign.json) | Uses local full file |
| xray_ambench_smb_json | Xray JSON reduced fixture (`strain_map.reduced.json`) | Core stack (informer disabled) | [scenarios/xray_ambench_smb_json/docker-compose.yml](scenarios/xray_ambench_smb_json/docker-compose.yml) | [scenarios/xray_ambench_smb_json/campaign.json](scenarios/xray_ambench_smb_json/campaign.json) | JSON monitor mode |
| xray_ambench_smb_json-storage | Xray JSON reduced fixture | Core stack + minio + nsdf-storage-service | [scenarios/xray_ambench_smb_json-storage/docker-compose.yml](scenarios/xray_ambench_smb_json-storage/docker-compose.yml) | [scenarios/xray_ambench_smb_json-storage/campaign.json](scenarios/xray_ambench_smb_json-storage/campaign.json) | Persists DIAL outputs |
| xray_ambench_smb_json-storage-dashboard | Xray JSON reduced fixture | JSON-storage stack + dashboard-seed + nsdf-strain-transform + dashboard | [scenarios/xray_ambench_smb_json-storage-dashboard/docker-compose.yml](scenarios/xray_ambench_smb_json-storage-dashboard/docker-compose.yml) | [scenarios/xray_ambench_smb_json-storage-dashboard/campaign.json](scenarios/xray_ambench_smb_json-storage-dashboard/campaign.json) | Dashboard on :8059 |
| neutron_ammdf_vulcan_json | Neutron JSON via `NEUTRON_DATASET_JSON` | Core stack (informer disabled) | [scenarios/neutron_ammdf_vulcan_json/docker-compose.yml](scenarios/neutron_ammdf_vulcan_json/docker-compose.yml) | [scenarios/neutron_ammdf_vulcan_json/campaign.json](scenarios/neutron_ammdf_vulcan_json/campaign.json) | `value_key` set for neutron strains |
| neutron_ammdf_vulcan_json-storage | Neutron JSON via `NEUTRON_DATASET_JSON` | Core stack + minio + nsdf-storage-service | [scenarios/neutron_ammdf_vulcan_json-storage/docker-compose.yml](scenarios/neutron_ammdf_vulcan_json-storage/docker-compose.yml) | [scenarios/neutron_ammdf_vulcan_json-storage/campaign.json](scenarios/neutron_ammdf_vulcan_json-storage/campaign.json) | Storage-enabled neutron flow |
| neutron_ammdf_vulcan_json-storage-dashboard | Neutron JSON via `NEUTRON_DATASET_JSON` | JSON-storage stack + dashboard-seed + nsdf-strain-transform + dashboard | [scenarios/neutron_ammdf_vulcan_json-storage-dashboard/docker-compose.yml](scenarios/neutron_ammdf_vulcan_json-storage-dashboard/docker-compose.yml) | [scenarios/neutron_ammdf_vulcan_json-storage-dashboard/campaign.json](scenarios/neutron_ammdf_vulcan_json-storage-dashboard/campaign.json) | Dashboard on :8059 |

## Validate Scenarios

Run the validation helper:

```bash
./scripts/check_scenarios.sh
```

What it checks:

- Every scenario has a valid compose config
- Every scenario has a valid `campaign.json`
- Every neutron scenario resolves compose config for all 4 neutron dataset filenames

## Utilities

- Reduced HDF5 fixture generator: [scripts/reduce_strain_map_hdf5.py](scripts/reduce_strain_map_hdf5.py)
- Reduced JSON export tool: [scripts/export_strain_map_to_json.py](scripts/export_strain_map_to_json.py)
