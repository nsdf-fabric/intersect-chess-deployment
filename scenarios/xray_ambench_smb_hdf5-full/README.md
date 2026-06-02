# Xray AMBench SMB HDF5 Full Scenario

This scenario runs the xray AMBench SMB workflow against a local full-size HDF5 file.

## Prepare dataset

Place the full file at:

`scenarios/xray_ambench_smb_hdf5-full/.local-data/strain_map.nxs`

## Run

```bash
docker compose -f scenarios/xray_ambench_smb_hdf5-full/docker-compose.yml up -d --force-recreate
```

## Campaign payload

`scenarios/xray_ambench_smb_hdf5-full/campaign.json`
