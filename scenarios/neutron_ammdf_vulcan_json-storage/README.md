# Neutron AMMDF VULCAN JSON + Storage Scenario

This scenario runs neutron AMMDF VULCAN JSON data plus MinIO and NSDF storage.

## Run

```bash
NEUTRON_DATASET_JSON=D2_2p3_30layers_VDriveSPF_LD.json \
  docker compose -f scenarios/neutron_ammdf_vulcan_json-storage/docker-compose.yml up -d --force-recreate
```

If omitted, `A2_2p1_30layers_VDriveSPF_LD.json` is used.

## Campaign payload

`scenarios/neutron_ammdf_vulcan_json-storage/campaign.json`
