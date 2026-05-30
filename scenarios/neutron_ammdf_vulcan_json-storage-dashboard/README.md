# Neutron AMMDF VULCAN JSON + Storage + Dashboard Scenario

This scenario runs neutron AMMDF VULCAN JSON data with storage and dashboard services.

## Run

```bash
NEUTRON_DATASET_JSON=D2_2p3_30layers_VDriveSPF_LD.json \
  docker compose -f scenarios/neutron_ammdf_vulcan_json-storage-dashboard/docker-compose.yml up -d --force-recreate
```

If omitted, `A2_2p1_30layers_VDriveSPF_LD.json` is used.

## Campaign payload

`scenarios/neutron_ammdf_vulcan_json-storage-dashboard/campaign.json`

## Dashboard

Open `http://localhost:8059/ORNL_CHESS_strain`.
