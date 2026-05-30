# Neutron AMMDF VULCAN JSON Scenario

This scenario runs neutron AMMDF VULCAN JSON data through the same campaign flow.

## Dataset selection

Select one of the dataset files via environment variable:

- `A2_2p1_30layers_VDriveSPF_LD.json`
- `D1_1p8_30layers_VDriveSPF_LD.json`
- `D2_2p3_30layers_VDriveSPF_LD.json`
- `D4_4p4_30layers_VDriveSPF_LD.json`

## Run

```bash
NEUTRON_DATASET_JSON=D2_2p3_30layers_VDriveSPF_LD.json \
  docker compose -f scenarios/neutron_ammdf_vulcan_json/docker-compose.yml up -d --force-recreate
```

If omitted, `A2_2p1_30layers_VDriveSPF_LD.json` is used.

## Campaign payload

`scenarios/neutron_ammdf_vulcan_json/campaign.json`
