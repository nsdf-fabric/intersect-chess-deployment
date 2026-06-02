# Xray AMBench SMB JSON + Storage + Dashboard Scenario

This scenario runs the xray AMBench SMB JSON+storage stack and the scientist dashboard.

## Run

```bash
docker compose -f scenarios/xray_ambench_smb_json-storage-dashboard/docker-compose.yml up -d --force-recreate
```

## Campaign payload

`scenarios/xray_ambench_smb_json-storage-dashboard/campaign.json`

## Dashboard

Open `http://localhost:8059/ORNL_CHESS_strain`.
