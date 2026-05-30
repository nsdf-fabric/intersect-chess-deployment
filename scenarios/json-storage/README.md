# JSON + Storage Scenario

Use the reduced JSON fixture with the CHESS data service, DIAL, and the NSDF
storage service. The campaign uses the same CHESS/DIAL workflow as the other
scenarios so behavior is consistent across all test modes.

NSDF storage is still exercised in this scenario through its normal event-driven
subscriptions while the shared campaign is running.

This scenario also starts MinIO locally so the storage service can upload its
JSON outputs into an object store during local testing.

## Run

```bash
docker compose -f docker-compose.yml -f scenarios/json-storage/docker-compose.override.yml up -d --force-recreate
```

## Campaign payload

`scenarios/json-storage/chess-autonomous-bootstrap.campaign.json`
