# JSON + Storage Scenario

Use the reduced JSON fixture with the CHESS data service, DIAL, and the NSDF
storage service. The campaign now includes a final storage task group that
invokes `nsdf_storage.new_measurement`, `nsdf_storage.next_point`, and
`nsdf_storage.surrogate_values` after the main CHESS/DIAL group completes.

This scenario also starts MinIO locally so the storage service can upload its
JSON outputs into an object store during local testing.

## Run

```bash
docker compose -f docker-compose.yml -f scenarios/json-storage/docker-compose.override.yml up -d --force-recreate
```

## Campaign payload

`scenarios/json-storage/chess-autonomous-bootstrap.campaign.json`
