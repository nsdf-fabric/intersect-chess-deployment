# JSON Storage + Dashboard Scenario

This scenario layers the collaborator dashboard on top of the local MinIO-backed json-storage stack.

It seeds the bundled CHESS strain fixture into MinIO at:

`s3://scientistcloud/IDX_TEST/ORNL_strain/reduced_data.json`

and starts the dashboard against that object while the json-storage campaign writes its own outputs into the same MinIO instance.

## Run

```bash
docker compose -f docker-compose.yml \
  -f scenarios/json-storage/docker-compose.override.yml \
  -f scenarios/json-storage-dashboard/docker-compose.override.yml \
  up -d --force-recreate
```

## Dashboard

Open `http://localhost:8059/ORNL_CHESS_strain` after the stack starts.