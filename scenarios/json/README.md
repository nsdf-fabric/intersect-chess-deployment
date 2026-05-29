# JSON Scenario

Use a static reduced JSON dataset directly with `chess-data-service` JSON monitor mode.

This is a lightweight mode intended for quick local validation.

## Run

```bash
docker compose -f docker-compose.yml -f scenarios/json/docker-compose.override.yml up -d --force-recreate
```

## Campaign payload

`scenarios/json/chess-autonomous-bootstrap.campaign.json`

## Dataset artifact

`datasets/strain_map.reduced.json`

Current fixture size is approximately 6 KB.
