# CHESS Autonomous Experiment Deployment

Full-stack Docker Compose deployment for the CHESS autonomous feedback loop using
pre-built GHCR images.

This repository runs:

- RabbitMQ broker for INTERSECT request/reply and events
- MongoDB for Dial workflow state
- Informer + CHESS data egress + CHESS instrument-control services
- Dial active-learning service
- Campaign Orchestrator REST API
- Optional one-shot campaign submitter

## Verified Working State

The current compose setup has been validated end-to-end with:

- Dial image: `ghcr.io/intersect-dial/dial:develop`
- Orchestrator in production mode (`PRODUCTION=true`) so it binds to `0.0.0.0`
- Campaign: [conf/chess-autonomous-bootstrap.campaign.json](conf/chess-autonomous-bootstrap.campaign.json)

Observed successful outcome:

- All 3 steps emit `STEP_COMPLETE`
- Campaign emits `CAMPAIGN_COMPLETE`

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Campaign Orchestrator REST API (:8000)                                       │
│ Executes campaign task graphs over INTERSECT request/reply                   │
└───────────┬─────────────────────┬───────────────────────┬────────────────────┘
            ▲                     │                       │
            │                     │                       │ 
            │                     │                       │ 
            │                     │                       │ 
            │                     │                       │ 
            │                     │                       │ 
            │ new_measurement     │                       │ 
            │ event               ▼                       ▼
 ┌──────────┴────────┐  ┌─────────┴─────────┐  ┌──────────┴───────────────────┐
 │ chess-data-egress │  │       Dial        │  │ chess-instrument-control     │
 │ (chess-data-      │  │ (active learning) │  │ (spec-side service)          │
 │  service)         │  │                   │  │                              │
 │ Monitors NEW HDF5 │  │ Initializes and   │  │ Writes loc files for SPEC    │
 │ Emits measurements│  │ updates workflow  │  │                              │
 └───────────────────┘  └───────────────────┘  └──────────────────────────────┘
                      ▲                                              │
                      │             shared-data volume (/data)       │
                      │      ┌──────────────────────────────────┐    │
                      └──────│ chess-instrument-control-informer│────┘
                             │ Full HDF5 -> New HDF5 + loc watch│
                             └──────────────────────────────────┘
```

## Services

| Service | Image | Ports | Role |
|---|---|---|---|
| broker | bitnamilegacy/rabbitmq:3.13.3 | 5672, 15672 | AMQP bus |
| mongodb | bitnamilegacy/mongodb:8.0 | 27017 | Dial workflow state |
| informer | ghcr.io/nsdf-fabric/chess-instrument-control-informer:0.2.0 | - | Full HDF5 -> New HDF5 + loc watcher |
| chess-data-service | ghcr.io/nsdf-fabric/intersect-chess-data-service:0.1.5 | - | Emits measurement events |
| dial | ghcr.io/intersect-dial/dial:develop | - | Active learning |
| chess-instrument-control-service | ghcr.io/nsdf-fabric/intersect-chess-instrument-control-service:0.1.2 | - | Instrument control requests |
| orchestrator | ghcr.io/intersect-sdk/campaign-orchestrator:latest | 8000 | Campaign REST API |
| submit-campaign | curlimages/curl:8.8.0 | - | One-shot startup submission |

## Quick Start

Run full stack including automatic one-shot campaign submission:

```bash
docker compose up -d --force-recreate
```

Open API docs:

- http://localhost:8000/docs

## Deterministic Manual Campaign Run

Use this when you want a clean, repeatable test run and explicit campaign ID
control.

1. Start core services without the auto submitter:

```bash
docker compose up -d --force-recreate broker mongodb informer chess-data-service dial chess-instrument-control-service orchestrator
```

2. Wait for orchestrator readiness and submit a fresh campaign ID:

```bash
until curl -fsS http://localhost:8000/v1/ping >/dev/null; do
     echo "waiting for orchestrator"
done

new_id=$(cat /proc/sys/kernel/random/uuid)
payload=$(jq --arg id "$new_id" '.id=$id' conf/chess-autonomous-bootstrap.campaign.json)

curl -sS -X POST http://localhost:8000/v1/orchestrator/start_campaign \
     -H 'Authorization: 1234567890abcdef1234567890abcdef' \
     -H 'Content-Type: application/json' \
     --data "$payload"

echo "CAMPAIGN_ID=$new_id"
```

3. Verify completion:

```bash
docker logs deployment-orchestrator-1 --tail 3000 | rg "STEP_START|STEP_COMPLETE|CAMPAIGN_COMPLETE|CAMPAIGN_ERROR|UNKNOWN_ERROR|SCHEMA_ERROR"
```

## Data Flow

1. Informer reads [strain_map.nxs](strain_map.nxs), creates/updates a derived HDF5 file in `/data`, and watches for location updates.
2. Data egress monitors the derived HDF5 and emits new measurement events.
3. Campaign orchestrator executes request/reply tasks from [conf/chess-autonomous-bootstrap.campaign.json](conf/chess-autonomous-bootstrap.campaign.json).
4. Instrument-control and Dial process their tasks and return replies.
5. Campaign completes when all task objectives are satisfied.

## Configuration Files

Note: The `conf/` directory is intentionally committed. It contains only test/local
credentials used by this Docker Compose deployment (for example,
`intersect_username` / `intersect_password`) and is not intended for production
secrets.

- Informer config: [conf/informer-config.yaml](conf/informer-config.yaml)
- Data service config: [conf/data-service.json](conf/data-service.json)
- Dial config: [conf/dial-service.json](conf/dial-service.json)
- Instrument control config: [conf/instrument-control-service.json](conf/instrument-control-service.json)
- Campaign payload: [conf/chess-autonomous-bootstrap.campaign.json](conf/chess-autonomous-bootstrap.campaign.json)

## Troubleshooting

### Duplicate campaign ID (`409`)

The bootstrap campaign file has a static `id`. Re-submitting the same payload can
return HTTP 409. Override `id` per run (see deterministic run section).

### Orchestrator submit race after restart

Immediately posting after restart can fail with transient connection reset.
Wait for `/v1/ping` before submitting.

### Dial userspace envelope validation errors

If Dial logs show messages like missing `messageId/operationId/headers/payload`,
use the current `dial:develop` image in [docker-compose.yml](docker-compose.yml).

### Startup AMQP connection refused logs

Short bursts of AMQP connection refused at startup are expected while RabbitMQ
becomes healthy. Services should recover and connect automatically.

## Volumes

| Volume | Mount | Purpose |
|---|---|---|
| shared-data | /data | Simulated shared/NFS-style data area |
| dial_mongodb_data | /bitnami/mongodb | Persist Dial state |

## Teardown

```bash
docker compose down
docker compose down -v
```
