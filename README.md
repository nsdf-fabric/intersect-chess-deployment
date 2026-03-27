# CHESS Autonomous Experiment Deployment

Full-stack deployment for the CHESS autonomous experiment feedback loop using
Docker Compose. Pulls pre-built container images from GHCR and runs the entire
workflow locally.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Campaign Orchestrator (dashboard on :8000)                                  │
│  Coordinates the full autonomous experiment loop                            │
└───────────┬──────────────────────┬───────────────────────┬──────────────────┘
            │                      │                       │
            ▼                      ▼                       ▼
┌───────────────────┐  ┌───────────────────┐  ┌──────────────────────────────┐
│ chess-data-egress │  │       Dial        │  │ chess-instrument-control     │
│    (data-service) │  │ (active learning) │  │      (spec-service)         │
│                   │  │                   │  │                             │
│ Monitors NEW HDF5 │  │ Bayesian opt.    │  │ Writes loc001.txt files     │
│ Emits events      │  │ Recommends next  │  │ for SPEC to read            │
│                   │  │ measurement point│  │                             │
└───────────────────┘  └───────────────────┘  └──────────────────────────────┘
            ▲                                              │
            │           shared-data volume (/data)         │
            │    ┌──────────────────────────────┐          │
            └────│  chess-instrument-control-   │──────────┘
                 │  informer                    │
                 │  Full HDF5 → New HDF5        │
                 │  watches loc_dir for new pts │
                 └──────────────────────────────┘
```

## Quick Start

```bash
docker compose up
```

This spins up:

| Service | Image | Port | Role |
|---------|-------|------|------|
| **broker** | `bitnamilegacy/rabbitmq:3.13.3` | 5672, 15672 | AMQP message bus |
| **mongodb** | `bitnamilegacy/mongodb:8.0` | 27017 | Dial state store |
| **informer** | `ghcr.io/nsdf-fabric/chess-instrument-control-informer:main` | — | Full HDF5 → New HDF5 + watch |
| **chess-data-service** | `ghcr.io/nsdf-fabric/intersect-chess-data-service:main` | — | HDF5 monitor → events |
| **dial** | `ghcr.io/intersect-dial/dial:main` | — | Active learning |
| **chess-instrument-control-service** | `ghcr.io/nsdf-fabric/intersect-chess-instrument-control-service:main` | — | Write motor positions |
| **orchestrator** | built locally | **8000** | Campaign coordinator + dashboard |

## Data Flow

1. **Informer** reads `strain_map.nxs` (bind-mounted), creates `new_strain_map.nxs` with 5 random initial points, and watches `/data/autonomous_experiment/experiment1/` for new location files
2. **Data service** monitors `new_strain_map.nxs` via SWMR polling and emits `new_measurement` events
3. **Orchestrator** receives events and sends `{next_x: [labx, labz], next_y: center_value}` to Dial
4. **Dial** performs Bayesian optimization and recommends the next `[labx, labz]` to measure
5. **Orchestrator** sends the recommendation to the instrument control service
6. **Instrument control service** writes `locNNN.txt` to the shared volume
7. **Informer** detects the new location file, looks up the nearest point in the full HDF5, and appends the interpolated data to `new_strain_map.nxs`
8. **Loop** back to step 2

## Configuration

### Informer config ([informer-config.yaml](informer-config.yaml))
- `full_file`: Source HDF5 (strain_map.nxs)
- `new_file`: Output HDF5 that gets populated
- `loc_dir`: Directory the informer watches for new location files
- `initial_count` / `seed`: Number of random initial points

### Campaign config ([campaign-conf.json](campaign-conf.json))
- Service INTERSECT hierarchy destinations
- `hdf5_filename`: Path to the new HDF5 (must match informer's `new_file`)
- `dataset_path`: HDF5-internal path to monitor (detector/fit/hkl)
- `initial_dataset_x/y`: Initial training data for Dial
- `bounds`: labx/labz bounds for optimization

### Service configs ([conf/](conf/))
- `data-service.json`, `instrument-control-service.json`, `dial-service.json`
- Each contains broker connection and INTERSECT hierarchy identity

## Dashboard

Open http://localhost:8000 after `docker compose up`. The dashboard shows:
- Broker connection status
- Campaign iteration counter
- 2D plot of measured points, next recommendation, and surrogate model heatmap

## Volumes

| Volume | Mount | Purpose |
|--------|-------|---------|
| `shared-data` | `/data` | Simulates NFS: holds strain_map.nxs, new_strain_map.nxs, and autonomous_experiment/ |
| `dial_mongodb_data` | `/bitnami/mongodb` | Persists Dial workflow state |

## Teardown

```bash
docker compose down -v   # removes containers + volumes
```
