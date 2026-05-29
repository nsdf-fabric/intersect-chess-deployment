# HDF5 Scenario

Use the HDF5-based informer flow with the committed reduced fixture
`datasets/strain_map.small.nxs` -> generated `/data/new_strain_map.nxs`.

This is the recommended quick-start mode for contributors and CI.

## Run

```bash
docker compose -f docker-compose.yml -f scenarios/hdf5/docker-compose.override.yml up -d --force-recreate
```

## Campaign payload

`scenarios/hdf5/chess-autonomous-bootstrap.campaign.json`

## Fixture size

- `datasets/strain_map.small.nxs` is approximately 9 MB.
