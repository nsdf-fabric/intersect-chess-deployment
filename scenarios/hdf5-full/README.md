# HDF5 Full Dataset Scenario

Use this mode when you want to run with the full-size dataset (not committed to git).

## Prepare dataset

We have a full experiment Nexus file for an AM benchmark provided by Amlan Das (ID1A3 instrument scientist)

Path on CHESS NFS: `/nfs/chess/user/ad785/data/analysis/user_data/yeung-3714-b/analysis/v8-p3-10s-0deg/output/strain_map.nxs`

Must contact team for this repo to get a copy, not publically available.

## Run

```bash
docker compose -f docker-compose.yml -f scenarios/hdf5-full/docker-compose.override.yml up -d --force-recreate
```

## Campaign payload

`scenarios/hdf5-full/chess-autonomous-bootstrap.campaign.json`
