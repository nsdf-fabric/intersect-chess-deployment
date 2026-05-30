#!/usr/bin/env python3
"""
Polls nsdf-storage-service outputs in MinIO and converts them to the CHESS strain JSON
shape expected by the collaborator dashboard (ornl_chess_strain_lib).

Source schema (NewMeasurementData written by nsdf-storage-service):
  {
    "dataset_x": [[x0, z0], [x1, z1], ...],   # 2-D motor positions
    "dataset_y": [y0, y1, ...],                # strain measurements
    ...
  }

Target schema (strain JSON consumed by the dashboard):
  {
    "labx":                    [x0, x1, ...],
    "labz":                    [z0, z1, ...],
    "0/data/uniform_strain":   [y0, y1, ...],
    # optional GP columns when surrogate.json counts match:
    "0/data/uniform_strain_gp_estimate":  [...],
    "0/data/uniform_strain_gp_variance":  [...],
  }

Environment variables:
  MINIO_ENDPOINT_URL       default: http://minio:9000
  AWS_ACCESS_KEY_ID        default: minioadmin
  AWS_SECRET_ACCESS_KEY    default: minioadmin
  SRC_BUCKET               default: nsdf-storage
  SRC_KEY                  default: nsdf-storage/data.json
  SRC_SURROGATE_KEY        default: nsdf-storage/surrogate.json
  DST_BUCKET               default: scientistcloud
  DST_KEY                  default: IDX_TEST/ORNL_strain/live_data.json
  TRANSFORM_POLL_INTERVAL_S  default: 5
"""

import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ENDPOINT_URL = os.environ.get("MINIO_ENDPOINT_URL", "http://minio:9000")
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")

SRC_BUCKET = os.environ.get("SRC_BUCKET", "nsdf-storage")
SRC_KEY = os.environ.get("SRC_KEY", "nsdf-storage/data.json")
SRC_SURROGATE_KEY = os.environ.get("SRC_SURROGATE_KEY", "nsdf-storage/surrogate.json")

DST_BUCKET = os.environ.get("DST_BUCKET", "scientistcloud")
DST_KEY = os.environ.get("DST_KEY", "IDX_TEST/ORNL_strain/live_data.json")

POLL_INTERVAL = float(os.environ.get("TRANSFORM_POLL_INTERVAL_S", "5"))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_s3() -> "boto3.client":
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="us-east-1",
    )


def ensure_bucket(s3, bucket: str) -> None:
    try:
        s3.create_bucket(Bucket=bucket)
        log.info("Created bucket %s", bucket)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise


def get_etag(s3, bucket: str, key: str) -> str | None:
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        return head.get("ETag")
    except ClientError:
        return None


def get_json(s3, bucket: str, key: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("NoSuchKey", "NoSuchBucket", "404"):
            return None
        raise


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def transform(data: dict, surrogate: dict | None) -> dict | None:
    """Convert NewMeasurementData → strain JSON.  Returns None if data is empty."""
    dataset_x = data.get("dataset_x") or []
    dataset_y = data.get("dataset_y") or []

    if not dataset_x or not dataset_y:
        log.warning("data.json has empty dataset_x or dataset_y; skipping")
        return None

    dim = len(dataset_x[0]) if isinstance(dataset_x[0], list) else 1
    labx = [row[0] for row in dataset_x]
    labz = [row[1] for row in dataset_x] if dim >= 2 else [0.0] * len(dataset_x)

    out: dict = {
        "labx": labx,
        "labz": labz,
        "0/data/uniform_strain": dataset_y,
    }

    # Include GP estimate and variance when surrogate count matches measurements
    if surrogate:
        est = surrogate.get("surrogate")
        unc = surrogate.get("uncertainty")
        n = len(dataset_y)
        if isinstance(est, list) and len(est) == n:
            out["0/data/uniform_strain_gp_estimate"] = est
        if isinstance(unc, list) and len(unc) == n:
            # variance = stdev^2 (uncertainty is interpreted as stdev)
            out["0/data/uniform_strain_gp_variance"] = [u * u for u in unc]

    return out


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main() -> None:
    s3 = make_s3()
    ensure_bucket(s3, DST_BUCKET)

    log.info(
        "Transform loop started: %s/%s -> %s/%s  poll=%.1fs",
        SRC_BUCKET, SRC_KEY, DST_BUCKET, DST_KEY, POLL_INTERVAL,
    )

    last_etag: str | None = None

    while True:
        try:
            etag = get_etag(s3, SRC_BUCKET, SRC_KEY)
            if etag is None:
                log.debug("Source not yet available: s3://%s/%s", SRC_BUCKET, SRC_KEY)
                time.sleep(POLL_INTERVAL)
                continue

            if etag == last_etag:
                time.sleep(POLL_INTERVAL)
                continue

            data = get_json(s3, SRC_BUCKET, SRC_KEY)
            if not data:
                time.sleep(POLL_INTERVAL)
                continue

            surrogate = get_json(s3, SRC_BUCKET, SRC_SURROGATE_KEY)
            result = transform(data, surrogate)
            if result is None:
                time.sleep(POLL_INTERVAL)
                continue

            body = json.dumps(result, indent=2, allow_nan=True).encode("utf-8")
            s3.put_object(
                Bucket=DST_BUCKET,
                Key=DST_KEY,
                Body=body,
                ContentType="application/json",
            )
            last_etag = etag
            n = len(result.get("0/data/uniform_strain") or [])
            log.info(
                "Wrote strain JSON -> s3://%s/%s  (%d points, gp=%s)",
                DST_BUCKET, DST_KEY, n,
                "0/data/uniform_strain_gp_estimate" in result,
            )

        except Exception:
            log.exception("Unexpected error in transform loop")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
