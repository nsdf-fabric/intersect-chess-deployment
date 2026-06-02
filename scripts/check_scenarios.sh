#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

NEUTRON_DATASETS=(
  "A2_2p1_30layers_VDriveSPF_LD.json"
  "D1_1p8_30layers_VDriveSPF_LD.json"
  "D2_2p3_30layers_VDriveSPF_LD.json"
  "D4_4p4_30layers_VDriveSPF_LD.json"
)

status=0

check_campaign_json() {
  local campaign_file="$1"
  if [[ ! -f "$campaign_file" ]]; then
    echo "[ERROR] Missing campaign file: $campaign_file"
    status=1
    return
  fi

  if ! jq -e '.id and .name and (.task_groups | type == "array") and ((.task_groups | length) > 0)' "$campaign_file" >/dev/null; then
    echo "[ERROR] Invalid campaign structure: $campaign_file"
    status=1
    return
  fi

  echo "[OK] campaign: $campaign_file"
}

check_compose() {
  local compose_file="$1"
  if [[ ! -f "$compose_file" ]]; then
    echo "[ERROR] Missing compose file: $compose_file"
    status=1
    return
  fi

  if ! docker compose -f "$compose_file" config >/dev/null; then
    echo "[ERROR] Compose validation failed: $compose_file"
    status=1
    return
  fi

  echo "[OK] compose: $compose_file"
}

check_neutron_variants() {
  local compose_file="$1"
  for dataset in "${NEUTRON_DATASETS[@]}"; do
    if ! NEUTRON_DATASET_JSON="$dataset" docker compose -f "$compose_file" config >/dev/null; then
      echo "[ERROR] Neutron dataset config failed: $compose_file with NEUTRON_DATASET_JSON=$dataset"
      status=1
      return
    fi
  done

  echo "[OK] neutron variants: $compose_file"
}

while IFS= read -r compose_file; do
  scenario_dir="$(dirname "$compose_file")"
  campaign_file="$scenario_dir/campaign.json"

  check_compose "$compose_file"
  check_campaign_json "$campaign_file"

  case "$scenario_dir" in
    *neutron_ammdf_vulcan_*)
      check_neutron_variants "$compose_file"
      ;;
  esac
done < <(find scenarios -mindepth 2 -maxdepth 2 -name docker-compose.yml | sort)

if [[ "$status" -ne 0 ]]; then
  echo "Scenario checks failed."
  exit 1
fi

echo "All scenario checks passed."
