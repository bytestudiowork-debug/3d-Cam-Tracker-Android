#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_VERSION="v0.66.0"
MODEL_URL="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/superpoint/releases/${MODEL_VERSION}/superpoint-tflite-float.zip"
ASSET_DIR="${ROOT_DIR}/app/src/main/assets/ai"
ASSET_PATH="${ASSET_DIR}/feature_model.tflite"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

command -v curl >/dev/null || { echo "curl is required." >&2; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required." >&2; exit 1; }

echo "Downloading the pinned SuperPoint TFLite release (${MODEL_VERSION})..."
curl --fail --location --retry 3 --retry-delay 2 \
  --output "${TEMP_DIR}/superpoint.zip" "${MODEL_URL}"

mapfile -t MODEL_ENTRIES < <(unzip -Z1 "${TEMP_DIR}/superpoint.zip" | grep -Ei '\.tflite$' || true)
if [[ "${#MODEL_ENTRIES[@]}" -ne 1 ]]; then
  echo "Expected exactly one TFLite model in the pinned archive; found ${#MODEL_ENTRIES[@]}." >&2
  unzip -Z1 "${TEMP_DIR}/superpoint.zip" >&2
  exit 1
fi

mkdir -p "${ASSET_DIR}"
unzip -p "${TEMP_DIR}/superpoint.zip" "${MODEL_ENTRIES[0]}" > "${ASSET_PATH}"
if [[ ! -s "${ASSET_PATH}" ]]; then
  echo "Downloaded TFLite model is empty." >&2
  exit 1
fi

echo "Installed model asset: ${ASSET_PATH}"
sha256sum "${ASSET_PATH}"
