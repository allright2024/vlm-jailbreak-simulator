#!/usr/bin/env bash
# Download and verify the official guided-diffusion checkpoint used by CIDER.
set -euo pipefail
cd "$(dirname "$0")/.."

url="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
destination="${CIDER_DENOISER_PATH:-defenses/cider_models/diffusion_denoiser/imagenet/256x256_diffusion_uncond.pt}"
expected_size=2211383297
expected_md5="fd9dd2335b8736d521de0aed54bd90ca"

mkdir -p "$(dirname "$destination")"
curl -L --fail --retry 3 --continue-at - --output "$destination" "$url"

actual_size=$(stat -c '%s' "$destination")
if [ "$actual_size" != "$expected_size" ]; then
  echo "CIDER checkpoint size mismatch: expected $expected_size, got $actual_size" >&2
  exit 1
fi

actual_md5=$(md5sum "$destination" | awk '{print $1}')
if [ "$actual_md5" != "$expected_md5" ]; then
  echo "CIDER checkpoint MD5 mismatch: expected $expected_md5, got $actual_md5" >&2
  exit 1
fi

echo "CIDER checkpoint ready: $destination"
echo "size=$actual_size md5=$actual_md5"
