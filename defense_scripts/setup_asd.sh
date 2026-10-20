#!/usr/bin/env bash
# Download and verify the universal protective image released by AsD.
set -euo pipefail
cd "$(dirname "$0")/.."

commit="ba71efe0972c92e38da156ec76911f9122bbe12f"
filename="L2_noNorm_clipgrad_tips_336_LR0_1-1_1.png"
destination="defenses/asd_assets/$filename"
expected_sha256="f02b3667c4140e9dc2bc1cafba24d8a3f6764265555904104cefd93b60df37d3"
url="https://raw.githubusercontent.com/AngelAlita/AsD/$commit/images/$filename"

mkdir -p "$(dirname "$destination")"
if [ ! -f "$destination" ]; then
  curl -L --fail --retry 3 --output "$destination" "$url"
fi

actual_sha256=$(sha256sum "$destination" | awk '{print $1}')
if [ "$actual_sha256" != "$expected_sha256" ]; then
  echo "AsD patch SHA-256 mismatch: expected $expected_sha256, got $actual_sha256" >&2
  exit 1
fi

echo "AsD patch ready: $destination"
echo "sha256=$actual_sha256"
