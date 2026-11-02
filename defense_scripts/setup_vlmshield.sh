#!/usr/bin/env bash
# Fetch and verify the released VLMShield classifier (pgqihere/VLMShield@344ca14).
set -euo pipefail
cd "$(dirname "$0")/.."

commit="344ca14778383aef57315229118f7669454fb4bb"
url="https://raw.githubusercontent.com/pgqihere/VLMShield/${commit}/VLMShield_script/model/vlmshield_classifier.pt"
destination="${VLMSHIELD_CLASSIFIER_PATH:-defenses/vlmshield_assets/vlmshield_classifier.pt}"
expected_size=8401604
expected_sha256="b36702877791f347f0b846d93b9b9162c86a8895efe962b6052d322ed0c53c34"

mkdir -p "$(dirname "$destination")"
if [ ! -f "$destination" ]; then
  curl -L --fail --retry 3 --output "$destination" "$url"
fi

actual_size=$(stat -c '%s' "$destination")
if [ "$actual_size" != "$expected_size" ]; then
  echo "VLMShield classifier size mismatch: expected $expected_size, got $actual_size" >&2
  exit 1
fi
actual_sha256=$(sha256sum "$destination" | awk '{print $1}')
if [ "$actual_sha256" != "$expected_sha256" ]; then
  echo "VLMShield classifier SHA-256 mismatch: expected $expected_sha256, got $actual_sha256" >&2
  exit 1
fi

PYTHON=${PYTHON:-python}
PYTHONNOUSERSITE=1 "$PYTHON" - <<'EOF'
from multimodalmodels.loaders import load_clip_model
model, processor = load_clip_model()
print(f"VLMShield CLIP ready: {model.config._name_or_path} (projection_dim={model.config.projection_dim})")
EOF
echo "VLMShield classifier ready: $destination"
echo "size=$actual_size sha256=$actual_sha256"
