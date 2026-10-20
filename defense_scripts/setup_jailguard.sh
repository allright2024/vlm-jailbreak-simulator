#!/usr/bin/env bash
# Install the official JailGuard semantic-similarity dependencies.
set -euo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

if ! "$PYTHON" -c 'import spacy; assert spacy.__version__ == "3.7.2"' >/dev/null 2>&1; then
  "$PYTHON" -m pip install 'spacy==3.7.2' 'nltk==3.8.1'
fi

if ! "$PYTHON" -c 'import spacy; spacy.load("en_core_web_md")' >/dev/null 2>&1; then
  model_version=${JAILGUARD_SPACY_MODEL_VERSION:-3.7.1}
  "$PYTHON" -m pip install \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-${model_version}/en_core_web_md-${model_version}-py3-none-any.whl"
fi

"$PYTHON" -c 'import spacy; nlp=spacy.load("en_core_web_md"); print(f"JailGuard ready: spaCy={spacy.__version__}, vectors={nlp.vocab.vectors_length}")'
