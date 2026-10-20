# MML adapter

This package integrates **Multi-Modal Linkage (MML)** from
`wangyu-ovo/MML@5966fca529011e8b5ccff61d70365997d6466abc`.

The sole shipped config uses `precomputed` mode and replays the released Drive
images. It also reads the exact attack and CLAS judge prompts from a hash-pinned
checkout of the upstream repository. The source is parsed as literals and is
never imported or executed because upstream declares no software license.

Prepare the two external inputs as follows:

```bash
git clone https://github.com/wangyu-ovo/MML.git data/MML/upstream
git -C data/MML/upstream checkout 5966fca529011e8b5ccff61d70365997d6466abc
# Download the Drive `dataset` contents into data/MML/{safebench,mm-safebench,hades}
```

`run_generate_completions.py --protocol mml_official` implements the released online control flow: query the
target, apply the `Content`/refusal gates, obtain a CLAS score from the pinned
GPT-4o-mini snapshot, stop immediately at score 5, and otherwise retain the
highest score after five queries. It reports ASR, reconstruction success rate,
mean query count, score distribution, and per-category ASR. Full runs fail
closed unless they contain exactly 500 SafeBench, 1,180 filtered
MM-SafetyBench, or 750 HADES behaviors. `LIMIT=N` in the shell runner marks an
explicit smoke subset.

The protocol implementation lives in `core/protocols/mml.py`; there is no
separate MML root entry point. The shared CLI writes completions, judged
results, and the MML summary in one run. Its MML defaults remain 2,000 target
tokens, temperature 0.7, three target retries, and five attempts. The shell
wrapper explicitly selects the pinned target and CLAS model snapshots.
`--save_path` (also accepted as `--completions_path`), `--results_path`,
`--summary_path`, and `--asr_results_file` retain output-path control.
Use Stage 1 behavior limits with `--allow_subset`, not Stage 2 `--limit`.
The official protocol remains baseline-only. For defense comparisons, use
the standard generation/evaluation path on both sides with matching budgets.

An explicit custom `generated` config can create word-replacement, horizontally
mirrored, 180-degree rotated, Base64, or the paper's Section 5.6 shift-cipher
typographic images from a simulator `Behavior`. Image rendering settings were
not released, so generated mode is not an official reproduction profile.

The upstream repository does not release its image-generation code or rendering
settings. It instead links to an external Drive folder containing prepared
images and `data.json` files. For exact asset replay, download that folder to
`data/MML/` and use the default `configs/attacks/MML.yaml`; the expected structure
is documented in `DATA_LAYOUT.md`. The official metadata is also the filter:
rows absent from it—including MM-SafetyBench rows removed by CLAS moderation—are
reported as skipped rather than aborting the run.

Word-replacement generation uses the paper's sequential adjective and noun
candidate lists and NLTK POS tags. Install the English tagger once with:

```bash
# requirements.txt pins NLTK 3.8.1
python -m nltk.downloader averaged_perceptron_tagger

# NLTK 3.9+ uses the language-specific resource instead
python -m nltk.downloader averaged_perceptron_tagger_eng
```

`pos_tagger: heuristic` is available as an explicitly adapted, offline fallback.
The upstream repository has no explicit software license; see `THIRD_PARTY.md`.
