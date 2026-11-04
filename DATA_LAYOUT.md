# Benchmark inputs and method assets

Download `data.zip` (benchmark inputs and method assets) from the
[shared Google Drive folder](https://drive.google.com/drive/folders/1zxLWKwmC8yzDeXKC3Th7CwqjBchQbh4r?usp=drive_link). The same folder also contains
`3000_samples.zip`: 3,000 samples generated through method composition and decomposition.
Extract the input bundle as `data/` and the generated samples as `runs/` at the repository root.
See the [composition guide](COMPOSED_GENERATION.md) for sample usage.

Benchmark inputs and reusable method resources are stored separately. Neither
data nor run outputs are tracked by Git. Set `VLM_JAILBREAK_DATA_ROOT` to the
directory containing both sections, not to either section itself. The default
is `<repo>/data`. The tree below matches the currently prepared directories.

```text
data/
├── benchmarks/
│   ├── advbench/                # Original AdvBench inputs
│   ├── AdvBenchM/               # Original public AdvBench-M release
│   ├── HADES/                   # Original HADES inputs
│   ├── MM-SafetyBench/          # Original MM-SafetyBench inputs
│   └── SafeBench/               # Original SafeBench inputs
└── method_assets/
    ├── CS-DJ/                   # Image pool, instructions and font
    ├── JOOD/                    # JOOD subset and mixing images
    ├── JailbreakInPieces/       # Fixed source images
    ├── MML/                     # Official replay images and metadata
    ├── VisualAdv/               # Optimization target corpus
    ├── fonts/                   # Composition font
    ├── nltk_data/               # POS tagger for MML word replacement
    └── provenance/              # Source and migration records
```

`runs/<experiment>/` contains generated attack images, model responses, fitted
patches and evaluation results. Those outputs do not belong in either data
section. A benchmark image distributed by its authors remains benchmark input;
an image pool or training corpus required by an attack belongs in method assets.

## Benchmark files

Paths are relative to `data/`.

| Dataset | Path | Contents |
| --- | --- | --- |
| AdvBench | `benchmarks/advbench/harmful_behaviors.csv` | 520 instructions |
| AdvBenchM | `benchmarks/AdvBenchM/` | 246 question records, 160 category-pool images; `instructions.jsonl`, `images.jsonl`, `dataset/` |
| HADES | `benchmarks/HADES/` | `hades.csv` and `images/` |
| MM-SafetyBench | `benchmarks/MM-SafetyBench/` | Scenario JSONs and SD/SD_TYPO/TYPO images |
| SafeBench | `benchmarks/SafeBench/safebench.csv` | 500 questions |

## Method resources

Paths are relative to `data/method_assets/`.

| Directory | Contents / use |
| --- | --- |
| `CS-DJ/` | `llava_images/`, `instructions/` and `Super Moods.ttf` |
| `JOOD/` | `prompts/`, `images/` and scenario metadata |
| `JailbreakInPieces/` | Fixed source images for image optimization |
| `MML/` | Official replay images, metadata and upstream constants; downloaded separately |
| `VisualAdv/` | Target sentence corpus and prompt records |
| `fonts/` | Composition font and license, prepared by `python -m core.prepare_five_method_assets` |
| `nltk_data/` | Reusable POS-tagging resources for MML word replacement |
| `provenance/` | Data-layout migration and source records |

## AdvBench-M and JOOD

The original public AdvBench-M release is under `benchmarks/AdvBenchM/`.
It has 246 categorized question records (231 unique texts), 160 primary images,
and a separate 483-record prompt pool. Questions and images are associated by
category, not as one-to-one pairs. The released files do not contain all 240
images described in the paper. The source manifest and availability record
retain that limitation; no missing images or pairings are invented.

`method_assets/JOOD/` holds the complete former `jood_subset/`: 170 evaluation
questions in seven categories, seven representative harmful images and ten
mixing assets. Use `--dataset AdvBenchM` for the original 246-record benchmark
and `--dataset JOOD` for this separate 170-question subset. The original loader
keeps all 20 catalog images per category as a pool in `Behavior.images`, with
`meta.image_association=category_pool_not_one_to_one`; it preserves duplicate
question records and released targets. The `CS-DJ` loader reads its method-specific
instruction set under method assets.

## Download layout and migration

The shared folder's `data.zip` uses the separated `benchmarks/` and `method_assets/`
layout described above. To migrate an existing flat data directory, run:

```bash
export VLM_JAILBREAK_DATA_ROOT=/path/to/data
python -m core.migrate_data_layout --data-root "$VLM_JAILBREAK_DATA_ROOT" --dry-run
python -m core.migrate_data_layout --data-root "$VLM_JAILBREAK_DATA_ROOT"
```

The migration checks all destinations before moving files and refuses to
overwrite existing data. It moves original file bytes, removes the obsolete
JOOD compatibility symlinks at the old AdvBenchM wrapper, and preserves wrapper
metadata under `method_assets/JOOD/provenance/previous_AdvBenchM/`.
Re-running it on the separated layout makes no moves. Existing run records and
historical reproduction settings are left unchanged.

`core.paths` resolves new paths against the configured data root and accepts
legacy flat paths, including `AdvBenchM/original_release/` and
`AdvBenchM/jood_subset/`. The former maps to the benchmark release and the latter
to JOOD assets; they are never interchanged. Old absolute paths within the
selected data root are also remapped by the common resolver. Archived standalone
scripts that open old paths directly must be given the new paths explicitly.

## Method-specific preparation

- MML official replay requires the separately downloaded upstream `dataset`
  tree and hash-pinned `upstream/const.py` at commit `5966fca5`. Its images are
  excluded from the default ZIP. Generated word-replacement compositions use
  `method_assets/nltk_data/`, which remains in the ZIP.
- Set `NLTK_DATA="$VLM_JAILBREAK_DATA_ROOT/method_assets/nltk_data"`. The pinned
  NLTK 3.8.1 uses `averaged_perceptron_tagger`; newer versions may use the `_eng`
  resource. Both are reusable dependencies, not VLM training examples.
- CS-DJ needs its CC3M image pool, instructions and exact font. The default
  config samples 10,000 images and rejects font fallback.
- JPS needs the `goal,target` CSV and base image required by the selected
  config. Upstream does not distribute every referenced file; record the source
  of any separately supplied input.
- VisualAdv reads its target corpus under `method_assets/VisualAdv/`; its base
  image remains the selected FORCE asset. JailbreakInPieces requires the two
  fixed source images, whose expected hashes are recorded in provenance.

Validate selected datasets with `python -m core.validation.data --dataset NAME`;
add `--layout-only` to check presence without loading rows. Original AdvBench-M
files can additionally be checked against their own `manifest.json`.
