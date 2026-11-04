# Benchmark inputs and method assets

Download `data.zip` (benchmark inputs and method assets) from the
[shared Google Drive folder](https://drive.google.com/drive/folders/1zxLWKwmC8yzDeXKC3Th7CwqjBchQbh4r?usp=drive_link). The same folder also contains
`3000_samples.zip`: 3,000 samples generated through method composition and decomposition.
Extract the input bundle as `data/` and the generated samples as `runs/` at the repository root.
See the [composition guide](COMPOSED_GENERATION.md) for sample usage.

Benchmark inputs and reusable method resources are stored separately. Neither
data nor run outputs are tracked by Git. Set `VLM_JAILBREAK_DATA_ROOT` to the
directory containing both sections, not to either section itself.

```text
data/
├── benchmarks/
│   ├── advbench/harmful_behaviors.csv
│   ├── AdvBenchM/                     # the former original_release/ contents
│   │   ├── dataset/{advbench,advimage}/
│   │   ├── instructions.jsonl         # 246 records, 231 unique instruction texts
│   │   ├── images.jsonl               # 160 primary images in eight categories
│   │   ├── manifest.json
│   │   └── availability.json
│   ├── HADES/{hades.csv,images/}
│   ├── MM-SafetyBench/
│   ├── SafeBench/safebench.csv
│   └── HarmBench/harmbench_standard.json  # optional JPS query/target split
└── method_assets/
    ├── CS-DJ/
    │   ├── llava_images/              # LLaVA-CC3M image pool
    │   ├── instructions/              # method-specific 750-instruction set
    │   └── Super Moods.ttf
    ├── JOOD/                          # former AdvBenchM/jood_subset/
    │   ├── images/{harmful,harmless,harmless_text}/
    │   ├── prompts/{eval_all_instructions,all_instructions,all_instructions_harmful_annotated}/
    │   ├── scenario_def.json
    │   └── scenario_repr.json
    ├── JailbreakInPieces/{meth_pill.png,white.jpeg}
    ├── VisualAdv/{derogatory_corpus.csv,rtp_prompts.jsonl}
    ├── nltk_data/taggers/             # English POS models for MML generation
    ├── MML/                           # optional; excluded from the default ZIP
    │   ├── upstream/const.py
    │   ├── safebench/<class>/{data.json,images_wr/,images_mirror/,images_rotate/,images_base64/}
    │   ├── mm-safebench/<class>/...
    │   └── hades/<class>/...
    ├── FORCE/clean.jpeg               # optional method input
    ├── JPS/{advbench_subset.csv,clean.jpeg}  # optional method inputs
    ├── UltraBreak/{images/,train_configs/}  # optional method inputs
    └── provenance/data_layout_migration.json
```

`runs/<experiment>/` contains generated attack images, model responses, fitted
patches and evaluation results. Those outputs do not belong in either data
section. A benchmark image distributed by its authors remains benchmark input;
an image pool or training corpus required by an attack belongs in method assets.

## AdvBench-M and JOOD

The original public AdvBench-M release is under `benchmarks/AdvBenchM/`.
It has 246 categorized question records (231 unique texts), 160 primary images,
and a separate 483-record prompt pool. Questions and images are associated by
category, not as one-to-one pairs. The released files do not contain all 240
images described in the paper. The source manifest and availability record
retain that limitation; no missing images or pairings are invented.

`method_assets/JOOD/` holds the complete former `jood_subset/`: 170 evaluation
questions in seven categories, seven representative harmful images and ten
mixing assets. To preserve historical experiments, the existing CLI dataset
name `AdvBenchM` still reads this JOOD subset. Reorganizing files does not replace
that evaluation set with the original release. The `CS-DJ` dataset loader also
continues reading its method-specific instruction set under method assets.

## Existing bundles and historical paths

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
