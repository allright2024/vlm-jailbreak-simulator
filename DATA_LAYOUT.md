# External benchmark data

Benchmark data and generated artifacts are intentionally not tracked by Git.
Download the data bundle from the project Drive, then point the simulator at
its root before running a command:

```bash
export VLM_JAILBREAK_DATA_ROOT=/path/to/vlm-jailbreak-data
python -m core.validation.data
```

The external directory must have this layout:

```text
vlm-jailbreak-data/
├── advbench/
│   └── harmful_behaviors.csv
├── AdvBenchM/
│   ├── jood_subset/               # existing AdvBenchM loader: 170 questions
│   │   ├── images/{harmful,harmless,harmless_text}/
│   │   ├── prompts/{eval_all_instructions,all_instructions,all_instructions_harmful_annotated}/
│   │   ├── scenario_def.json
│   │   └── scenario_repr.json
│   └── original_release/          # independent public release; optional
│       ├── dataset/{advbench,advimage}/
│       ├── instructions.jsonl     # 246 categorized records (231 unique texts)
│       └── images.jsonl           # 160 primary images in 8 categories
├── CS-DJ/
│   ├── llava_images/               # LLaVA-CC3M-Pretrain-595K images
│   ├── instructions/               # five JSON files from the pinned repo
│   └── Super Moods.ttf
├── FORCE/
│   └── clean.jpeg
├── HADES/
│   ├── hades.csv
│   └── images/
├── HarmBench/
│   └── harmbench_standard.json     # JPS query/target evaluation split
├── JPS/
│   ├── advbench_subset.csv
│   └── clean.jpeg
├── MM-SafetyBench/
├── SafeBench/
│   └── safebench.csv
└── VisualAdv/
    └── derogatory_corpus.csv
```

The dataset name `AdvBenchM` continues to load JOOD's 170 questions across
7 categories, with one representative harmful image per category. Collecting
`original_release/` does not switch the experiment to a different dataset.
Its 246 records and 160 images form category-level pools, not one-to-one pairs.

The loader, path resolver and data validator also accept older Drive bundles
with `AdvBenchM/{prompts,images,scenario_*.json}` directly under the root.
For migrated local data, relative compatibility symlinks at those old paths
keep existing experiment manifests and absolute image paths usable. New
configurations use `AdvBenchM/jood_subset/`. Historical settings and results
retain the paths recorded when they were produced. The published `data.zip`
has not been updated by this local migration.

VisualAdv reads its optimization-target corpus from
`data/VisualAdv/derogatory_corpus.csv`; its base image remains
`data/FORCE/clean.jpeg`. Older bundles kept this CSV under `legacy/`: relocate
the CSV to `VisualAdv/` when upgrading. The remaining `legacy/` data is not
required by the registered dataset loaders or the current composition plan.
Archived pre-refactor runners that request other HarmBench files require those
optional inputs to be supplied separately; they are not part of this layout.

For MML, download the public `dataset` directory linked by the upstream README
and preserve this layout under `data/MML/` (or set
`precomputed_root` to an equivalent location):

```text
MML/
├── upstream/
│   └── const.py                     # pinned repo @ 5966fca5; hash-verified
├── safebench/
│   └── <class>/
│       ├── data.json
│       ├── images_wr/<id>.png
│       ├── images_mirror/<id>.png
│       ├── images_rotate/<id>.png
│       └── images_base64/<id>.png
├── mm-safebench/
│   └── <class>/...
└── hades/
    └── <class>/
        ├── data.json
        └── images_<variant>/<id>.jpg
```

The adapter indexes `data.json` by `original_prompt` and matches it against the
normalized simulator behavior's `statement` and `instruction`. Rows missing
from the released metadata are explicitly skipped. This is required for the
paper's MM-SafetyBench subset: CLAS filtering retains 1,180 of 1,680 queries.
The four exact attack prompts and exact CLAS judge prompt are parsed from the
hash-pinned `upstream/const.py` without executing it.

Paths beginning with `data/` in attack YAML files are logical paths. When
`VLM_JAILBREAK_DATA_ROOT` is set, the resolver maps, for example,
`data/FORCE/clean.jpeg` to
`$VLM_JAILBREAK_DATA_ROOT/FORCE/clean.jpeg`. Absolute paths in custom configs
remain valid.

The two `JPS/` files are required by `JPS.yaml`. Obtain the training CSV
linked by the JPS README and preserve its `goal,target` columns. The original
`clean.jpeg` is not distributed; record the provenance of the supplied base image.

`CS-DJ/llava_images/` is required only when running CS-DJ. Download the
LLaVA-CC3M-Pretrain-595K image archive referenced by the upstream project and
place its image files directly in that directory. Copy the upstream five
`instructions/*.json` files and `Super Moods.ttf` as shown above. `CS-DJ.yaml`
samples 10,000 images and rejects font fallback.

`python -m core.validation.data` verifies expected row counts, unique behavior IDs, and all
dataset image references. Use `--layout-only` for a fast presence check or
repeat `--dataset NAME` to validate selected datasets only.

Historical experiment settings are retained under
`configs/experiments/reproduction_settings/`. CS-DJ, JOOD and MML require
API credentials when their selected backend uses remote services; supply
credentials through environment variables, never configuration snapshots.
