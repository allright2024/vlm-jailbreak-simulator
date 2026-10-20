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
│   ├── images/{harmful,harmless}/
│   └── prompts/
│       ├── eval_all_instructions/
│       ├── all_instructions/
│       └── all_instructions_harmful_annotated/
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
├── JailbreakInPieces/
│   ├── meth_pill.png              # upstream fixed target
│   └── white.jpeg                 # upstream base image
├── MM-SafetyBench/
├── SafeBench/
│   └── safebench.csv
├── UltraBreak/
│   ├── images/white.jpeg
│   └── train_configs/safebench_white_force_jailbroken_mode.csv
└── VisualAdv/
    └── derogatory_corpus.csv
```

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

The two `JPS/` files are required by the sole `JPS.yaml`. The upstream
repository references both files but does not distribute them; preserve the
official `goal,target` CSV columns. The default JPS config instead keeps the
official optimization settings while using the local AdvBench loader and the
shared UltraBreak white image.

`CS-DJ/llava_images/` is required only when running CS-DJ. Download the
LLaVA-CC3M-Pretrain-595K image archive referenced by the upstream project and
place its image files directly in that directory. Copy the upstream five
`instructions/*.json` files and `Super Moods.ttf` as shown above. `CS-DJ.yaml`
samples 10,000 images and rejects font fallback.

`JailbreakInPieces/{meth_pill.png,white.jpeg}` are the two files under the
upstream repository's `images/` directory and are required by
`JailbreakInPieces.yaml`; their expected SHA-256 values are recorded in the
attack provenance.

`python -m core.validation.data` verifies expected row counts, unique behavior IDs, and all
dataset image references. Use `--layout-only` for a fast presence check or
repeat `--dataset NAME` to validate selected datasets only.

Run `python -m core.validation.official_reproduction` before a five-method official
matrix. CS-DJ, JOOD, and MML require `OPENAI_API_KEY`; JOOD and JPS also require
`HF_TOKEN` for the gated Llama Guard checkpoints. The validator checks presence
only and never prints credential values.
