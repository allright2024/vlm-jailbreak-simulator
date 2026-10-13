# External benchmark data

Benchmark data and generated artifacts are intentionally not tracked by Git.
Download the data bundle from the project Drive, then point the simulator at
its root before running a command:

```bash
export VLM_JAILBREAK_DATA_ROOT=/path/to/vlm-jailbreak-data
python validate_data.py
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
├── FORCE/
│   └── clean.jpeg
├── HADES/
│   ├── hades.csv
│   └── images/
├── MM-SafetyBench/
├── SafeBench/
│   └── safebench.csv
├── UltraBreak/
│   ├── images/white.jpeg
│   └── train_configs/safebench_white_force_jailbroken_mode.csv
└── legacy/
    └── derogatory_corpus.csv
```

Paths beginning with `data/` in attack YAML files are logical paths. When
`VLM_JAILBREAK_DATA_ROOT` is set, the resolver maps, for example,
`data/FORCE/clean.jpeg` to
`$VLM_JAILBREAK_DATA_ROOT/FORCE/clean.jpeg`. Absolute paths in custom configs
remain valid.

`validate_data.py` verifies expected row counts, unique behavior IDs, and all
dataset image references. Use `--layout-only` for a fast presence check or
repeat `--dataset NAME` to validate selected datasets only.
