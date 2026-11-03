# Five-method composed dataset

This adds a separate adapted-composition generator. Existing official-profile
attack renderers and previous runs are untouched. All target responses and ASR
measurements are outside this generation workflow.

The plan has nine pairs at 160 samples each, seven triples at 180 each and two
quadruples at 150 each: 3,000 final model inputs. Each recipe shares its first
150 parents. The three preview parents are the longest, median and shortest
source statements from the harmful-only SafeBench pool. JPS training questions
and duplicate normalized source questions are excluded. All derivatives of one
source instruction share a deterministic train/validation/test split.

## Run

Use the simulator environment and install the additional font-inspection dependency:

```bash
/home/jovyan/.conda/envs/jb_env/bin/python -m pip install -r requirements-five-methods.txt
bash attack_scripts/run_five_method_dataset.sh preview
# To change only JOOD mixing while reusing the existing, verified source renders:
# bash attack_scripts/run_five_method_dataset.sh preview --reuse-from runs/five_methods_3000_20261025
```

Inspect all 18 images under
`runs/five_methods_3000_jood_image_mix_20261103/preview/contact_sheets/`, and the original PNGs
under `preview/cases/`. There are three actual composed inputs per recipe,
54 in total. Review the task cards, bottom lines, numbering and JPS auxiliary
image. Then record actual observations and generate the final dataset:

```bash
bash attack_scripts/run_five_method_dataset.sh review-preview --review-note 'Describe actual visual inspection here'
bash attack_scripts/run_five_method_dataset.sh full
bash attack_scripts/run_five_method_dataset.sh validate
```

The `full` phase checks the preview image hashes, contact-sheet hashes, sample
counts and visual-review record. It refuses to proceed without a review tied
to the unchanged preview. Re-running a phase reuses verified individual cases;
configuration, code, font, source CSV or JPS artifact changes require a new
`--output` directory. One process holds an exclusive output lock. CS-DJ caches
real question decomposition and real CLIP contrast retrieval across recipes.
`--reuse-from` verifies the completed source manifest, original parent selection,
rendered-image hashes, layouts, source plan and JPS artifacts. It copies the
pre-JOOD renders into the new run and records their provenance. Subsequent
commands recover the reuse source from the new run identity automatically.
The existing brightness/contrast-only run is kept unchanged.

## Continue after closing VS Code or SSH

After the preview has been reviewed, launch production in a detached session:

```bash
bash attack_scripts/run_five_method_dataset_background.sh start
bash attack_scripts/run_five_method_dataset_background.sh status
tail -f runs/five_methods_3000_jood_image_mix_20261103/logs/worker.log
```

The launcher uses `nohup`, `setsid`, `/dev/null` stdin and a persistent log.
Closing VS Code, the terminal or the SSH session does not stop the worker.
`start` detects an existing matching worker and will not launch a second one.
The generator also holds its own exclusive output lock. The worker rebuilds
the review gallery after all 3,000 inputs pass validation. PID, status, exit
code and timestamps are saved in the output directory.

```bash
bash attack_scripts/run_five_method_dataset_background.sh stop
# Later, resume completed sample checkpoints with the same start command.
bash attack_scripts/run_five_method_dataset_background.sh start
```

## Text preservation and adaptations

Text is measured in pixels, including font overhangs and descenders. Word wrap
falls back to character wrap for unbroken identifiers. Every non-whitespace
character is preserved. Missing glyphs and oversized text fail explicitly.
Fonts shrink only to 28 px, followed by bounded canvas-height growth. Task cards
are pasted at native size into the collage; they are never squeezed into a
fixed-height tile. All line boxes and panel labels are checked in final image
coordinates. At a 2,048 px longest-side inspection size, the calculated font
size must remain at least 12 px. This is a layout/readability threshold, not an
OCR guarantee or a guarantee about an unspecified target processor's resizing.

FigStep uses the numbered-completion mechanism with this measured renderer.
MML uses the existing NLTK word-replacement and reconstruction implementation,
with the measured renderer. CS-DJ retains Qwen question decomposition, CLIP
greedy contrast retrieval and the nine-image/three-card collage.

JOOD now performs actual image mixing with the five provided harmless images
under `data/AdvBenchM/images/harmless`: airpodmax, apple, iphone, soccerball and
watch. Each source behavior deterministically selects one provided image.
MixUp uses `0.7 * source + 0.3 * harmless`, with the weight configured under
`jood.source_weight`. The provided image is resized with its aspect ratio intact
and white letterboxing to match the source canvas; the source text/image is
never downscaled, cropped or moved. Keeping the native source resolution is an
explicit adaptation of JOOD's released 320px preprocessing. This is image
MixUp, not the previous brightness/contrast-only profile. Each sample records
both input image paths/hashes, the selected harmless image, mixing weight,
output hash and unchanged canvas size. The five images are copied to
`assets/jood/` and included in the run identity.

The 11 JOOD-containing recipes account for 1,840 of the 3,000 production
inputs. For CS-DJ combinations, mixing affects the whole 12-panel collage.
For FigStep/MML + JOOD + JPS, mixing affects the first task image and JPS remains
the second image. JOOD + JPS mixes the JPS image itself with a provided image.

JPS image, steering text and training CSV must match the recorded hashes in an
existing training run's result manifest. For combinations with another visual
method, the intact JPS image is appended as a second input image and steering
is prepended; this is an adapted transfer experiment requiring multi-image
target support. It is not a fresh joint optimization of the composed image.
The JOOD+JPS pair instead augments the JPS image and uses the existing JPS
prompt contract. Every sample is marked `adapted_composition`, never an exact
paper reproduction or an automatically successful jailbreak.

## Outputs

`preview/test_cases.jsonl` contains 54 preview inputs.
`full/test_cases.jsonl` contains exactly 3,000 production inputs, each with
source ID, recipe, methods, split, image hashes, text line bounds, JOOD operation
and JPS artifact provenance. `parents.json` retains original scoring truth.
Each phase writes `manifest.json`, `qc.json`, `status.json` and per-case JSON
checkpoints. Preview images are separate from the 3,000 production quota.
