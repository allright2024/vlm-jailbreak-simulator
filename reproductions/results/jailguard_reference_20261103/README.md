# JailGuard reference-repository experiment

This run executes the official `JailGuard/main_img.py` body from commit `419e1d3d08f4d02998a11a687f925518d2d5dd0c`, with its actual mutation, response-file, similarity, KL, and detection functions. The only source edit in `main_img_runtime.py` removes the hardcoded physical GPU index. See `main_img_runtime.diff`.

The model is the original `reference_repos/VisualAdv_upstream/minigpt4/models/mini_gpt4.py`, including its own Llama implementation and original `Chat` conversation code. `low_resource=True` gives an actual 8-bit Vicuna-13B and CPU FP32 vision processing. All eight variants are queried sequentially using the original `model_inference` function. No simulator inference or detection code is imported.

`worker.py` supplies local paths, isolates the nested argparse call, caches initialized models, adds deterministic per-item seeds, and records instrumentation. `supervise.py` runs four preselected smoke items (one per label), validates runtime behavior without selecting by accuracy, then launches two workers over all 1,000 official items (600 benign, 400 attacks). Smoke results are excluded from final metrics. Completion automatically writes `comparison.json` and `comparison.md` against the two historical simulator runs and paper Table 4 PL. This is detection evaluation, not an ASR evaluation.

The original tracked reference sources are unchanged. Source hashes, patches and limitations are recorded in `manifest.json`. Modern CUDA/Torch/bitsandbytes are retained for B200 support, while Transformers 4.28.0 matches the original MiniGPT-4 dependency environment. This cannot establish that every published runtime detail is reproduced or attribute changes to one variable.

The supervisor is detached from the terminal with stdin disconnected and logs redirected; closing VSCode will not stop it while the server remains running. Check `status.json`, `*.status.json` and `logs/`.

Compatibility: `minigpt_utils_runtime.py` removes only the unused Gradio UI import from the original helper (see its `.diff`). Its model loading and inference functions are unchanged. Figures are closed after each item to avoid accumulating the official heatmap figures across the full dataset.
