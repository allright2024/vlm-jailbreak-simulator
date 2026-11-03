# JailGuard 8-bit follow-up (2026-11-02)

Purpose: investigate the historical FP16 reproduction gap using a quantized Vicuna-13B v0 language model. This is an 8-bit ablation, not a claim that every upstream runtime difference is resolved.

- Dataset: all 1,000 official JailGuard image items (600 benign, 400 attacks).
- Protocol: paper Table 4 PL; eight variants; threshold 0.025; spaCy first 100 characters.
- Unchanged: local MiniGPT-4/Vicuna weights, prompt, mutation seed 2026, generation settings, batched eight-variant decoding, GPU vision path and detector.
- Changed: Vicuna LLM linear layers use bitsandbytes LLM.int8 instead of FP16. Vision/projector and unquantized modules retain their existing precision behavior.
- Baseline: `../jailguard_official_repro/pl_all.jsonl` (Accuracy 75.90%, Precision 67.32%, Recall 77.25%).
- Paper: Accuracy 82.90%, Precision 74.00%, Recall 88.25%.
- Runtime smoke: one preselected item per class (serials 0, 100, 200, 400). Validate real int8 modules, generation, matching input/protocol metadata and finite detector outputs; do not select by detection accuracy.
- Full run: two GPU shards, 500 inputs per shard, each input generating eight responses. Smoke records are separate and not included in the final 1,000-item result.
- Supervisor: detached OS session, stdin disconnected and output redirected to files; continues when VSCode closes while the server remains running. Automatically validates 1,000 unique records, aggregates and compares with the baseline and paper.
- Results: `status.json`, `logs/`, per-worker `*.runtime.json`, `summary_pl.json`, and `comparison.md`/`comparison.json`.

The upstream code queries variants serially and uses a CPU vision path in low-resource mode. Those differences remain intentionally fixed at the historical reproduction settings for this precision experiment. A historical baseline does not establish causality if software or hardware changed; actual versions and quantization details are recorded. Published paper scores are references, not acceptance criteria.
