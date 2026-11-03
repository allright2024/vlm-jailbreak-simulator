# JailGuard: 8-bit comparison

Same official 1,000 images; PL, eight variants, threshold 0.025.

| Setting | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| Paper Table 4 PL | 82.90% | 74.00% | 88.25% |
| Historical FP16 | 75.90% | 67.32% | 77.25% |
| 8-bit LLM (this run) | 77.00% | 69.23% | 76.50% |

Historical FP16 baseline; software versions and GPU kernels may differ. Batching and GPU vision encoder still differ from the original upstream runtime.

The smoke check was a runtime check; no samples were selected by detection accuracy.
