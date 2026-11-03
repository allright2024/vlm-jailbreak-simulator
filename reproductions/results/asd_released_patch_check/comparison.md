# Released AsD patch: paired 64-sample check

| Inference path | Unsafe / 64 | ASR |
|---|---:|---:|
| official_serial | 13 / 64 | 20.31% |
| adapter_batch8 | 13 / 64 | 20.31% |

Same released patch, patch positions, prompt tokens and processed image tensors.

Diagnostic subset. Direct official AsD.main call vs existing batch-eight adapter; not a complete Table 1 reproduction.

GPT-4o judge differs from the paper's unspecified GPT-4 snapshot.
Both arms use the current server and transformers 4.37 runtime, rather than restoring the paper's full historical software environment.
Fixed patch positions isolate batching; they are not fresh random positions for the 64-sample subset.
