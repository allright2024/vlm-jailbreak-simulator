# VLMShield official-code reproduction

Official code/weights/evaluation sets: https://github.com/pgqihere/VLMShield @ 344ca14; CLIP `openai/clip-vit-large-patch14` (snapshot 32bd642); released classifier `vlmshield_classifier.pt` (sha256 b36702877791). AdvBench-M adversarial images from the imgJP repository (`dataset/advimage`, 8 x 20 jpg), which the VLMShield JSON references but does not ship.

VLMShield is a detector (CLIP ViT-L/14 features -> released 3-layer MLP); ASR = malicious prompts predicted safe, ACC = benign prompts predicted safe. No target VLM is run.

| Dataset | n | images | metric | paper (arXiv:2604.06502) | this run | diff |
| --- | --- | --- | --- | --- | --- | --- |
| Advbench_M | 242 | 160 | ASR % (Table 2 (RQ1)) | 0.41 | 0.41 | +0.00 |
| VLSafe | 1110 | 1110 | ASR % (Table 2 (RQ1)) | 1.62 | 1.71 | +0.09 |
| MM-Vet | 218 | 218 | ACC % (Table 3 (RQ2)) | 96.33 | 96.79 | +0.46 |

Processing time: MM-Vet 45 ms/sample, VLSafe 30 ms/sample, Advbench_M 20 ms/sample (paper Table 4: 0.34 s detection incl. model load amortisation not comparable).

## Deviations
- Environment: transformers 4.57.1 / torch 2.9.1 / B200 MIG instead of the released conda env (CUDA 11.7, Python 3.8); the official script is run unmodified.
- AdvBench-M: 82 of 242 entries have no image in the released JSON and are scored text-only by the official code (zero image vector); 160 entries use the imgJP adversarial images.
- The paper's remaining Table 2 columns (JailbreakV_28K, FigStep, HADES, MM-SafetyBench) and Table 3 columns (GPT4V-Caption, CC3M, MMBench) are not shipped in the repository and are not run.
