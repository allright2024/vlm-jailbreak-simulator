"""Run the official CS-DJ scripts and then the official released HADES data."""
import collections
import difflib
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
SOURCE = RUN / "source"
REF = ROOT.parent / "reference_repos"
CATEGORIES = ["Animal", "Financial", "Privacy", "Self-Harm", "Violence"]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(name, value):
    p = RUN / name
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def patch_source():
    orig = (REF / "CS-DJ/script/main.py").read_text()
    source = orig.replace("from config import CS_DJ_parser", "from config import CS_DJ_parser\nfrom run_support import is_done, seed_case, record_attempt, record_sample, record_failure")
    source = source.replace("        for item in tqdm(data):\n", "        for item in tqdm(data):\n            if is_done(harmful_type, i):\n                i += 1\n                continue\n            seed_case(harmful_type, i)\n")
    source = source.replace("            while True:\n", "            tip = 0\n            while True:\n")
    source = source.replace("                # print(len(sub_question_list), sub_question_list)", "                record_attempt(harmful_type, i, tip + 1, response, sub_question_list)")
    source = source.replace("                tip += 1\n                if tip > 5 and len(sub_question_list) == 3:\n                    break\n", "                tip += 1\n                if tip >= 5:\n                    break\n")
    source = source.replace("            if not os.path.exists('./combine_image_data/'):", "            if len(sub_question_list) != 3 or 'First sub-question' in response:\n                record_failure(harmful_type, i, item['instruction'], input_text, tip)\n                i += 1\n                continue\n\n            if not os.path.exists('./combine_image_data/'):")
    start = source.index("            base64_image = encode_image(IMAGE_PATH)")
    end = source.index("            final_res.append", start)
    source = source[:start] + "            record_sample(harmful_type, i, item['instruction'], input_text, IMAGE_PATH, sub_question_list, tip + 1)\n            output_text = None  # Target inference is a separate native LLaVA-NeXT stage.\n\n" + source[end:]
    (SOURCE / "script/main.py").write_text(source)
    (RUN / "official_source.patch").write_text("".join(difflib.unified_diff(orig.splitlines(True), source.splitlines(True), fromfile="official/script/main.py", tofile="working/script/main.py")))
    assert source != orig and "openai.chat.completions.create(" not in source
    compile(source, str(SOURCE / "script/main.py"), "exec")


def prepare():
    patch_source()
    pool = RUN / "image_pool"
    pool.mkdir(exist_ok=True)
    old = ROOT / "data/CS-DJ/llava_images"
    paths = sorted(p for p in old.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"])
    assert len(paths) == 10000
    for p in paths:
        q = pool / p.name
        if not q.exists():
            q.symlink_to(p)
    # Keep the exact filesystem ordering that upstream random.shuffle receives.
    save("image_pool_order.json", os.listdir(pool))
    save("execution_plan.json", {
        "cs_dj_source": str(REF / "CS-DJ"), "worktree": str(SOURCE),
        "cs_dj_commit": "4577b05fe93e86030ba2e4c3339e3aa59cdbb5ff",
        "llava_next_commit": subprocess.check_output(["git", "-C", str(REF / "LLaVA-NeXT"), "rev-parse", "HEAD"], text=True).strip(),
        "hades_commit": "a2670b76e42448831c86d44dec02a8fa3a12d468",
        "cs_dj_samples": 750, "cs_dj_target_responses": 750,
        "hades_samples": 750, "hades_target_responses": 750,
        "hades_variant": "Paper Table 2 LLaVA-1.5-7B +Adv image; official final composites; official Beaver-Dam evaluation. Separate paper reproduction, not a CS-DJ comparison.",
        "hades_target": "liuhaotian/llava-v1.5-7b; original model code bundled in HADES",
        "hades_data_preparation": "All 6000 published images prepared by existing audited preparation script; select 750 final adv composites for inference",
        "cs_dj_generation": {"target": "lmms-lab/llava-onevision-qwen2-7b-ov-chat", "do_sample": False, "max_new_tokens": 1000, "batch_size": 1},
        "sample_generation": "Run upstream image_embeding.py and Visual-Enhanced_Distraction.py unchanged; run minimally patched upstream main.py",
        "image_library": "Existing verified 10000-image CC3M subset reused; embeddings and retrieval are recomputed by the original scripts",
        "deviations": ["Initialize undefined tip counter and enforce paper section 8.1 maximum five decomposition attempts; record construction failures and keep them in the 750-case denominator (failure accounting is our explicit interpretation)", "Fixed per-case seed 2025-derived for resumability; paper seed unpublished", "Skip API calls and run native OneVision in a separate phase", "Checkpoint and skip completed inputs without resampling", "Native runtime uses isolated transformers 4.45.2, current GPU-compatible torch and SDPA MATH; paper environment not established"],
        "no_paid_api": True, "no_gpt2_ppl": True,
        "source_patch": "official_source.patch",
    })
    print("PREPARED_OFFICIAL_EXECUTION", flush=True)


def download():
    from huggingface_hub import snapshot_download
    from ranged_download import fetch_weight
    for meta_name in ["original_model_metadata.json", "vision_model_metadata.json"]:
        meta = json.loads((RUN / meta_name).read_text())
        path = Path(snapshot_download(meta["id"], revision=meta["sha"],
                    allow_patterns=["*.json", "*.txt", "README.md"], max_workers=4))
        checked = []
        for f in meta["files"]:
            if f.get("lfs") and f["name"].endswith(".safetensors"):
                fetch_weight(meta, f, path)
                p = path / f["name"]
                assert p.stat().st_size == f["lfs"]["size"]
                assert sha(p) == f["lfs"]["sha256"]
                checked.append(f["name"])
        save(meta_name.replace("metadata", "validation"), {"all_weights_verified": True, "files": checked})
        if meta_name.startswith("original"):
            save("original_model_snapshot.json", {"repo": meta["id"], "revision": meta["sha"], "path": str(path)})
        else:
            refs = path.parents[1] / "refs"
            refs.mkdir(exist_ok=True)
            (refs / "main").write_text(meta["sha"])


def official(stage):
    import PIL
    import torch
    assert PIL.__version__ == "9.5.0"
    assert json.loads((RUN / "native_smoke.json").read_text())["passed"]
    torch.manual_seed(2025)
    sys.path.insert(0, str(SOURCE / "script"))
    os.chdir(SOURCE)
    script = {"embedding": "image_embeding.py", "retrieval": "Visual-Enhanced_Distraction.py", "samples": "main.py"}[stage]
    # Only the sample stage is patched. Other source files must be identical.
    for name in ["image_embeding.py", "Visual-Enhanced_Distraction.py", "config.py"]:
        assert sha(SOURCE / "script" / name) == sha(REF / "CS-DJ/script" / name)
    sys.argv = [script, "--src_dir", str(RUN / "image_pool"), "--num_images", "10000", "--seed", "2025",
                "--save_embeding_path", str(RUN / "image_embedding"), "--save_map_path", str(RUN / "image_map"),
                "--jailbreak_folder_path", str(SOURCE / "instructions"), "--distraction_image_save_path", str(RUN / "images"),
                "--jailbreak_response_save_path", str(RUN / "sample_metadata")]
    target = RUN / ("image_embedding/map_seed_2025_num_10000.json" if stage == "embedding" else "image_map/distraction_image_map_seed_2025_num_10000.json")
    if stage != "samples" and target.exists():
        assert len(json.loads(target.read_text())) == (10000 if stage == "embedding" else 750)
        print("REUSE_COMPLETED_OFFICIAL_STAGE", stage, flush=True)
        return
    runpy.run_path(str(SOURCE / "script" / script), run_name="__main__")
    if stage == "samples":
        rows = [json.loads(x) for x in (RUN / "test_cases.jsonl").read_text().splitlines()]
        assert len(rows) == len({x["custom_id"] for x in rows}) == 750
        assert collections.Counter(x["category"] for x in rows) == {c:150 for c in CATEGORIES}
        for x in rows:
            assert 1 <= x["decomposition_attempts"] <= 5
            if x.get("construction_failed", False):
                assert x["image"] is None and x["decomposition_attempts"] == 5
            else:
                assert sha(x["image"]) == x["image_sha256"]
        failures = sum(x.get("construction_failed", False) for x in rows)
        save("samples_validation.json", {"n": 750, "all_unique": True, "all_images_verified": True,
            "construction_failed": failures, "images_generated": 750 - failures,
            "decomposition_over_five_attempts": sum(x["decomposition_attempts"] > 5 for x in rows),
            "test_cases_sha256": sha(RUN / "test_cases.jsonl")})


if __name__ == "__main__":
    action = sys.argv[1]
    if action in ["embedding", "retrieval", "samples"]:
        official(action)
    else:
        {"prepare": prepare, "patch_source": patch_source, "download": download}[action]()
