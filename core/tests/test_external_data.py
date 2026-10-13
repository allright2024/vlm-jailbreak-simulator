"""External data-root behavior without requiring the benchmark bundle."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from core import REPO_ROOT


def main() -> None:
    with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as run_dir:
        probe = (
            "import os\n"
            "from core import DATA_ROOT, resolve_path\n"
            "assert DATA_ROOT == os.path.abspath(os.environ['VLM_JAILBREAK_DATA_ROOT'])\n"
            "assert resolve_path('data/FORCE/clean.jpeg') == "
            "os.path.join(DATA_ROOT, 'FORCE', 'clean.jpeg')\n"
        )
        env = {
            **os.environ,
            "VLM_JAILBREAK_DATA_ROOT": data_root,
            "PYTHONNOUSERSITE": "1",
        }
        checked = subprocess.run(
            [sys.executable, "-c", probe], cwd=REPO_ROOT, env=env,
            capture_output=True, text=True,
        )
        assert checked.returncode == 0, checked.stderr

        # Stage 1 must fail before writing partial outputs and explain how to
        # configure the separately distributed data.
        failed = subprocess.run(
            [sys.executable, "run_generate_test_cases.py", "--attack", "FigStep",
             "--dataset", "SafeBench", "--save_dir", run_dir, "--limit", "1"],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        )
        assert failed.returncode != 0
        combined = failed.stdout + failed.stderr
        assert "VLM_JAILBREAK_DATA_ROOT" in combined
        assert "SafeBench/safebench.csv" in combined
        assert not os.path.exists(os.path.join(run_dir, "test_cases.jsonl"))
        assert not os.path.exists(os.path.join(run_dir, "run_manifest.json"))

    print("external data root OK: data/ remapping + preflight hard-fail")


if __name__ == "__main__":
    main()
