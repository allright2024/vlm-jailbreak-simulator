"""Resume official HADES parquet downloads, then use the audited data preparer."""
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def main():
    import ranged_download
    spec = importlib.util.spec_from_file_location("audited_hades_data", ROOT / "attack_scripts/hades_reproduction_preparation.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    prep.source_checks()
    # The queue independently seals the current CS-DJ evaluation before this call.
    # Preserve the old preparer's historical dependency checks and data semantics.
    prep.verify_cs()
    ranged_download.RUN = prep.RUN
    cache = prep.RUN / "download_cache"
    snapshot = cache / "snapshots" / prep.DATA_REVISION
    (cache / "blobs").mkdir(parents=True, exist_ok=True)
    (snapshot / "data").mkdir(parents=True, exist_ok=True)
    target_dir = prep.RUN / "data/parquet"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, size, digest in prep.PARQUETS:
        target = target_dir / name
        if target.exists():
            assert target.stat().st_size == size and prep.sha(target) == digest
            continue
        entry = {"name": "data/" + name, "lfs": {"size": size, "sha256": digest}}
        ranged_download.fetch_weight({"id": "datasets/Monosail/HADES", "sha": prep.DATA_REVISION}, entry, snapshot)
        target.symlink_to(snapshot / entry["name"])
    prep.prepare()


if __name__ == "__main__":
    main()
