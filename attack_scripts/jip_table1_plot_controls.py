"""Plot measured benign controls; run in an environment with matplotlib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.out / "control/run_manifest.json").read_text())
    if manifest["experiment_scope"] != "benign_control":
        raise ValueError("this figure is restricted to the benign control experiment")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = ("textual", "ocr", "visual", "combined")
    colors = {"textual": "#6f42c1", "ocr": "#087e8b", "visual": "#ce6a26", "combined": "#2356a8"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for key, artifact in manifest["artifacts"].items():
        kind, concept = key.split(":")
        history = json.loads(Path(artifact["files"]["history"]["path"]).read_text())
        axes[0].plot([row["updates"] for row in history], [row["loss"] for row in history],
                     color=colors[kind], linestyle="-" if concept == "red_square" else "--",
                     linewidth=1.2, label=f"{kind}: {concept.replace('_', ' ')}")
    axes[0].set(xlabel="Adam updates", ylabel="Projected CLIP embedding L2", title="Optimization trajectories")
    axes[0].legend(fontsize=7, ncol=2)
    variants = (("best_tensor_l2", "Float tensor", "#306a9c"),
                ("png_l2", "PNG", "#31947b"), ("jpeg_l2", "JPEG", "#cf673d"))
    for variant, (field, label, color) in enumerate(variants):
        scores = [mean(a[field] for a in manifest["artifacts"].values() if a["trigger_type"] == kind)
                  for kind in kinds]
        axes[1].bar([index + (variant - 1) * 0.24 for index in range(4)], scores,
                    width=0.23, label=label, color=color)
    axes[1].set(xticks=range(4), xticklabels=kinds, ylabel="Mean L2 over two control images",
                title="Serialization of the same optimized pixels")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_yscale("log")
        axis.axhline(0.3, color="#b64050", linestyle=":", linewidth=1)
        axis.grid(axis="y", alpha=0.15)
    figure.suptitle("Benign geometry controls — not Table 1 attack ASR", fontsize=13)
    cap = manifest["settings"]["max_steps"]
    figure.text(0.5, 0.01, f"Dotted line: L2 threshold 0.3. Optimization cap: {cap} updates; the original Table 1 data is unavailable.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    for extension in ("png", "svg"):
        figure.savefig(args.out / f"control_optimization.{extension}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
