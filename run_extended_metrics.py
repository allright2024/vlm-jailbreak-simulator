"""Aggregate the simulator's extended attack/defense/detector metrics.

The report contains aggregates only: prompts and model responses are never
copied into the output. Optional metrics remain explicitly unavailable until
their required raw inputs are supplied.

Example:
    python run_extended_metrics.py \
      --results runs/example/results_llava_v1_5__defense-CIDER.jsonl \
      --baseline-results runs/example/results_llava_v1_5.jsonl \
      --behaviors runs/example/behaviors.jsonl \
      --output runs/example/extended_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from core import Behavior, Completion, read_jsonl
from core.metrics import (
    attack_budget_curve,
    attack_success_metrics,
    binary_detection_metrics,
    continuous_detection_metrics,
    efficiency_metrics,
    efficiency_overhead_metrics,
    evaluation_coverage_metrics,
    grouped_attack_metrics,
    judge_agreement_metrics,
    paired_binary_utility_metrics,
    paired_defense_metrics,
    pass_at_k_curve,
    response_quality_metrics,
    seed_stability_metrics,
    success_efficiency_metrics,
    transfer_attack_metrics,
)
from core.multimodal_metrics import (
    aggregate_image_pair_metrics,
    image_perturbation_metrics,
    modality_ablation_metrics,
    transformation_robustness_metrics,
    variant_robustness_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute extended, model-free jailbreak benchmark metrics."
    )
    parser.add_argument("--results", required=True, help="Judged attack/defense Completion JSONL.")
    parser.add_argument("--behaviors", default=None, help="Optional Behavior JSONL for requested IDs/groups.")
    parser.add_argument("--baseline-results", default=None, help="Paired undefended Completion JSONL.")
    parser.add_argument("--benign-results", default=None, help="Defended benign Completion JSONL for FPR.")
    parser.add_argument("--benign-baseline-results", default=None, help="Baseline benign JSONL for utility retention.")
    parser.add_argument(
        "--benign-label-means-utility", action="store_true",
        help="Treat benign Completion.label as task correctness; otherwise require meta.utility_correct.",
    )
    parser.add_argument("--source-results", default=None, help="Source-model results for transfer ASR.")
    parser.add_argument("--second-judge-results", default=None, help="Same cases judged by a second judge.")
    parser.add_argument(
        "--seed-results", action="append", default=[],
        help="Additional judged seed run; repeat for stability mean/std/CI.",
    )
    parser.add_argument("--text-only-results", default=None, help="Text-only ablation results.")
    parser.add_argument("--image-only-results", default=None, help="Image-only ablation results.")
    parser.add_argument(
        "--variant", action="append", default=[], metavar="NAME=PATH",
        help="Aligned variant results; repeat for variant worst-case/consistency.",
    )
    parser.add_argument(
        "--transformed", action="append", default=[], metavar="NAME=PATH",
        help="JPEG/resize/other transformed results; repeat as needed.",
    )
    parser.add_argument(
        "--image-pairs", default=None,
        help="JSON/JSONL rows with id, reference, candidate image paths.",
    )
    parser.add_argument("--lpips", action="store_true", help="Attempt optional LPIPS for image pairs.")
    parser.add_argument("--clip", action="store_true", help="Attempt optional CLIP image similarity.")
    parser.add_argument(
        "--allow-clip-download", action="store_true",
        help="Allow transformers to download CLIP weights; default uses local cache only.",
    )
    parser.add_argument(
        "--query-budgets", default="1,2,4,8,16,32",
        help="Comma-separated budgets for ASR@model-query budget.",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Output JSON (default beside --results).")
    return parser.parse_args()


def _named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError(f"expected non-empty NAME=PATH, got {value!r}")
    return name, path


def _behavior_outcomes(rows: Iterable[Completion]) -> dict[str, Optional[int]]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        if row.label is not None:
            grouped.setdefault(row.behavior_id, []).append(int(row.label))
    return {
        behavior_id: int(any(labels)) if labels else None
        for behavior_id, labels in grouped.items()
    }


def _aligned_behavior_labels(*runs: Sequence[Completion]) -> tuple[list[str], list[list[int]]]:
    outcomes = [_behavior_outcomes(run) for run in runs]
    common = sorted(set.intersection(*(set(values) for values in outcomes))) if outcomes else []
    common = [
        behavior_id for behavior_id in common
        if all(values.get(behavior_id) is not None for values in outcomes)
    ]
    return common, [
        [int(values[behavior_id]) for behavior_id in common]
        for values in outcomes
    ]


def _aligned_attempt_labels(
    left: Sequence[Completion], right: Sequence[Completion]
) -> tuple[list[Optional[int]], list[Optional[int]]]:
    right_by_id = {row.test_case_id: row for row in right}
    paired = [
        (row.label, right_by_id[row.test_case_id].label)
        for row in left if row.test_case_id in right_by_id
    ]
    return [value[0] for value in paired], [value[1] for value in paired]


def _detector_observation(row: Completion) -> tuple[Optional[int], Optional[float], Optional[str]]:
    meta = row.meta
    decision = meta.get("detected_adversarial")
    source = "detected_adversarial"
    if not isinstance(decision, bool):
        decision = meta.get("unsafe_detected")
        source = "unsafe_detected"
    if not isinstance(decision, bool) and meta.get("defense") and row.blocked:
        decision = True
        source = "blocked_with_defense_metadata"
    prediction = int(decision) if isinstance(decision, bool) else None

    score: Optional[float] = None
    score_source: Optional[str] = None
    raw = meta.get("detection_score")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        score, score_source = float(raw), "detection_score"
    elif isinstance(meta.get("max_divergence"), (int, float)):
        score, score_source = float(meta["max_divergence"]), "max_divergence"
    else:
        deltas = meta.get("cosine_deltas")
        if isinstance(deltas, list) and len(deltas) > 1 and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in deltas[1:]
        ):
            # CIDER detects when the minimum delta is below a negative
            # threshold, so negate it to honor higher-is-more-malicious.
            score = -min(float(value) for value in deltas[1:])
            score_source = "negative_min_cider_cosine_delta"
    if score is not None and not math.isfinite(score):
        score, score_source = None, None
    return prediction, score, source if prediction is not None else score_source


def _detection_report(
    malicious: Sequence[Completion], benign: Sequence[Completion]
) -> dict[str, Any]:
    decisions: list[int] = []
    truths: list[int] = []
    decision_sources: dict[str, int] = {}
    scored_truths: list[int] = []
    scores: list[float] = []
    score_sources: dict[str, int] = {}
    for truth, rows in ((1, malicious), (0, benign)):
        for row in rows:
            decision, score, source = _detector_observation(row)
            if decision is not None:
                truths.append(truth)
                decisions.append(decision)
                key = source or "unknown"
                decision_sources[key] = decision_sources.get(key, 0) + 1
            if score is not None:
                scored_truths.append(truth)
                scores.append(score)
                # Recover an unambiguous score source independently of decision.
                if isinstance(row.meta.get("detection_score"), (int, float)):
                    score_key = "detection_score"
                elif isinstance(row.meta.get("max_divergence"), (int, float)):
                    score_key = "max_divergence"
                else:
                    score_key = "negative_min_cider_cosine_delta"
                score_sources[score_key] = score_sources.get(score_key, 0) + 1
    binary = binary_detection_metrics(truths, decisions) if decisions else None
    comparable_scores = bool(scores) and len(score_sources) == 1
    continuous = (
        continuous_detection_metrics(scored_truths, scores)
        if comparable_scores else None
    )
    return {
        "available": bool(decisions),
        "binary": binary,
        "continuous_available": comparable_scores,
        "continuous": continuous,
        "decision_sources": decision_sources,
        "score_sources": score_sources,
        "n_malicious_rows_supplied": len(malicious),
        "n_benign_rows_supplied": len(benign),
        "fpr_available": bool(benign) and any(value == 0 for value in truths),
        "unavailable_reason": None if decisions else "no_detector_decision_metadata",
        "continuous_unavailable_reason": (
            None if comparable_scores
            else "no_continuous_score_metadata" if not scores
            else "mixed_score_types_are_not_comparable"
        ),
    }


def _metadata_annotations(rows: Sequence[Completion]) -> dict[str, Any]:
    refusal: list[int] = []
    partial: list[int] = []
    severity: list[float] = []
    for row in rows:
        for key in ("refused", "is_refusal"):
            if isinstance(row.meta.get(key), bool):
                refusal.append(int(row.meta[key]))
                break
        for key in ("partial_compliance", "is_partial_compliance"):
            if isinstance(row.meta.get(key), bool):
                partial.append(int(row.meta[key]))
                break
        for key in ("severity_score", "harm_severity"):
            value = row.meta.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                severity.append(float(value))
                break
    return response_quality_metrics(
        refused=refusal or None,
        partial_compliance=partial or None,
        severity_scores=severity or None,
    )


def _utility_annotations(
    rows: Sequence[Completion], allow_label: bool
) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        value = row.meta.get("utility_correct", row.meta.get("task_correct"))
        if isinstance(value, bool) or value in (0, 1):
            result[row.test_case_id] = int(value)
        elif allow_label and row.label in (0, 1):
            result[row.test_case_id] = int(row.label)
    return result


def _utility_report(
    baseline: Sequence[Completion], defended: Sequence[Completion], allow_label: bool
) -> dict[str, Any]:
    left, right = _utility_annotations(baseline, allow_label), _utility_annotations(defended, allow_label)
    common = sorted(set(left) & set(right))
    if not common:
        return {
            "available": False,
            "metrics": None,
            "unavailable_reason": "no_paired_utility_correct_annotations",
        }
    return {
        "available": True,
        "metrics": paired_binary_utility_metrics(
            [left[key] for key in common], [right[key] for key in common]
        ),
        "unavailable_reason": None,
    }


def _read_image_pairs(path: str) -> list[dict[str, Any]]:
    pair_path = Path(path)
    if pair_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in pair_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(pair_path.read_text(encoding="utf-8"))
        rows = payload.get("pairs", []) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("--image-pairs must contain a JSON list or {'pairs': [...]}")
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not row.get("reference") or not row.get("candidate"):
            raise ValueError(f"image pair {index} needs reference and candidate")
        def resolve(value: str) -> str:
            candidate = Path(value)
            return str(candidate if candidate.is_absolute() else pair_path.parent / candidate)
        normalized.append({
            "id": str(row.get("id", index)),
            "reference": resolve(str(row["reference"])),
            "candidate": resolve(str(row["candidate"])),
        })
    return normalized


def _safe_json(value: Any) -> Any:
    """Replace non-finite floats so output is strict RFC-compatible JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    results = read_jsonl(args.results, Completion)
    behaviors = read_jsonl(args.behaviors, Behavior) if args.behaviors else []
    requested_ids = [behavior.behavior_id for behavior in behaviors] or sorted(
        {row.behavior_id for row in results}
    )
    group_map = {behavior.behavior_id: behavior.category for behavior in behaviors}
    query_budgets = [
        float(value.strip()) for value in args.query_budgets.split(",") if value.strip()
    ]

    attack = attack_success_metrics(results, requested_ids, args.confidence)
    report: dict[str, Any] = {
        "schema_version": 1,
        "inputs": {
            "results": os.path.abspath(args.results),
            "behaviors": os.path.abspath(args.behaviors) if args.behaviors else None,
        },
        "conventions": {
            "attack_unit": "unique_behavior",
            "attack_aggregation": "pass@k_any_success",
            "detection_positive": "malicious_or_jailbreak",
            "predicted_positive": "detected_or_blocked",
            "missing_metric_value": None,
        },
        "attack": {
            "headline": attack,
            "pass_at_k": pass_at_k_curve(results, requested_ids, confidence=args.confidence),
            "asr_at_query_budget": attack_budget_curve(
                results, query_budgets, requested_ids,
                cost_key="model_query_count", confidence=args.confidence,
            ),
            "groups": grouped_attack_metrics(
                results, group_map, requested_ids, args.confidence
            ) if behaviors else {
                "available": False,
                "unavailable_reason": "behaviors_file_not_supplied",
            },
            "transfer": None,
            "response_quality": _metadata_annotations(results),
            "time_and_queries_to_success": success_efficiency_metrics(results, requested_ids),
        },
        "reliability": {
            "coverage": evaluation_coverage_metrics(results, requested_ids),
            "judge_agreement": None,
            "seed_stability": None,
        },
        "efficiency": efficiency_metrics(results),
        "defense": None,
        "detection": None,
        "benign_utility": None,
        "multimodal": {
            "image_perturbation": None,
            "modality_ablation": None,
            "variants": None,
            "transformations": {},
        },
    }

    baseline: list[Completion] = []
    if args.baseline_results:
        baseline = read_jsonl(args.baseline_results, Completion)
        report["inputs"]["baseline_results"] = os.path.abspath(args.baseline_results)
        report["defense"] = {
            "available": True,
            "paired": paired_defense_metrics(baseline, results, requested_ids),
            "efficiency_overhead": efficiency_overhead_metrics(baseline, results),
        }
    else:
        report["defense"] = {
            "available": False,
            "unavailable_reason": "baseline_results_not_supplied",
        }

    benign = read_jsonl(args.benign_results, Completion) if args.benign_results else []
    report["detection"] = _detection_report(results, benign)
    if args.benign_results:
        report["inputs"]["benign_results"] = os.path.abspath(args.benign_results)

    if args.benign_baseline_results and benign:
        benign_baseline = read_jsonl(args.benign_baseline_results, Completion)
        report["benign_utility"] = _utility_report(
            benign_baseline, benign, args.benign_label_means_utility
        )
    else:
        report["benign_utility"] = {
            "available": False,
            "metrics": None,
            "unavailable_reason": (
                "requires_benign_results_and_benign_baseline_results"
            ),
        }

    if args.source_results:
        source = read_jsonl(args.source_results, Completion)
        report["attack"]["transfer"] = {
            "available": True,
            "metrics": transfer_attack_metrics(source, results, requested_ids),
        }
    else:
        report["attack"]["transfer"] = {
            "available": False,
            "unavailable_reason": "source_results_not_supplied",
        }

    if args.second_judge_results:
        second = read_jsonl(args.second_judge_results, Completion)
        left, right = _aligned_attempt_labels(results, second)
        report["reliability"]["judge_agreement"] = {
            "available": bool(left),
            "metrics": judge_agreement_metrics(left, right) if left else None,
            "unavailable_reason": None if left else "no_matching_test_case_ids",
        }
    else:
        report["reliability"]["judge_agreement"] = {
            "available": False,
            "unavailable_reason": "second_judge_results_not_supplied",
        }

    if args.seed_results:
        seed_runs = [results] + [read_jsonl(path, Completion) for path in args.seed_results]
        values = [
            attack_success_metrics(run, requested_ids, args.confidence)["asr"]
            for run in seed_runs
        ]
        finite_values = [float(value) for value in values if value is not None]
        report["reliability"]["seed_stability"] = {
            "available": len(finite_values) >= 2,
            "metrics": seed_stability_metrics(
                finite_values,
                confidence=args.confidence,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            ),
            "unavailable_reason": (
                None if len(finite_values) >= 2 else "fewer_than_two_scored_seed_runs"
            ),
        }
    else:
        report["reliability"]["seed_stability"] = {
            "available": False,
            "unavailable_reason": "additional_seed_results_not_supplied",
        }

    if args.text_only_results and args.image_only_results:
        text_rows = read_jsonl(args.text_only_results, Completion)
        image_rows = read_jsonl(args.image_only_results, Completion)
        common, labels = _aligned_behavior_labels(text_rows, image_rows, results)
        report["multimodal"]["modality_ablation"] = {
            "available": bool(common),
            "n_aligned_behaviors": len(common),
            "metrics": modality_ablation_metrics(*labels) if common else None,
            "unavailable_reason": None if common else "no_jointly_scored_behavior_ids",
        }
    else:
        report["multimodal"]["modality_ablation"] = {
            "available": False,
            "unavailable_reason": "requires_text_only_and_image_only_results",
        }

    if args.variant:
        named = [_named_path(value) for value in args.variant]
        variant_runs = {"main": results, **{
            name: read_jsonl(path, Completion) for name, path in named
        }}
        common, labels = _aligned_behavior_labels(*variant_runs.values())
        report["multimodal"]["variants"] = {
            "available": bool(common),
            "n_aligned_behaviors": len(common),
            "metrics": variant_robustness_metrics(dict(zip(variant_runs, labels))) if common else None,
            "unavailable_reason": None if common else "no_jointly_scored_behavior_ids",
        }
    else:
        report["multimodal"]["variants"] = {
            "available": False,
            "unavailable_reason": "variant_results_not_supplied",
        }

    for value in args.transformed:
        name, path = _named_path(value)
        transformed = read_jsonl(path, Completion)
        common, labels = _aligned_behavior_labels(results, transformed)
        report["multimodal"]["transformations"][name] = {
            "available": bool(common),
            "n_aligned_behaviors": len(common),
            "metrics": transformation_robustness_metrics(
                labels[0], labels[1], transformation=name
            ) if common else None,
            "unavailable_reason": None if common else "no_jointly_scored_behavior_ids",
        }

    if args.image_pairs:
        pairs = _read_image_pairs(args.image_pairs)
        pair_results = []
        for pair in pairs:
            metrics = image_perturbation_metrics(
                pair["reference"], pair["candidate"],
                compute_lpips=args.lpips,
                compute_clip=args.clip,
                clip_local_files_only=not args.allow_clip_download,
            )
            pair_results.append({"id": pair["id"], **metrics})
        report["multimodal"]["image_perturbation"] = {
            "available": bool(pair_results),
            "aggregate": aggregate_image_pair_metrics(pair_results),
            "pairs": pair_results,
            "unavailable_reason": None if pair_results else "empty_image_pair_file",
        }
    else:
        report["multimodal"]["image_perturbation"] = {
            "available": False,
            "unavailable_reason": "image_pairs_not_supplied",
        }

    output = args.output or str(Path(args.results).with_name("extended_metrics.json"))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    safe_report = _safe_json(report)
    Path(output).write_text(
        json.dumps(safe_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    concise = {
        "asr": safe_report["attack"]["headline"]["asr"],
        "asr_ci": safe_report["attack"]["headline"]["asr_ci"],
        "coverage": safe_report["reliability"]["coverage"]["behavior_coverage"],
        "detection_available": safe_report["detection"]["available"],
        "defense_comparison_available": safe_report["defense"]["available"],
        "output": os.path.abspath(output),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
