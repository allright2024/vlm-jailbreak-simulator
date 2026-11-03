"""Check category/label balance after grouping duplicate questions and images."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from attack_scripts.jailbound_paper_protocol import read_rows, sha
from attack_scripts.mmsafetybench_baseline_audit import write


def duplicate_groups(cases):
    parent = {r["id"]: r["id"] for r in cases}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    seen = {}
    for row in cases:
        keys = [("question", row["question_sha256"])]
        keys += [("image", value) for value in row["all_image_sha256"].values()]
        for key in keys:
            if key in seen:
                first, second = find(row["id"]), find(seen[key])
                parent[first] = second
            else:
                seen[key] = row["id"]
    members = defaultdict(list)
    for row in cases:
        members[find(row["id"])].append(row["id"])
    groups = {min(ids): sorted(ids) for ids in members.values()}
    case_group = {i: group for group, ids in groups.items() for i in ids}
    return groups, case_group


def assign(cases, records, case_group, per_label, metric):
    choices = defaultdict(lambda: defaultdict(list))
    for row in cases:
        stratum = (row["category"], 1 - records[row["id"]][metric])
        choices[stratum][case_group[row["id"]]].append(row["id"])
    strata = [(category, label) for category in sorted({r["category"] for r in cases}) for label in (0, 1)]
    adjacency = {}
    for stratum in strata:
        groups = sorted(choices[stratum])
        random.Random(f"0:{stratum}:{metric}").shuffle(groups)
        for slot in range(per_label):
            adjacency[(*stratum, slot)] = groups
    owner = {}

    def augment(slot, seen):
        for group in adjacency[slot]:
            if group in seen:
                continue
            seen.add(group)
            if group not in owner or augment(owner[group], seen):
                owner[group] = slot
                return True
        return False

    for slot in sorted(adjacency, key=lambda k: len(adjacency[k])):
        augment(slot, set())
    selected = []
    for group, slot in owner.items():
        category, label, _ = slot
        selected.append({"id": min(choices[(category, label)][group]), "category": category,
                         "label": label, "duplicate_group": group})
    selected.sort(key=lambda r: (r["category"], r["label"], r["id"]))
    counts = Counter((r["category"], r["label"]) for r in selected)
    missing = [{"category": category, "label": label, "shortfall": per_label - counts[(category, label)]}
               for category, label in strata if counts[(category, label)] < per_label]
    return {"per_category_per_label": per_label, "target_n": len(strata) * per_label,
            "matched_n": len(selected), "feasible": not missing,
            "missing_slots": missing, "selected": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads((args.run / "source_cases.json").read_text())
    records = {r["id"]: r for r in read_rows(args.run / "baseline.jsonl")}
    verification = json.loads((args.run / "verification.json").read_text())
    if not verification["passed"] or sha(args.run / "baseline.jsonl") != verification["baseline_sha256"]:
        raise ValueError("Baseline audit must be complete and verified")
    groups, case_group = duplicate_groups(cases)
    results = {}
    for metric in ("paper_success", "casefold_success"):
        requested = assign(cases, records, case_group, 5, metric)
        counts = Counter((r["case"]["category"], 1 - r[metric]) for r in records.values())
        strata = [(c, label) for c in {r["category"] for r in cases} for label in (0, 1)]
        low, high = 0, min(counts[s] for s in strata)
        while low < high:
            middle = (low + high + 1) // 2
            if assign(cases, records, case_group, middle, metric)["feasible"]:
                low = middle
            else:
                high = middle - 1
        chosen = requested["selected"]
        chosen_ids = {r["id"] for r in chosen}
        chosen_groups = {r["duplicate_group"] for r in chosen}
        if len(chosen_groups) != len(chosen):
            raise ValueError("Proposed training rows repeat a duplicate component")
        heldout = [r["id"] for r in cases if case_group[r["id"]] not in chosen_groups]
        excluded = [r["id"] for r in cases if r["id"] not in chosen_ids and case_group[r["id"]] in chosen_groups]
        assert set(chosen_ids).isdisjoint(heldout)
        assert len(chosen_ids) + len(heldout) + len(excluded) == len(cases)
        uniform = assign(cases, records, case_group, low, metric)
        assert uniform["feasible"]
        uniform_ids = {r["id"] for r in uniform["selected"]}
        uniform_groups = {r["duplicate_group"] for r in uniform["selected"]}
        assert len(uniform_ids) == len(uniform_groups) == 26 * low
        uniform["proposed_heldout_ids"] = [r["id"] for r in cases if case_group[r["id"]] not in uniform_groups]
        uniform["excluded_duplicate_ids"] = [r["id"] for r in cases
            if r["id"] not in uniform_ids and case_group[r["id"]] in uniform_groups]
        assert len(uniform_ids) + len(uniform["proposed_heldout_ids"]) + len(uniform["excluded_duplicate_ids"]) == len(cases)
        results[metric] = {"requested_five_each": requested,
            "maximum_uniform_pairs_per_category_after_duplicate_grouping": low,
            "maximum_uniform_assignment": uniform,
            "proposed_heldout_ids": heldout, "excluded_duplicate_ids": excluded,
            "proposed_split_complete": requested["feasible"], "training_started": False}
    write(args.run / "balance_feasibility.json", {
        "method": "Exact bipartite matching: at most one training row per connected duplicate group",
        "group_keys": ["normalized_question_sha256", "any_SD_SD_TYPO_TYPO_image_sha256"],
        "n": len(cases), "duplicate_group_count": len(groups),
        "multirow_groups": {k: v for k, v in groups.items() if len(v) > 1},
        "baseline_sha256": verification["baseline_sha256"], "script_sha256": sha(Path(__file__)),
        "results": results, "note": "Proposals only. No boundary fit or attack was run; labels remain lexical proxies."})
    print(json.dumps({metric: {"five_each_feasible": r["requested_five_each"]["feasible"],
        "matched": r["requested_five_each"]["matched_n"],
        "max_uniform_pairs": r["maximum_uniform_pairs_per_category_after_duplicate_grouping"]}
        for metric, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
