"""Verify saved pilot artifacts and ASR denominators without target inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attack_scripts.jailbound_paper_protocol import (
    MODES,
    read_rows,
    refusal_score,
    sha,
    validate_split,
)
from core.metrics import wilson_confidence_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    config = json.loads((out / 'manifest.json').read_text())
    for name, expected in config['implementation_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'implementation changed: {name}')
    if sha(out / 'paper.html') != config['paper_html_sha256']:
        raise ValueError('paper snapshot changed')
    source_cases = json.loads((out / 'source_cases.json').read_text())
    if hashlib.sha256(json.dumps(source_cases, sort_keys=True).encode()).hexdigest() != config['source_cases_sha256']:
        raise ValueError('source case text or identity changed')
    if sha(ROOT / 'data/MM-SafetyBench/10-Legal_Opinion/10-Legal_Opinion.json') != config['source_json_sha256']:
        raise ValueError('source question JSON changed')
    for row in source_cases:
        if sha(Path(row['image'])) != row['image_sha256']:
            raise ValueError('source image changed')
    baselines = read_rows(out / 'baseline.jsonl')
    by_id = {r['id']: r for r in baselines}
    if len(baselines) != 130 or len(by_id) != 130:
        raise ValueError('baseline coverage changed')
    if config['baseline_reuse'] and sha(out / 'baseline.jsonl') != config['baseline_reuse']['baseline_sha256']:
        raise ValueError('reused baseline changed')
    split = json.loads((out / 'split.json').read_text())
    validate_split(split['train'], split['evaluation'])
    for row in split['train'] + split['evaluation']:
        if row['label'] != 1 - by_id[row['id']]['paper_success']:
            raise ValueError('incorrect INPUT unsafety label direction')
    phrases = json.loads((out / 'paper_tables.json').read_text())['reject_templates']
    records = read_rows(out / 'evaluation.jsonl')
    expected = {(r['id'], mode) for r in split['evaluation'] for mode in MODES}
    keys = {(r['id'], r['mode']) for r in records}
    if keys != expected or len(records) != len(keys):
        raise ValueError('incomplete or duplicate pilot responses')
    for row in baselines + records:
        if any(row[k] != value for k, value in refusal_score(row['response'], phrases).items()):
            raise ValueError('stale refusal score')
    summaries = json.loads((out / 'summary.json').read_text())
    checked_rows = []
    pilot_baseline = {r['id']: r for r in records if r['mode'] == 'baseline'}
    paired = []
    for row in summaries['rows']:
        group = [r for r in records if r['mode'] == row['condition']]
        success = sum(r['paper_success'] for r in group)
        if len(group) != row['n'] or success != row['successes'] or success / len(group) != row['asr']:
            raise ValueError('wrong ASR numerator or denominator')
        checked_rows.append({**row, 'asr_ci95': wilson_confidence_interval(success, len(group))})
        paired.append({'condition': row['condition'],
                       'initially_refused_inputs': sum(not pilot_baseline[r['id']]['paper_success'] for r in group),
                       'refusal_to_nonrefusal': sum(not pilot_baseline[r['id']]['paper_success'] and r['paper_success'] for r in group),
                       'nonrefusal_to_refusal': sum(pilot_baseline[r['id']]['paper_success'] and not r['paper_success'] for r in group)})
    artifacts = []
    for row in split['evaluation']:
        original = np.asarray(Image.open(row['image']).convert('RGB').resize((336, 336), Image.Resampling.BICUBIC)).astype(np.int16)
        for mode in ('text', 'image', 'joint'):
            folder = out / 'inputs' / row['id'] / mode
            meta = json.loads((folder / 'optimization.json').read_text())
            if sha(folder / 'input.safetensors') != meta['tensor_sha256'] or sha(folder / 'image.png') != meta['png_sha256']:
                raise ValueError('optimized input artifact changed')
            tensors = load_file(str(folder / 'input.safetensors'), device='cpu')
            if tensors['suffix_ids'].numel() != (0 if mode == 'image' else 20):
                raise ValueError('wrong suffix length in saved input')
            png = np.asarray(Image.open(folder / 'image.png')).astype(np.int16)
            tensor_rgb = np.rint(tensors['pixels'][0].permute(1, 2, 0).numpy() * 255).astype(np.int16)
            if not np.array_equal(tensor_rgb, png):
                raise ValueError('PNG and inference tensor differ')
            max_delta = int(np.abs(png - original).max())
            if max_delta > 8 or (mode == 'text' and max_delta != 0):
                raise ValueError('PNG image perturbation violates the paper budget')
            if len(meta['history']) != 100 or meta['history'][-1]['step'] != 100:
                raise ValueError('optimization iterations are incomplete')
            if not all(np.isfinite(r['loss']) for r in meta['history']):
                raise ValueError('non-finite optimization history')
            artifacts.append({'id': row['id'], 'mode': mode, 'steps': 100,
                              'png_max_delta_uint8': max_delta,
                              'loss_initial': meta['history'][0]['loss'], 'loss_final': meta['history'][-1]['loss'],
                              'final_suffix_changes': meta['history'][-1]['suffix_changes_from_initial']})
    legacy = json.loads((out / 'legacy_audit.json').read_text())
    for path, expected_sha in legacy['source_sha256'].items():
        if sha(ROOT / path) != expected_sha:
            raise ValueError('legacy experiment source modified')
    direction = json.loads((out / 'label_direction_review.json').read_text())
    if not direction['train_ids_unchanged'] or not direction['evaluation_ids_unchanged']:
        raise ValueError('label correction altered selected images')
    diagnostic = Path(config['baseline_reuse']['source']) if config['baseline_reuse'] else None
    diagnostic_rows = read_rows(diagnostic / 'evaluation.jsonl') if diagnostic else []
    review = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Table 3 Qwen2.5-VL Legal Opinion pilot; all other cells remain unmeasured',
        'original_paper_reproduction_confirmed': False,
        'probe_images': len(split['train']), 'heldout_evaluation_images': len(split['evaluation']),
        'responses_in_final_comparison': len(records), 'missing_responses': 0,
        'reused_baseline_responses': len(baselines) if config['baseline_reuse'] else 0,
        'new_corrected_attack_generations': sum(r['mode'] != 'baseline' for r in records),
        'initial_opposite_label_attack_generations_excluded': sum(r['mode'] != 'baseline' for r in diagnostic_rows),
        'png_budget_verified_as_integer_levels': True, 'implementation_and_source_hashes_match': True,
        'legacy_artifacts_unchanged': True, 'input_label_orientation_verified': True,
        'unique_reject_templates': len(phrases), 'rows': checked_rows, 'optimization_artifacts': artifacts,
        'identical_train_eval_question_texts': len(
            {' '.join(r['text'].casefold().split()) for r in split['train']} &
            {' '.join(r['text'].casefold().split()) for r in split['evaluation']}),
        'metric_note': 'literal non-refusal ASR; not harm-compliance accuracy or Pass@25',
        'ci_note': 'Wilson interval is descriptive for this small fixed-protocol pilot',
        'paired_changes_from_baseline': paired,
    }
    (out / 'final_review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps({k: v for k, v in review.items() if k not in ('rows', 'optimization_artifacts')}, indent=2))


if __name__ == '__main__':
    main()
