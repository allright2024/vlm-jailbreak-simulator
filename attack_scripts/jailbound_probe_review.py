"""Held-out probe accuracy for the saved pilot; performs no text generation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attack_scripts.jailbound_paper_protocol import sha, validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.out / 'manifest.json').read_text())
    split = json.loads((args.out / 'split.json').read_text())
    validate_split(split['train'], split['evaluation'])
    integrity = json.loads((args.out / 'boundary_integrity.json').read_text())
    if sha(args.out / 'boundaries.pt') != integrity['sha256']:
        raise ValueError('boundary artifact changed')
    from PIL import Image

    from multi_attacks.jailbound.qwen_probe_adapter import QwenProbeAdapter
    from multimodalmodels.loaders import load_torch_checkpoint

    boundaries = load_torch_checkpoint(str(args.out / 'boundaries.pt'), map_location='cpu', weights_only=True)
    adapter = QwenProbeAdapter(config['model_id'], config['revision'])
    predictions = []
    for row in split['evaluation']:
        if sha(Path(row['image'])) != row['image_sha256']:
            raise ValueError('held-out image changed')
        image = Image.open(row['image'])
        features = adapter.extract_features(adapter.prepare_image(image), adapter.prepare_context(image, row['text']), list(boundaries))
        for layer, feature in features.items():
            boundary = boundaries[layer]
            score = float(feature.cpu() @ boundary['w'] + boundary['b'])
            predictions.append({'id': row['id'], 'layer': layer, 'label': row['label'],
                                'logit': score, 'prediction_at_05': int(score >= 0),
                                'prediction_at_03': int(score >= -0.8472978603872036)})
    table = []
    for layer, boundary in boundaries.items():
        group = [r for r in predictions if r['layer'] == layer]
        table.append({'layer': layer, 'train_n': len(split['train']),
                      'training_accuracy': boundary['training_accuracy'],
                      'heldout_n': len(group),
                      'heldout_accuracy_at_05': sum(r['label'] == r['prediction_at_05'] for r in group) / len(group),
                      'heldout_accuracy_at_03': sum(r['label'] == r['prediction_at_03'] for r in group) / len(group)})
    report = {'scope': 'held-out classification diagnostic, not attack ASR',
              'label_definition': config['boundary_labels'], 'new_generation_queries': 0,
              'boundary_sha256': integrity['sha256'], 'rows': table, 'predictions': predictions}
    (args.out / 'heldout_probe_review.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.out / 'heldout_probe_accuracy.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({'layers': len(table), 'heldout_images': len(split['evaluation']),
                      'training_accuracy_range': [min(r['training_accuracy'] for r in table), max(r['training_accuracy'] for r in table)],
                      'heldout_accuracy_range': [min(r['heldout_accuracy_at_05'] for r in table), max(r['heldout_accuracy_at_05'] for r in table)]}))


if __name__ == '__main__':
    main()
