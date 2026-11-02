"""Extract numerical paper references; never fill unexecuted cells with zero."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attack_scripts.jailbound_paper_protocol import parse_paper


def reference_rows(table, models):
    result = []
    categories = None
    for row in table[2:]:
        if len(row) == 10:
            categories = row[0], row[5]
            parts = (row[1:5], row[6:10])
        elif len(row) == 8:
            parts = (row[:4], row[4:])
        else:
            raise ValueError(f"unexpected paper table layout: {len(row)} cells")
        for category, part in zip(categories, parts):
            for model, value in zip(models, part[1:]):
                if not value.endswith('%'):
                    raise ValueError("paper ASR cell is not a percentage")
                result.append({"category": category, "condition": part[0], "model": model,
                               "paper_asr": float(value.rstrip('% ')) / 100,
                               "measured_asr": None, "measured_n": 0, "status": "not_executed"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paper = parse_paper(args.out / 'paper.html')
    references = {
        3: reference_rows(paper['tables']['S4.T3.5.1'], ('Llama-3.2-11B', 'Qwen2.5-VL-7B', 'MiniGPT-4')),
        4: reference_rows(paper['tables']['S4.T4.5.1'], ('GPT-4o', 'Gemini 2.0 Flash', 'Claude 3.5 Sonnet')),
    }
    for table, rows in references.items():
        if len(rows) != 210:
            raise ValueError(f'Table {table} must have 14 categories including Overall x 5 x 3')
        with (args.out / f'paper_table{table}_reference.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    audit = {
        'paper': paper['paper'], 'html_sha256': paper['html_sha256'],
        'table1': 'configuration definitions; not numerical attack results',
        'table2': 'white-box model components; existing LLaVA backend is a substitute',
        'table3': {'scope': 'white-box comparison', 'reference_cells_including_overall': 210,
                   'completed_full_table_cells': 0, 'current_experiment': 'Qwen2.5-VL Legal Opinion five-condition pilot'},
        'table4': {'scope': 'black-box transfer', 'reference_cells_including_overall': 210, 'executed_cells': 0},
        'tables5_and6': {'scope': 'rejection phrase taxonomy used by the ASR scorer',
                         'unique_templates': len(paper['reject_templates']),
                         'casing_and_printed_escape_ambiguity': paper['matching_note'],
                         'broad_substrings': [p for p in ('No', 'public', 'unknown', 'however', 'Note') if p in paper['reject_templates']]},
        'missing_for_exact_replication': ['author probe/evaluation IDs and safety labels',
                                          'exact checkpoint revisions and decoding',
                                          'image variant and preprocessing', 'sequence pooling',
                                          'suffix regularizer definition'],
        'geometric_sign_issue': 'Algorithm 2 targets h - epsilon*v, but Eq.7 attracts normalized delta_h toward +v. Published signs preserved.',
        'dataset_count_discrepancy': {'paper_stated': 1719, 'local_public_questions': 1680},
        'paper_reproduction': False,
    }
    (args.out / 'paper_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
