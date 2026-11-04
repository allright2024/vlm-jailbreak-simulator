#!/usr/bin/env python
"""Build a local review report from validated preview/production artifacts."""
import argparse
import json
import os
import sys
from collections import defaultdict
from html import escape
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.paths import resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    cases = [json.loads(line) for line in (root / "preview/test_cases.jsonl").read_text().splitlines()]
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["meta"]["recipe_id"]].append(case)
    qc = json.loads((root / "preview/qc.json").read_text())
    if qc["status"] != "passed" or qc["n_cases"] != 54:
        raise ValueError("a validated 54-case preview is required")
    full = root / "full/qc.json"
    full_qc = json.loads(full.read_text()) if full.exists() else None
    html = ['<!doctype html><html lang="ko"><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>5개 방법 조합 샘플 검토</title><style>',
            'body{font:16px/1.6 system-ui;margin:32px auto;max-width:1500px;padding:0 20px;background:#f4f5f7;color:#172033}',
            'section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{max-width:100%;height:auto}',
            'a{color:#155dc0}nav a{display:inline-block;padding:6px 12px}li{margin:8px 0}',
            '</style><h1>5개 방법 조합 샘플 검토</h1>',
            '<p>18개 조합 × 3개 미리보기 = 54개. 모든 미리보기의 텍스트 보존·글자 영역·파일 해시 검사를 통과했습니다.</p>',
            '<p>이미지를 누르면 원본 검토 시트를 열 수 있습니다. 각 샘플의 입력 이미지도 아래 링크에서 확인할 수 있습니다.</p>']
    if full_qc and full_qc.get("status") == "passed":
        html.append(f'<p><strong>전체 생성: {full_qc["n_cases"]:,}개 / 중복 없는 입력: {full_qc["unique_inputs"]:,}개 / 글자 영역 초과: {full_qc["clipped_layouts"]}개</strong></p>')
        html.append('<p><a href="full/test_cases.jsonl">전체 입력 JSONL</a> · <a href="full/qc.json">전체 검사 결과</a></p>')
    identity = json.loads((root / "run_identity.json").read_text())
    mixing = identity.get("jood")
    if mixing:
        weights = mixing.get("source_weights", [mixing.get("source_weight", 0.7)])
        html.append(f'<p><strong>JOOD 이미지 혼합:</strong> 제공 이미지 {len(mixing["paths"])}종과 MixUp. '
                    f'원본 이미지 비율 {min(weights):.0%}–{max(weights):.0%}, {len(weights)}단계. 원래 해상도를 유지합니다.</p>')
        if full_qc and "jood_image_mixed_cases" in full_qc:
            html.append(f'<p>JOOD 제공 이미지와 혼합된 입력: {full_qc["jood_image_mixed_cases"]:,}개.</p>')
    sources = (full_qc or {}).get("source_counts") or identity.get("source_selection", {}).get("full_source_counts")
    if sources:
        html.append('<p><strong>벤치마크별 전체 입력:</strong> ' +
                    ' · '.join(f'{escape(k)} {v:,}개' for k,v in sources.items()) + '</p>')
    selection = identity.get("source_selection", {})
    if selection.get("selection_policy") == "all_records_once_then_resample":
        html.append(f'<p><strong>전체 레코드 사용:</strong> {selection["unique_parents"]:,}개 원본 레코드를 모두 한 번씩 사용하고, '
                    f'{selection["resampled_draws"]:,}개 입력만 추가합니다. 조언·의견 범주와 중복 원본 레코드도 보존합니다.</p>')
        html.append(f'<p>서로 다른 원문: {selection["unique_instructions"]:,}개. 각 레코드는 최대 두 번 사용하고, '
                    '동일한 질문·조합 쌍은 중복되지 않습니다.</p>')
    if selection.get("selection_policy") == "full_pool_cycles_one_recipe_per_draw":
        html.append('<p><strong>샘플링:</strong> 전체 가용 질문 풀을 사용하고, 질문을 한 번 뽑을 때 조합 하나만 적용합니다. '
                    '각 출처의 전체 풀을 모두 사용한 뒤 필요한 만큼 재샘플링합니다.</p>')
        html.append(f'<p>서로 다른 원본 질문: {selection["unique_parents"]:,}개. '
                    '같은 원본 질문·조합 쌍은 중복되지 않습니다.</p>')
    html.append('<nav>' + ''.join(f'<a href="#{escape(k)}">{escape(k)}</a>' for k in grouped) + '</nav>')
    for recipe, items in grouped.items():
        methods = ' + '.join(items[0]["meta"]["methods"])
        sheet = f'preview/contact_sheets/{recipe}.png'
        html.extend([f'<section id="{escape(recipe)}"><h2>{escape(recipe)} · {escape(methods)}</h2>',
                     f'<a href="{sheet}"><img loading="lazy" src="{sheet}" alt="{escape(recipe)} 세 개 미리보기"></a><ul>'])
        for case in items:
            links = []
            for i, name in enumerate(case["images"]):
                relative = os.path.relpath(resolve_path(name), root)
                links.append(f'<a href="{quote(relative)}">입력 이미지 {i+1}</a>')
            aug = case["meta"].get("jood")
            description = ''
            if aug and aug.get("operation") == "mixup":
                harmless = os.path.relpath(resolve_path(aug["harmless_image"]), root)
                description = (f' · JOOD: <a href="{quote(harmless)}">{escape(aug["harmless"])}</a>'
                               f' · 원래 이미지 비율 {aug["source_weight"]:.0%}')
            source = escape(case["meta"].get("source_dataset", ""))
            html.append(f'<li>{source} · {escape(case["behavior_id"])}: ' + ' · '.join(links) + description + '</li>')
        html.append('</ul></section>')
    html.append('</html>')
    destination = root / "index.html"
    destination.write_text('\n'.join(html), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
