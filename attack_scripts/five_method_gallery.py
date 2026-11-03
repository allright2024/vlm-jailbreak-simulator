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
        source = mixing["source_weight"]
        html.append(f'<p><strong>JOOD 이미지 혼합:</strong> 제공 이미지 {len(mixing["paths"])}종과 MixUp. '
                    f'원래 이미지 {source:.0%} + JOOD 이미지 {1-source:.0%}. 원래 해상도를 유지합니다.</p>')
        if full_qc and "jood_image_mixed_cases" in full_qc:
            html.append(f'<p>JOOD 제공 이미지와 혼합된 입력: {full_qc["jood_image_mixed_cases"]:,}개.</p>')
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
            html.append(f'<li>{escape(case["behavior_id"])}: ' + ' · '.join(links) + description + '</li>')
        html.append('</ul></section>')
    html.append('</html>')
    destination = root / "index.html"
    destination.write_text('\n'.join(html), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
