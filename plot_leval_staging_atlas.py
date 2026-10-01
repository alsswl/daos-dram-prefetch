#!/usr/bin/env python3
"""Render existing staging telemetry only; does not run models or touch caches."""
import argparse
import html
import json
import math
from pathlib import Path
import subprocess


def render(root, rows, concurrency, repeat, xmax):
    width, height = 1650, 1720
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" fill="#182333">']

    def label(x, y, value, size=15, **attrs):
        attr = ' '.join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" {attr}>{html.escape(str(value))}</text>')

    suffix = 'all 3 repeats' if repeat is None else f'repeat {repeat}'
    label(40, 43, f'L-Eval | GPU staging occupancy | concurrency {concurrency} | {suffix}', 27)
    label(40, 76, 'Qwen3-14B | 72 requests/run | 2 DRAM prefetch workers | 18 documents x 4 questions', 17)
    colors = {1: '#0072b2', 2: '#d55e00', 3: '#009e73'}
    for r, color in colors.items():
        x = 40 + (r-1)*160
        parts.append(f'<line x1="{x}" x2="{x+30}" y1="105" y2="105" stroke="{color}" stroke-width="3"/>')
        label(x+40, 111, f'Repeat {r}', 16)
    label(570, 111, 'Solid: sampled peak per 2s bin   |   Dashed: sampled mean per 2s bin', 16)
    label(40, 143, 'Y: allocated staging / configured capacity (%)    X: seconds since each run started', 16)
    modes = [('off', 'OFF'), ('wait', 'ON / wait'), ('cancel', 'ON / cancel queued')]
    for col, (_, title) in enumerate(modes):
        label(305+col*540, 186, title, 23, text_anchor='middle')
    for row, (dram, staging) in enumerate((d, s) for d in (4, 8, 16) for s in (4, 8)):
        for col, (mode, _) in enumerate(modes):
            left, top, w, h = 80+540*col, 246+232*row, 460, 143
            label(left, top-29, f'DRAM {dram} GiB / staging {staging} GiB', 18)
            for tick in (0, 25, 50, 75, 100):
                y = top+h*(1-tick/100)
                parts.append(f'<line x1="{left}" x2="{left+w}" y1="{y}" y2="{y}" stroke="#dfe4e9"/>')
                label(left-12, y+5, tick, 13, text_anchor='end')
            for tick in range(0, xmax+1, 10):
                x = left+w*tick/xmax
                label(x, top+h+21, tick, 13, text_anchor='middle')
            for r in (range(1, 4) if repeat is None else [repeat]):
                name = f'c{concurrency}_d{dram}_s{staging}_r{r}_{mode}'
                bins = json.loads((root/name/'timeline_bins.json').read_text())
                # Plot at bin midpoint; clip last bin to the actual run end.
                end = rows[name]['elapsed_seconds']
                for key, dash in [('sampled_peak_gib', ''), ('sampled_mean_gib', 'stroke-dasharray="5 4"')]:
                    segments = []
                    current = []
                    for b in bins:
                        value = b[key]
                        if value is None:
                            if current:
                                segments.append(current)
                                current = []
                            continue
                        assert 0 <= value <= staging+1e-6, (name, value)
                        seconds = (b['seconds']+min(b['seconds']+2, end))/2
                        assert 0 <= seconds <= xmax, (name, seconds)
                        current.append(f'{left+w*seconds/xmax:.2f},{top+h*(1-value/staging):.2f}')
                    if current:
                        segments.append(current)
                    for segment in segments:
                        parts.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{colors[r]}" stroke-width="1.8" {dash} opacity="0.85"/>')
    label(40, 1647, 'All panels share the same axes. Runs end at different times; no zero padding or time normalization.', 16)
    label(40, 1673, 'Sampled peaks may miss brief allocation spikes. Occupancy is not GPU compute utilization.', 16)
    label(40, 1699, 'OFF disables only DRAM prefetch; DAOS prefetch remains enabled. Equal times do not imply equal requests.', 16)
    parts.append('</g></svg>')
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = json.loads((root/'summary.json').read_text())
    assert len(rows) == 108 and all(r['requests'] == 72 for r in rows)
    rows = {r['name']: r for r in rows}
    assert len(rows) == 108
    xmax = math.ceil(max(r['elapsed_seconds'] for r in rows.values())/10)*10
    out = root/'staging_atlas'
    out.mkdir(exist_ok=True)
    pages = []
    for repeat in (None, 1, 2, 3):
        for c in (8, 16):
            stem = f'staging_c{c}_' + ('all_repeats' if repeat is None else f'r{repeat}')
            svg = out/f'{stem}.svg'
            svg.write_text(render(root, rows, c, repeat, xmax))
            subprocess.run(['rsvg-convert', '-o', str(out/f'{stem}.png'), str(svg)], check=True)
            pages.append(svg)
    subprocess.run(['rsvg-convert', '--format=pdf', '-o', str(out/'staging_all_108_runs.pdf'),
                    *map(str, pages)], check=True)
    notes = ['# L-Eval staging 점유율 전체 모음', '',
             '36개 조건 × 3회 반복 = 108회 실행. 기존 계측 자료만 시각화했으며 실험을 재실행하지 않았다.', '',
             '[전체 PDF: 8페이지](staging_all_108_runs.pdf)', '',
             '- 1~2페이지: 동시 요청 8/16, 3회 반복을 겹쳐 표시.',
             '- 3~8페이지: 반복 1/2/3 각각의 동시 요청 8/16 그래프.',
             '- 행: DRAM 4/8/16GiB × staging 4/8GiB. 열: OFF / ON / ON+대기열 취소.',
             '- 색상: 파랑=1회, 주황=2회, 초록=3회. 실선=2초 구간 표본 최댓값, 점선=표본 평균.',
             '- 모든 패널의 Y축은 0~100%, X축은 동일한 실제 경과 시간(초). 마지막 구간은 실행 종료 시각으로 제한했다.',
             '- 반복을 평균내지 않았고 종료 후 0을 추가하지 않았다. 같은 시각에 같은 요청이 진행된다는 뜻은 아니다.',
             '- 샘플 사이에 일어난 순간 할당 피크는 빠질 수 있다. 요약표의 이벤트 기반 최대 점유율과 다를 수 있다.',
             '- OFF는 DRAM 프리페치만 끈 상태. 점유율은 GPU 연산 사용률이 아니다.', '',
             '원본: 각 실행의 `timeline_bins.json` 및 `summary.json`.', '']
    for page in pages:
        notes += [f'## {page.stem}', '', f'![{page.stem}]({page.stem}.png)', '']
    (out/'README_KO.md').write_text('\n'.join(notes))
    print(json.dumps({'output': str(out), 'runs': len(rows), 'pages': len(pages), 'xmax_seconds': xmax}))


if __name__ == '__main__':
    main()
