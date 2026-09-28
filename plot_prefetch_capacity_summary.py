#!/usr/bin/env python3
"""Plot derived results only; never starts a model or changes cache state."""
import argparse
import json
from pathlib import Path

from report_capacity_matrix import save_chart


def main(root):
    rows = json.loads((root / 'summary.json').read_text())
    assert len(rows) == 72
    warm = {(r['concurrency'], r['cpu_gib'], r['staging_gib'], r['mode']): r
            for r in rows if r['phase'] == 'warm'}
    colors = ['#777777', '#0072b2', '#d55e00']
    modes = ['off', 'wait', 'cancel']
    labels = ['OFF', 'ON / wait', 'ON / cancel + early readiness']
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820">',
           '<rect width="1200" height="820" fill="white"/>',
           '<g font-family="sans-serif" fill="#222">',
           '<text x="35" y="30" font-size="22">Warm mean TTFT: 36-condition capacity sweep</text>',
           '<text x="35" y="55" font-size="14">Qwen3-14B / chunk128 / 256 fixed replay requests per phase / one trial per condition</text>']
    for i, (color, label) in enumerate(zip(colors, labels)):
        x = 35 + i * 330
        svg += [f'<rect x="{x}" y="73" width="14" height="14" fill="{color}"/>',
                f'<text x="{x+21}" y="85" font-size="14">{label}</text>']
    for concurrency, top in [(8, 135), (16, 465)]:
        svg.append(f'<text x="35" y="{top-15}" font-size="18">Concurrency {concurrency} (lower is better)</text>')
        bottom = top + 225
        for value in range(0, 251, 50):
            y = bottom - value * .9
            svg += [f'<line x1="75" x2="1170" y1="{y}" y2="{y}" stroke="#ddd"/>',
                    f'<text x="30" y="{y+4}" font-size="12">{value}</text>']
        svg.append(f'<text x="5" y="{top+20}" font-size="12">ms</text>')
        for i, (dram, staging) in enumerate((d, s) for d in (8, 4, 2) for s in (8, 4)):
            left = 98 + i * 178
            for j, mode in enumerate(modes):
                row = warm[concurrency, dram, staging, mode]
                val = row['ttft_ms']['mean']
                x, y = left + j * 48, bottom - val * .9
                svg += [f'<rect x="{x}" y="{y}" width="40" height="{val*.9}" fill="{colors[j]}"/>',
                        f'<text x="{x+20}" y="{y-7}" text-anchor="middle" font-size="12">{val:.1f}</text>']
            svg.append(f'<text x="{left+67}" y="{bottom+25}" text-anchor="middle" font-size="13">D{dram} / S{staging} GiB</text>')
    svg += ['<text x="35" y="755" font-size="13">D = DRAM, S = GPU staging. Queued cancellations: 0 in every case.</text>',
            '<text x="35" y="780" font-size="13">Some C16 ON cases lost reusable prefix after staging allocation failures; compute work then differs.</text>',
            '<text x="35" y="803" font-size="13">No error bars: one trial. ON/cancel also changes readiness timing, not just cancellation.</text></g></svg>']
    save_chart(root, 'warm_ttft_comparison', svg)
    lines = ['# 시간별 staging 점유율·hit 그래프', '',
             '각 링크에는 위쪽 staging 점유율, 아래쪽 DRAM/DAOS lookup hit 비율이 있다.',
             'staging의 파랑은 2초 구간 내 표본 최댓값, 초록은 표본 평균이다. 짧은 점유 급증과 구간 평균은 다르다.',
             'hit의 파랑은 DRAM, 초록은 DAOS다. lookup hit는 최종 재사용 성공과 다를 수 있다.',
             '각 그래프의 시간 0은 해당 단계의 시작이다. 서로 다른 실행의 같은 시각이 같은 요청을 뜻하지 않는다.', '']
    for phase in ('warm', 'cold'):
        lines += [f'## {phase}', '', '|동시 요청|DRAM GiB|staging GiB|OFF|ON/취소 OFF|ON/취소 ON|',
                  '|---:|---:|---:|---|---|---|']
        for c in (8, 16):
            for d in (8, 4, 2):
                for s in (8, 4):
                    links = [f'[그래프](c{c}_d{d}_s{s}_{m}/{phase}/staging_hits.png)' for m in modes]
                    lines.append(f'|{c}|{d}|{s}|' + '|'.join(links) + '|')
    (root / 'GRAPH_INDEX_KO.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    main(parser.parse_args().folder.resolve())
