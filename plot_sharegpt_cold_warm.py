#!/usr/bin/env python3
"""One-page native SVG/PNG and a self-contained Korean HTML experiment report."""
import argparse
import base64
import copy
import html
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root.resolve()
    read = lambda name: json.loads((root/name).read_text())
    assert read('status.json')['status'] == 'completed'
    rows = {(s['case'], s['phase']): s for s in read('summary.json')}
    order = [(f'c16_{mode}', phase) for phase in ('cold', 'warm') for mode in ('off', 'on')]
    assert len(rows) == 4
    off, on = [rows[(f'c16_{m}', 'warm')] for m in ('off', 'on')]
    ttft_gain = 100*(1-on['ttft']['mean']/off['ttft']['mean'])
    time_gain = 100*(1-on['elapsed_seconds']/off['elapsed_seconds'])
    svg = ET.Element(f'{{{NS}}}svg', width='2000', height='1730', viewBox='0 0 2000 1730')
    def element(tag, **kwargs):
        return ET.SubElement(svg, f'{{{NS}}}{tag}', {k:str(v) for k,v in kwargs.items()})
    def text(x, y, value, size=20, color='#172b4d', weight='normal'):
        e = element('text', x=x, y=y, fill=color)
        e.set('font-family', 'DejaVu Sans, sans-serif')
        e.set('font-size', str(size)); e.set('font-weight', weight)
        e.text = value
    element('rect', width=2000, height=1730, fill='white')
    text(45, 46, 'ShareGPT | Cold + Warm | DRAM prefetch OFF vs ON', 32, weight='bold')
    text(45, 82, 'Qwen3-14B BF16  |  DRAM 256GiB  |  GPU staging 8GiB  |  concurrency 16  |  chunk 128 tokens', 22)
    text(45, 115, '421 conversations x 4 fixed-history turns = 1,684 requests/phase. Same live cache from cold to warm within each arm.', 19)
    element('rect', x=35, y=137, width=1930, height=52, rx=8, fill='#edf7f3')
    text(52, 171, f'Warm: mean TTFT {ttft_gain:.2f}% lower  |  elapsed time {time_gain:.2f}% lower  |  input reuse {on["input_reuse_pct"]:.2f}% in both arms', 25, '#14634d', 'bold')
    xs = [50, 210, 390, 630, 850, 1070, 1290, 1510, 1760]
    headers = ['Phase', 'Prefetch', 'Time (s)', 'TTFT (ms)', 'p95 (ms)', 'DRAM hit', 'DAOS hit', 'Reuse*', 'Peak stage']
    element('rect', x=35, y=205, width=1930, height=43, fill='#edf1f7')
    for x, header in zip(xs, headers):
        text(x, 234, header, 21, weight='bold')
    for i, key in enumerate(order):
        s = rows[key]
        y = 279+42*i
        if i % 2:
            element('rect', x=35, y=y-27, width=1930, height=40, fill='#f8fafc')
        values = [s['phase'].title(), key[0].split('_')[-1].upper(), f'{s["elapsed_seconds"]:.2f}',
            f'{s["ttft"]["mean"]:.2f}', f'{s["ttft"]["p95"]:.2f}', f'{s["dram_hit_pct"]:.2f}%',
            f'{s["daos_hit_pct"]:.2f}%', f'{s["input_reuse_pct"]:.2f}%', f'{s["peak_staging_gib"]:.2f} / 8GiB']
        for x, value in zip(xs, values):
            text(x, y, value, 22)
    text(45, 448, '*Reuse = cached input tokens / all input tokens. Tier hit ratios use initial lookup candidate chunks (different denominator).', 18)
    text(45, 477, 'Each arm: cold fill -> drain staging/mirror -> warm replay. Between arms: fresh process, empty DRAM, new DAOS namespace.', 18)
    for i, (case, phase) in enumerate(order):
        panel = copy.deepcopy(ET.parse(root/case/phase/'staging_hits.svg').getroot())
        panel.set('x', str((i % 2)*1000))
        panel.set('y', str(500+(i//2)*570))
        svg.append(panel)
    text(45, 1665, 'All charts: x = seconds since phase start; y = percent. Independent time axes. 20ms samples, 2-second bins.', 18)
    text(45, 1691, 'Table peak uses allocation events; charts can miss short peaks. Staging includes reads and stores, not GPU compute utilization.', 18)
    text(45, 1717, 'One trial per arm; identical cached-token counts per request, but some outputs differ. No capacity-attributed recomputation observed.', 18)
    ET.ElementTree(svg).write(root/'OVERVIEW.svg', encoding='utf-8', xml_declaration=True)
    subprocess.run(['rsvg-convert', '-o', str(root/'OVERVIEW.png'), str(root/'OVERVIEW.svg')], check=True)

    table_rows = []
    for key in order:
        s = rows[key]
        values = [s['phase'], key[0].split('_')[-1].upper(), str(s['requests']),
            f'{s["elapsed_seconds"]:.2f}', f'{s["ttft"]["mean"]:.2f}', f'{s["ttft"]["p95"]:.2f}',
            f'{s["dram_hit_pct"]:.2f}%', f'{s["daos_hit_pct"]:.2f}%', f'{s["input_reuse_pct"]:.2f}%',
            f'{s["peak_staging_gib"]:.2f}', f'{100*s["mean_sampled_staging_gib"]/8:.2f}%',
            str(s['capacity_recomputed_tokens'])]
        table_rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    cards = []
    for case, phase in order:
        payload = base64.b64encode((root/case/phase/'staging_hits.png').read_bytes()).decode()
        cards.append(f'<section class="card"><h2>{phase.upper()} · {case.split("_")[-1].upper()}</h2>'
                     f'<img src="data:image/png;base64,{payload}" alt="{case} {phase} staging and hit timeline"></section>')
    headers_ko = ['단계','DRAM 프리페치','요청 수','전체 시간(s)','평균 TTFT(ms)','p95(ms)',
                  'DRAM hit','DAOS hit','입력 재사용','최대 staging(GiB)','평균 staging 점유','용량 실패 재계산(토큰)']
    document = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ShareGPT cold/warm 한눈에 보기</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#172b4d;background:#f4f7fb}}
main{{max-width:1800px;margin:auto}}h1{{font-size:28px}}p,li{{line-height:1.65}}
.highlight{{background:#e6f4ed;padding:16px;border-radius:10px;font-size:21px;font-weight:650}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.card{{background:white;padding:12px;border-radius:10px}}
.card h2{{margin:4px 12px;font-size:20px}}img{{width:100%;height:auto}}.scroll{{overflow:auto;background:white;border-radius:10px}}
table{{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}}td,th{{padding:12px 10px;text-align:right;border-bottom:1px solid #ddd}}
th{{background:#eaf0f7}}td:first-child,th:first-child{{text-align:left}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}body{{margin:12px}}}}
@media print{{body{{background:white;margin:0}}.card{{break-inside:avoid}}}}
</style><main><h1>ShareGPT cold / warm — 프리페치 OFF·ON 한눈에 보기</h1>
<p>Qwen3-14B BF16 · DRAM <b>256GiB</b> · GPU staging <b>8GiB</b> · 동시 요청 <b>16</b> · 청크128토큰<br>
421개 대화 × 앞4턴 = 단계당 1,684개 요청, 총6,736개. 기존 원본 대화 이력 재생이며 새 생성 답변을 다음 입력에 넣지 않음.</p>
<p class="highlight">Warm: 평균 TTFT {ttft_gain:.2f}% 감소 · 전체 시간 {time_gain:.2f}% 감소 · 입력 토큰 {on['input_reuse_pct']:.2f}% 재사용</p>
<p>각 OFF/ON은 빈 캐시로 cold 시작 → 비동기 쓰기와 staging 해제 완료 → <b>같은 프로세스·DRAM·DAOS를 유지해 warm</b> 재실행.
DRAM 프리페치만 비교하며 <b>DAOS 프리페치는 항상 ON</b>. 작업자1, 대기열 취소OFF, 점유율 조기차단 없음.</p>
<div class="scroll"><table><thead><tr>{''.join('<th>'+h+'</th>' for h in headers_ko)}</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<h2>그래프 보는 법</h2><ul>
<li><b>왼쪽 OFF / 오른쪽 ON, 위쪽 cold / 아래쪽 warm.</b> 휴대폰에서는 순서대로 표시.</li>
<li>각 그림 위 패널: staging 점유율 — 파란색은2초 구간의 샘플 최대, 초록색은 샘플 평균. 100% = 8GiB.</li>
<li>각 그림 아래 패널: 파란색은 DRAM hit, 초록색은 DAOS hit. X축은 해당 단계 시작 이후 초이며 그림마다 종료 시간이 다름.</li>
<li>Warm의 DRAM+DAOS는 조회 후보 청크 기준100% hit. 전체 입력 토큰 기준 재사용은98.16%로, 분모가 다름.</li>
</ul><div class="grid">{''.join(cards)}</div>
<h2>해석할 때 주의</h2><p>각 조건1회 예비실험. 요청별 cached_tokens는 OFF/ON 모두 일치하지만 생성 결과는 일부 다름.
최대 점유는 할당 이벤트 기준이고 그래프는20ms 샘플을2초 단위로 묶어 짧은 peak가 빠질 수 있음.
staging은 읽기와 쓰기 모두 포함하며 GPU 계산 사용률이 아님. 모델 시작·메모리 초기화·drain·삭제는 요청 시간에서 제외.
OS/서버 캐시는 강제 초기화하지 않음. 이 HTML은 그래프를 내장해 파일 하나로 열 수 있음.</p></main></html>'''
    (root/'OVERVIEW.html').write_text(document)
    report = root/'RESULT_KO.md'
    marker = '\n## 한눈에 보기\n'
    content = report.read_text().split(marker)[0]
    report.write_text(content+marker+'\n[통합 이미지](OVERVIEW.png) · [한글 대시보드](OVERVIEW.html)\n\n'
                      '왼쪽 OFF / 오른쪽 ON, 위쪽 cold / 아래쪽 warm. 결과 표와 네 조건의 점유율·hit 그래프를 한 장에 모았다.\n')
    print(root/'OVERVIEW.png')
    print(root/'OVERVIEW.html')


if __name__ == '__main__':
    main()
