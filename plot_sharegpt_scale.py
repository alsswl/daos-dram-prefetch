#!/usr/bin/env python3
"""Combine completed ShareGPT SVG panels and append measured interpretation."""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    root = p.parse_args().root.resolve()
    assert json.loads((root/'status.json').read_text())['status'] == 'completed'
    values = json.loads((root/'summary.json').read_text())
    off, on = [next(r for r in values if r['case'] == name) for name in ('c16_off','c16_on')]
    svg = ET.Element(f'{{{NS}}}svg', width='2000', height='1010', viewBox='0 0 2000 1010')
    ET.SubElement(svg, f'{{{NS}}}rect', width='2000', height='1010', fill='white')
    def text(x, y, value, size=20):
        obj = ET.SubElement(svg, f'{{{NS}}}text', x=str(x), y=str(y),
                            attrib={'font-family':'sans-serif', 'font-size':str(size)})
        obj.text = value
    text(45,30,'ShareGPT: DRAM prefetch OFF vs ON | Qwen3-14B | DRAM 256GiB / staging 8GiB / concurrency 16')
    text(45,57,'421 conversations x 4 original-history turns = 1,684 requests per arm; independent cold starts.',16)
    for column, name in enumerate(('c16_off','c16_on')):
        for filename, y in (('staging_hits.svg',75), ('dram_residency.svg',645)):
            panel = ET.parse(root/name/filename).getroot()
            panel.set('x',str(column*1000))
            panel.set('y',str(y))
            svg.append(panel)
    text(45,998,'20ms occupancy sampling, 2s plot bins. Each arm uses its actual elapsed-time axis. Staging occupancy is not GPU compute utilization.',15)
    path = root/'staging_comparison.svg'
    ET.ElementTree(svg).write(path, encoding='utf-8', xml_declaration=True)
    subprocess.run(['rsvg-convert','-o',str(root/'staging_comparison.png'),str(path)],check=True)
    change = 100*(1-on['ttft']['mean']/off['ttft']['mean'])
    elapsed_change = 100*(1-on['elapsed_seconds']/off['elapsed_seconds'])
    marker = '\n## 점유율과 해석\n'
    report = root/'RESULT_KO.md'
    original = report.read_text().split(marker)[0]
    lines = [marker, '', '[OFF/ON 점유율 비교 그림](staging_comparison.png)', '',
             '|항목|OFF|ON|', '|---|---:|---:|',
             f'|할당 이벤트 기준 최대 staging 점유율|{100*off["peak_staging_gib"]/8:.2f}%|{100*on["peak_staging_gib"]/8:.2f}%|',
             f'|20ms 샘플 평균 staging 점유율|{100*off["mean_sampled_staging_gib"]/8:.2f}%|{100*on["mean_sampled_staging_gib"]/8:.2f}%|',
             f'|staging 할당 실패에 귀속된 재계산 토큰|{off["capacity_recomputed_tokens"]}|{on["capacity_recomputed_tokens"]}|',
             f'|새로 계산한 입력 토큰|{off["computed_tokens"]}|{on["computed_tokens"]}|',
             f'|생성 토큰|{off["output_tokens"]}|{on["output_tokens"]}|', '',
             f'평균 TTFT 감소율은 {change:.2f}%, 전체 시간 감소율은 {elapsed_change:.2f}%이다. '
             'p95 TTFT는 ON이 더 길었다. 각 조건1회라 성능 우열이나 통계적 유의성을 확정하지 않는다.', '',
             'DRAM 256GiB는 두 조건 모두 실제로 찼다. 하지만 staging이 60% 이상 찬 샘플은 두 조건 모두 없고, '
             'DAOS 용량 실패 재계산 및 ON의 DRAM 프리페치 capacity fallback도 0이다. '
             '이 실행에서 지속적인 staging 공간 병목은 관측되지 않았다. 짧은 복사·작업 대기 등 다른 병목이 없다는 뜻은 아니다.', '',
             '시간별 hit 그래프에서 초반은 신규 계산, 중간은 DRAM 재사용, 뒤쪽은 DAOS 재사용 비중이 커진다. '
             '전체 요청의 hit 비율을 매 순간 동일한 혼합 비율로 해석하면 안 된다. '
             'DRAM이 가득 찬 시점과 뒤쪽 DAOS hit 증가가 함께 관측되며, 순환 재방문과 LRU 교체의 영향이 시사된다.', '',
             'ON/OFF의 요청별 cached_tokens는 1,684개 모두 같고 새 계산 토큰 수도 같다. '
             '다만 생성 문자열은 710개만 일치했다. 이 실험은 입력을 원본 이력으로 고정해 생성 차이가 다음 입력을 바꾸지 않도록 했다.', '',
             '모델 시작·256GiB 메모리 초기화·조건 사이 공간 회수 대기 시간은 HTTP 요청 측정에서 제외했다. '
             'OS/서버 캐시를 강제로 비운 실험은 아니며, cold는 빈 LMCache DRAM과 새로운 DAOS namespace를 뜻한다.', '',
             '[공간 회수 대기 기록](SPACE_RECOVERY_KO.md). 사전 검증388개, OFF19,853개, ON19,855개 실험 KV 키만 삭제했다. '
             'KV payload 별도 백업은 없으며, 다른 namespace 키54,225개와 모든 로그·그래프는 보존했다.', '']
    report.write_text(original+'\n'.join(lines))


if __name__ == '__main__':
    main()
