#!/usr/bin/env python3
"""Collect existing occupancy SVG panels without changing their measurements."""
import argparse
import copy
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root.resolve()
    assert json.loads((root / 'status.json').read_text())['status'] == 'completed'
    rows = json.loads((root / 'summary.json').read_text())
    phases = list(dict.fromkeys(row['phase'] for row in rows))
    indexed = {(row['case'], row['phase']): row for row in rows}
    # This layout is explicitly for the completed 512/8 GiB C16 comparison.
    height = 180 + 320 * len(phases) + 95
    svg = ET.Element(f'{{{NS}}}svg', width='2000', height=str(height),
                     viewBox=f'0 0 2000 {height}')
    ET.SubElement(svg, f'{{{NS}}}rect', width='2000', height=str(height), fill='white')

    def text(x, y, value, size=22, color='#172b4d', bold=False):
        node = ET.SubElement(svg, f'{{{NS}}}text', x=str(x), y=str(y), fill=color,
                             attrib={'font-family': 'DejaVu Sans, sans-serif',
                                     'font-size': str(size),
                                     'font-weight': 'bold' if bold else 'normal'})
        node.text = value

    text(40, 45, 'GPU staging occupancy | ShareGPT cold + 4 warm replays', 32, bold=True)
    text(40, 84, 'Qwen3-14B | DRAM 512 GiB | staging 8 GiB | concurrency 16 | 1,684 requests per phase', 23)
    text(40, 118, '100% = 8 GiB. Blue: sampled maximum per 2-second bin. Green: sampled mean.', 22)
    text(55, 163, 'DRAM PREFETCH OFF', 27, bold=True)
    text(1055, 163, 'DRAM PREFETCH ON', 27, bold=True)
    for i, phase in enumerate(phases):
        for j, mode in enumerate(('off', 'on')):
            case = f'c16_{mode}'
            row = indexed[case, phase]
            source = root / case / phase / 'staging_hits.svg'
            panel = copy.deepcopy(ET.parse(source).getroot())
            # Existing SVG: occupancy panel ends at y=262; tier-hit panel starts at 308.
            # A clipped nested SVG preserves the original data and time axis exactly.
            panel.attrib.update(x=str(j * 1000), y=str(180 + i * 320),
                                width='1000', height='280',
                                viewBox='0 0 1000 280', overflow='hidden')
            svg.append(panel)
            text(j * 1000 + 65, 180 + i * 320 + 300,
                 f'Event peak: {row["peak_staging_gib"]:.2f} GiB '
                 f'({row["peak_staging_gib"] / 8 * 100:.1f}%)  |  '
                 f'Sampled mean: {row["mean_sampled_staging_gib"] / 8 * 100:.2f}%', 19)
    bottom = 180 + 320 * len(phases)
    text(40, bottom + 25, 'X = seconds since each phase began (each panel has its own duration); Y = staging occupancy (%).', 21)
    text(40, bottom + 55, '20 ms samples can miss brief peaks. Event peaks above use allocation events, not the sampled line.', 21)
    text(40, bottom + 85, 'Includes read and write buffers; NOT GPU compute utilization. Warm phases retain the same live cache.', 21)
    output = root / 'STAGING_OCCUPANCY_ALL.svg'
    ET.ElementTree(svg).write(output, encoding='utf-8', xml_declaration=True)
    subprocess.run(['rsvg-convert', '-o', str(output.with_suffix('.png')), str(output)], check=True)
    print(output)
    print(output.with_suffix('.png'))


if __name__ == '__main__':
    main()
