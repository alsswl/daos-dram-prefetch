"""Run a GPU round trip and verify the native libraries actually mapped.

Launch with run_vllm.sh and DAOSGDS_TRANSPORT=dfs or object.
Each round-trip test removes only its own UUID-named test data.
"""
import os
from pathlib import Path
import runpy
import sys


def main():
    mode = os.environ['DAOSGDS_TRANSPORT']
    if mode not in {'dfs', 'object'}:
        raise ValueError(mode)
    script = Path(__file__).with_name(f'{mode}_gpu_roundtrip.py')
    sys.argv = [str(script), '--size-mib', '64']
    try:
        runpy.run_path(str(script), run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    paths = sorted({line.split()[-1] for line in
                    Path('/proc/self/maps').read_text().splitlines()
                    if '/' in line})
    expected = {
        'libdaos.so': '/opt/daos-gds-gpu/',
        'libmercury.so': '/opt/daos-gds-gpu/',
        'libfabric.so': '/opt/ofi-cuda/',
    }
    if mode == 'dfs':
        expected['libdfs.so'] = '/opt/daos-gds-gpu/'
    else:
        expected['libdaosgdr.so'] = '/root/discos_minji/'
    for library, prefix in expected.items():
        matches = [p for p in paths if Path(p).name.startswith(library)]
        assert matches and all(p.startswith(prefix) for p in matches), (
            library, matches, prefix
        )
        print(f'LOADED {library}: {matches}')
    print(f'PASS common native stack: {mode}')


if __name__ == '__main__':
    main()
