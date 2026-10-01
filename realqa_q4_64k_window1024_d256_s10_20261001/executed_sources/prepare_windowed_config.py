#!/usr/bin/env python3
"""Create isolated vLLM config for repeatable window-size experiments.

Example:
  ./venv/bin/python3 prepare_windowed_config.py --window-mib 256 --output /root/discos_minji/window256.yaml
Use --window-mib 0 for the SAME synchronous lookup/control path without windows.
All generated configs disable both payload prefetches. No benchmark is launched.
"""
import argparse
from pathlib import Path
import uuid
import yaml

ROOT=Path(__file__).resolve().parent


def make_config(window_mib,staging_gib=8,cpu_gib=256):
    if window_mib<0 or staging_gib<=0 or cpu_gib<=0 or window_mib>staging_gib*1024:
        raise ValueError('Window must be 0..staging MiB; staging and DRAM must be positive')
    cfg=yaml.safe_load((ROOT/'lmcache_config_daosgds_windowed.yaml').read_text())
    cfg['max_local_cpu_size']=cpu_gib
    namespace='minji-windowed-'+uuid.uuid4().hex
    cfg['extra_config'].update({'daosgds.retrieve_window_mib':window_mib,
        'daosgds.gpu_buffer_gb':staging_gib,'daosgds.object_namespace':namespace+':',
        'daosgds.root':'/'+namespace})
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--window-mib',type=int,required=True)
    p.add_argument('--staging-gib',type=int,default=8)
    p.add_argument('--dram-gib',type=int,default=256)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    cfg=make_config(a.window_mib,a.staging_gib,a.dram_gib)
    dest=a.output.resolve()
    if not dest.is_relative_to(ROOT):p.error('Output must be inside discos_minji')
    with dest.open('x') as f:
        yaml.safe_dump(cfg,f,sort_keys=False)
    print(f'Created {dest}; no GPU/DAOS action performed.')
    print('Set LMCACHE_CONFIG_FILE to this path and DAOS_GDS_STAGING_TRACE to a new log prefix before starting vLLM.')
    print('Both prefetches OFF; synchronous lookup. Restart the process after changing window size.')
