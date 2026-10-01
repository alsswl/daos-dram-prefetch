"""Check that Kineto captures CUDA copies issued by a background thread."""
import argparse
from collections import Counter
import json
from pathlib import Path
import threading
import time
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    folder = parser.parse_args().output
    folder.mkdir(exist_ok=False)
    assert torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities()
    host = torch.ones(20*2**20, dtype=torch.uint8, pin_memory=True)
    device = torch.empty_like(host, device='cuda')
    matrix = torch.randn((2048, 2048), device='cuda', dtype=torch.float16)
    result = torch.empty_like(matrix)
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    ready, go = threading.Event(), threading.Event()
    def copy():
        ready.set()
        go.wait()
        with torch.cuda.stream(copy_stream):
            for _ in range(16):
                device.copy_(host, non_blocking=True)
        copy_stream.synchronize()
    worker = threading.Thread(target=copy)
    worker.start()
    ready.wait()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        start_ns = time.time_ns()
        go.set()
        with torch.cuda.stream(compute_stream):
            for _ in range(16):
                torch.mm(matrix, matrix, out=result)
        worker.join()
        torch.cuda.synchronize()
        end_ns = time.time_ns()
    profile.export_chrome_trace(str(folder/'trace.json'))
    trace = json.loads((folder/'trace.json').read_text())
    cats = Counter(e.get('cat') for e in trace['traceEvents'])
    copies = [e for e in trace['traceEvents'] if e.get('cat') == 'gpu_memcpy']
    kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel']
    assert len(copies) == 16 and kernels, 'Background copies or compute missing'
    assert len({e['args']['stream'] for e in copies}) == 1
    (folder/'result.json').write_text(json.dumps(dict(result='PASS', categories=dict(cats),
        background_copies=len(copies), kernels=len(kernels), start_ns=start_ns, end_ns=end_ns,
        copy_stream_handle=int(copy_stream.cuda_stream),
        sample_copy=copies[0], sample_kernel=kernels[0],
        trace_metadata={k:v for k,v in trace.items() if k != 'traceEvents'}), indent=2))
    print('PASS: captured background CUDA copies and GPU kernels')


if __name__ == '__main__':
    main()
