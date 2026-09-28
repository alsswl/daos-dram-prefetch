"""Run the supplied DiscoveryBench ReAct agent, isolating only its Python tool."""
import argparse
import json
import os
import re
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
DISCOVERY = ROOT / 'discoverybench'
sys.path.insert(0, str(DISCOVERY))
from langchain_core.callbacks import BaseCallbackHandler
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from agents import react_utils, react_agent
from utils.autonomous_single_agent import run_autonomous_single_agent_discoverybench


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class FinalAliasParser(react_utils.CustomOutputParser):
    """Accept the FINAL_ANSWER spelling used in the original prompt example."""
    def parse(self, llm_output):
        match = re.search(r'(?im)^\s*FINAL_ANSWER\s*:\s*', llm_output)
        if match:
            return react_utils.AgentFinish(return_values={'output': llm_output[match.end():].strip()},
                                           log=llm_output)
        return super().parse(llm_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8017)
    parser.add_argument('--metadata', type=Path,
        default=DISCOVERY / 'discoverybench/synth/train/adventure-travel_0_0/metadata_0.json')
    parser.add_argument('--dataset-type', choices=['real', 'synth'], default='synth')
    parser.add_argument('--max-iterations', type=int, default=25)
    parser.add_argument('--stream', action='store_true', help='Record client first-text latency')
    parser.add_argument('--disable-thinking', action='store_true')
    parser.add_argument('--tool-site', type=Path, help='Optional experiment-only extra Python packages')
    parser.add_argument('--accept-final-alias', action='store_true')
    a = parser.parse_args()
    if not a.metadata.resolve().is_relative_to((DISCOVERY/'discoverybench').resolve()):
        parser.error('Metadata must belong to the local DiscoveryBench dataset')
    folder = a.folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    work = folder / 'work'; work.mkdir()
    task = a.metadata.resolve().parent
    metadata = json.loads(a.metadata.read_text())
    query = metadata['queries'][0]
    if isinstance(query, list):
        query = query[0]
    datasets = ['/data/' + d['name'] for d in metadata['datasets']]
    for d in metadata['datasets']:
        source = (task/d['name']).resolve()
        if not source.is_relative_to(task) or not source.is_file():
            parser.error(f'Invalid or unavailable task dataset: {d["name"]}')
    save(folder / 'api.json', {'openai': 'dummy'})
    name = 'minji-agent-' + uuid.uuid4().hex[:12]
    command = ['podman', 'run', '--rm', '-i', '--name', name, '--network', 'none',
               '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--security-opt', 'label=disable', '--pids-limit', '128', '--memory', '4g',
               '--cpus', '4', '--tmpfs', '/tmp:rw,size=256m',
               '-e', 'PYTHONPATH=/opt/site', '-e', 'OPENBLAS_NUM_THREADS=1',
               '-e', 'OMP_NUM_THREADS=1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
               '-v', f'{ROOT}/agent-venv/lib/python3.12/site-packages:/opt/site:ro',
               '-v', f'{ROOT}/agent_python_worker.py:/worker.py:ro',
               '-v', f'{task}:/data:ro', '-v', f'{work}:/work:rw', '-w', '/work',
               'docker.io/library/python:3.12-slim', 'python', '-u', '/worker.py']
    if a.tool_site:
        tool_site = a.tool_site.resolve()
        if not tool_site.is_dir() or not tool_site.is_relative_to(ROOT):
            parser.error('tool-site must be an existing experiment directory under discos_minji')
        command[command.index('PYTHONPATH=/opt/site')] = 'PYTHONPATH=/opt/extra:/opt/site'
        idx = command.index('docker.io/library/python:3.12-slim')
        command[idx:idx] = ['-v', f'{tool_site}:/opt/extra:ro']
    save(folder / 'sandbox_command.json', command)
    errors = (folder / 'sandbox_stderr.log').open('w')
    worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=errors, text=True, bufsize=1)
    selector = selectors.DefaultSelector(); selector.register(worker.stdout, selectors.EVENT_READ)
    calls, tools, finishes = {}, [], []
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Experiment coordinator requested shutdown')
    signal.signal(signal.SIGTERM, interrupted)

    def receive():
        if not selector.select(timeout=60):
            raise TimeoutError('Python tool exceeded 60 seconds')
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError('Python sandbox exited unexpectedly')
        return json.loads(line)

    class SandboxTool(PythonAstREPLTool):
        def _run(self, query, run_manager=None):
            started = time.perf_counter()
            worker.stdin.write(json.dumps({'code': query}) + '\n'); worker.stdin.flush()
            output = receive()['output']
            tools.append({'code': query, 'output': output,
                          'elapsed_seconds': time.perf_counter() - started})
            save(folder / 'tool_calls.json', tools)
            return output

    class Recorder(BaseCallbackHandler):
        def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
            calls[str(run_id)] = {'started': time.perf_counter(), 'start_ns': time.time_ns(), 'prompts': prompts}
            save(folder / 'llm_calls.json', calls)

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            calls[str(run_id)] = {'started': time.perf_counter(), 'start_ns': time.time_ns(),
                                  'messages': [[m.content for m in batch] for batch in messages]}
            save(folder / 'llm_calls.json', calls)

        def on_llm_new_token(self, token, *, run_id, **kwargs):
            row = calls.setdefault(str(run_id), {})
            if token and 'first_text_ns' not in row:
                row['first_text_ns'] = time.time_ns()
                row['ttft_ms'] = (time.perf_counter() - row.get('started', time.perf_counter())) * 1000

        def on_llm_end(self, response, *, run_id, **kwargs):
            row = calls.setdefault(str(run_id), {})
            row['elapsed_seconds'] = time.perf_counter() - row.pop('started', time.perf_counter())
            row['end_ns'] = time.time_ns()
            row['llm_output'] = response.llm_output
            row['generations'] = [[{'text': g.text, 'info': g.generation_info} for g in batch]
                                  for batch in response.generations]
            save(folder / 'llm_calls.json', calls)

        def on_llm_error(self, error, *, run_id, **kwargs):
            calls.setdefault(str(run_id), {})['error'] = str(error)
            calls[str(run_id)]['end_ns'] = time.time_ns()
            save(folder / 'llm_calls.json', calls)

        def on_agent_finish(self, finish, **kwargs):
            finishes.append(finish.return_values)
            save(folder / 'final_answers.json', finishes)

    recorder = Recorder()
    original_create = react_agent.create_agent
    def create_with_recording(**kwargs):
        kwargs['handlers'] = [*kwargs['handlers'], recorder]
        return original_create(**kwargs)

    class LocalAgent(react_agent.ReactAgent):
        def get_model(self, **kwargs):
            options = {'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}} if a.disable_thinking else {}
            return ChatOpenAI(model='comparison-model', api_key='dummy',
                              base_url=f'http://127.0.0.1:{a.port}/v1',
                              temperature=0, max_tokens=2048, request_timeout=180,
                              max_retries=0, callbacks=[recorder], streaming=a.stream, model_kwargs=options)

    started = None
    try:
        if receive() != {'ready': True}:
            raise RuntimeError('Python sandbox not ready')
        react_utils.PythonAstREPLTool = SandboxTool
        if a.accept_final_alias:
            react_utils.CustomOutputParser = FinalAliasParser
        react_agent.create_agent = create_with_recording
        agent = LocalAgent(model_config=str(DISCOVERY / 'config/model_config.json'),
                           api_config=str(folder / 'api.json'), model_name='qwen3-14b',
                           log_file=str(folder / 'agent.log'), max_iterations=a.max_iterations)
        started = time.perf_counter()
        run_autonomous_single_agent_discoverybench(agent=agent, datasets=datasets,
            metadata=metadata, nl_query=query['question'], provide_domain_knowledge=False,
            provide_workflow_tags=False, dataset_type=a.dataset_type)
        elapsed = time.perf_counter() - started
        final = str(finishes[-1].get('output', '')) if finishes else ''
        complete = bool(final and tools and calls and not final.startswith('Agent stopped'))
        save(folder / 'status.json', {'status': 'complete' if complete else 'incomplete',
             'workflow_seconds': elapsed, 'llm_calls': len(calls), 'python_calls': len(tools),
             'final_answer': final, 'grading': 'not run, as in supplied document',
             'query': query['question'], 'max_iterations': a.max_iterations, 'temperature': 0,
             'max_output_tokens_per_call': 2048, 'metadata_source': str(a.metadata.resolve()),
             'dataset_type': a.dataset_type, 'streaming': a.stream,
             'disable_thinking': a.disable_thinking, 'tool_site': str(a.tool_site) if a.tool_site else None,
             'accept_final_alias': a.accept_final_alias})
        if not complete:
            raise RuntimeError('Agent did not reach a final answer after executing Python')
        print(f'COMPLETE: {len(calls)} model calls, {len(tools)} Python calls, {elapsed:.2f}s', flush=True)
    except BaseException as exc:
        if not (folder / 'status.json').exists():
            save(folder / 'status.json', {'status': 'failed', 'error': repr(exc),
                 'llm_calls': len(calls), 'python_calls': len(tools)})
        raise
    finally:
        try:
            worker.stdin.close()
        except BrokenPipeError:
            pass
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            subprocess.run(['podman', 'stop', '--time', '2', name], capture_output=True, timeout=15)
            worker.wait(timeout=10)
        selector.close(); errors.close()


if __name__ == '__main__':
    main()
