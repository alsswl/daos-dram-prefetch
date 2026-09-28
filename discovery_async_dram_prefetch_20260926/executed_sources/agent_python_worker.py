"""Persistent Python tool in a read-only, network-isolated container."""
import contextlib
import io
import json
import sys
from langchain_experimental.tools.python.tool import PythonAstREPLTool

tool = PythonAstREPLTool()
print(json.dumps({'ready': True}), flush=True)
for line in sys.stdin:
    captured = io.StringIO()
    try:
        code = json.loads(line)['code']
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            value = tool.invoke(code)
        output = captured.getvalue() + str(value)
    except BaseException as exc:
        output = captured.getvalue() + f'{type(exc).__name__}: {exc}'
    if len(output) > 65536:
        output = output[:65536] + '\n[Tool output truncated at 65536 characters]'
    print(json.dumps({'output': output}), flush=True)
