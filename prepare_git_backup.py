#!/usr/bin/env python3
"""Create a NEW, credential-checked Git snapshot without touching live results.

No upload, deletion, model import, GPU call, DAOS call or background job.
Large raw logs/environments and running experiment trees are excluded.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

EXTENSIONS = {'.py','.c','.h','.sh','.toml','.yaml','.yml','.md','.txt',
              '.json','.csv','.svg','.png','.pdf','.ini','.cfg','.orig','.patch'}
SKIP_DIRS = {'.git','venv','.venv','agent-venv','__pycache__','.pytest_cache',
             'discovery_tool_deps_20260926','node_modules','.cache'}
UPSTREAMS = {'discoverybench', 'eqbench_longform_upstream_20260928'}
SENSITIVE_NAMES = {'api_config.json','credentials.json','credentials','token','token.json',
                   'id_rsa','id_ed25519','.netrc','.pypirc'}
SECRET_PATTERNS = [
    rb'\b(?:hf_|gh[pousr]_)[A-Za-z0-9]{20,}',
    rb'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}',
    rb'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----',
    rb'\bAKIA[0-9A-Z]{16}\b',
    rb'(?i)(?:api[_-]?key|password|access[_-]?token|hf_token)\s*[=:]\s*[\x22\x27][A-Za-z0-9_+/=-]{24,}[\x22\x27]',
]


def sensitive_name(path):
    return (path.name in SENSITIVE_NAMES or path.name.startswith('.env')
            or path.suffix.lower() in {'.pem','.key','.p12','.pfx'})


def check_content(data, path):
    # Error reports intentionally never include the matched credential itself.
    if any(re.search(pattern,data) for pattern in SECRET_PATTERNS):
        raise ValueError(f'Possible credential: {path}; no snapshot will be staged')


def dump(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def git(source, *args):
    return subprocess.check_output(['git','-C',str(source),*args])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-file-mib',type=int,default=10)
    a=p.parse_args(); source=a.source.resolve(); dest=a.output.resolve()
    assert source.is_dir() and not dest.exists(), 'Destination must be new'
    assert not dest.is_relative_to(source), 'Keep backups outside the live source tree'
    active=set()
    for status in source.glob('*/status.json'):
        try:
            if json.loads(status.read_text()).get('status')=='running': active.add(status.parent.name)
        except (ValueError,OSError):
            active.add(status.parent.name)  # A changing status is not a stable snapshot.
    excluded=[]; candidates=[]
    for directory, dirs, files in os.walk(source):
        relative=Path(directory).relative_to(source)
        for name in list(dirs):
            path=Path(directory)/name; rel=path.relative_to(source)
            if name in SKIP_DIRS or path.is_symlink() or (relative==Path('.') and name in UPSTREAMS|active):
                dirs.remove(name)
                excluded.append(dict(path=str(rel)+'/',reason='running' if name in active else 'environment/upstream/cache'))
        for name in files:
            path=Path(directory)/name; rel=path.relative_to(source)
            reason=None
            if path.is_symlink(): reason='symlink'
            elif sensitive_name(path): reason='credential filename'
            elif path.suffix.lower() not in EXTENSIONS and name not in {'Makefile','LICENSE','Dockerfile'}:
                reason='raw log/binary/non-allowlisted extension'
            elif path.stat().st_size>a.max_file_mib*2**20: reason='file size limit'
            if reason: excluded.append(dict(path=str(rel),reason=reason));continue
            candidates.append((path,rel))
    # Validate all selected content before creating the destination.
    records=[]
    for path,rel in candidates:
        before=path.stat();data=path.read_bytes();after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise RuntimeError(f'File changed while reading: {rel}; retry after experiment')
        if path.suffix.lower() not in {'.png','.pdf'}: check_content(data,rel)
        if path.suffix.lower()=='.json':
            try:json.loads(data)
            except ValueError:
                excluded.append(dict(path=str(rel),reason='invalid/incomplete JSON'));continue
        records.append(dict(path=str(rel),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    upstream=[];patches={}
    for name in sorted(UPSTREAMS):
        path=source/name
        if not path.exists():continue
        # Exclude the credential-bearing API config and temporary .bak copies.
        patch=git(path,'diff','HEAD','--binary','--','.',':(exclude)config/api_config.json')
        check_content(patch,name+' patch')
        patches[name]=patch
        remotes=git(path,'remote','get-url','origin').decode().strip()
        if re.search(r'https?://[^/]*@',remotes): raise ValueError('Credential-bearing remote URL')
        check_content(remotes.encode(),name+' remote')
        upstream.append(dict(directory=name,url=remotes,commit=git(path,'rev-parse','HEAD').decode().strip(),
            patch='backup_meta/'+name+'.patch' if patch else None,
            status=git(path,'status','--porcelain').decode(),
            excluded=['config/api_config.json','untracked files (including .bak)']))
    dest.mkdir(parents=True)
    for r in records:
        src=source/r['path']; target=dest/r['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=r['sha256']:
            raise RuntimeError(f"Source changed before copying: {r['path']}; snapshot not staged")
    meta=dest/'backup_meta';meta.mkdir(exist_ok=True)
    for name,patch in patches.items():
        if patch:(meta/(name+'.patch')).write_bytes(patch)
    dump(meta/'upstreams.json',upstream)
    # Read package versions only: no CUDA/PyTorch imports or pip resolution.
    envs={}
    for env in ('venv','agent-venv'):
        python=source/env/'bin/python3'
        if python.exists():
            raw=subprocess.check_output([str(python),'-c',
                'import importlib.metadata as m,json,sys; print(json.dumps(dict(python=sys.version,packages=sorted((d.metadata["Name"],d.version) for d in m.distributions() if d.metadata["Name"]))))'])
            envs[env]=json.loads(raw)
    dump(meta/'python_packages.json',envs)
    dump(meta/'manifest.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(source),files=records,total_bytes=sum(r['bytes'] for r in records),
        excluded=excluded,active_experiments_excluded=sorted(active),max_file_mib=a.max_file_mib,
        not_full_environment_backup=True,remote_upload=False))
    (dest/'.gitignore').write_text('''# Keep credentials and runtime payloads out of later commits.
venv/
.venv/
agent-venv/
__pycache__/
.pytest_cache/
*.pyc
*.so
*.o
*.a
*.log
*.jsonl
*.safetensors
*.pt
*.pth
*.bin
.env
.env.*
*.pem
*.key
*.p12
*.pfx
**/api_config.json
credentials.json
discoverybench/
eqbench_longform_upstream_20260928/
''')
    (dest/'GIT_BACKUP_KO.md').write_text('''# Git 업로드용 백업

실행 중인 discos_minji와 분리한 시점 스냅샷이다. 원본 파일은 수정/삭제하지 않았다.
코드·설정·문서·작은 JSON/CSV·그래프를 포함한다. 제외 항목과 SHA-256은
`backup_meta/manifest.json`에서 확인한다. 파일당 기본 10MiB 제한이 있다.

가상환경, 컴파일 바이너리, 모델, 원시 .log/.jsonl, 실행 중 실험 디렉터리는 제외했다.
DAOS 서버의 실제 KV 데이터 및 /opt 네이티브 라이브러리도 포함하지 않는다.
그러므로 전체 서버/원시 실험 데이터 백업이 아니며, 복구 후 바로 실행 가능하다는 뜻도 아니다.
스크립트의 /root/discos_minji, /opt 및 HF 캐시 경로는 새 환경에 맞춰 확인해야 한다.

`backup_meta/python_packages.json`은 설치 버전 목록이며 재현이 검증된 lock 파일은 아니다.
벤치마크 원본은 `backup_meta/upstreams.json`의 URL/commit으로 복원한다.
같은 commit에 해당 .patch를 적용한다. 비밀정보가 들어갈 수 있는 api_config.json과
원본 저장소의 untracked .bak는 제외했다. API 설정은 별도로 복원한다.
libdaosgdr.so는 설치된 GPU 지원 DAOS 스택을 준비한 뒤 Makefile로 다시 빌드한다.

## 업로드

초기 커밋은 자동 스냅샷임을 나타내는 Local Backup <backup@localhost> 명의다.
서버 경로·실험 내용·생성 답변이 들어 있으므로 비공개 저장소를 권장한다.
토큰 패턴 검사는 완전한 비밀정보/개인정보 검사를 대체하지 않으므로 공개 전 검토한다.
비밀정보를 URL에 넣지 말고 SSH 또는 Git 자격증명 관리자를 사용한다.

빈 원격 저장소를 준비한 뒤 이 디렉터리에서:

```bash
git remote add origin <저장소_URL>
git push -u origin main
```

원격에 기존 이력이 있으면 강제 push하지 말고 먼저 통합 방법을 확인한다.
현재 작업의 자동 동기화나 자동 업로드는 하지 않는다. 실행 중이던 실험이 끝난 뒤
원본의 prepare_git_backup.py --output <새_백업_절대경로>로 새 스냅샷을 만들 수 있다.
다음 스냅샷은 별도 Git 이력이므로 기존 원격을 덮어쓰지 말고 검토 후 기존 백업에 반영한다.
''')
    subprocess.run(['git','init','-b','main',str(dest)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(dest),'add','--all'],check=True)
    subprocess.run(['git','-C',str(dest),'-c','user.name=Local Backup','-c','user.email=backup@localhost',
                    'commit','-m','Backup discos_minji code and selected experiment artifacts'],check=True,stdout=subprocess.DEVNULL)
    assert not git(dest,'status','--porcelain')
    print(json.dumps(dict(directory=str(dest),files=len(records),bytes=sum(r['bytes'] for r in records),
        excluded_by_reason=dict(Counter(e['reason'] for e in excluded)),running_excluded=sorted(active),
        commit=git(dest,'rev-parse','HEAD').decode().strip(),uploaded=False),indent=2))


if __name__=='__main__':main()
