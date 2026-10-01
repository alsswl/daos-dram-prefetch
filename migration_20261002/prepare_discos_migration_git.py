"""Create an isolated, credential-filtered Git migration snapshot; never push."""
import collections
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

SRC=Path('/root/discos_minji')
DEST=Path('/home/discos_git_migration_20261002')
OLD=Path('/root/discos_minji_git_backup_20260928')
META=DEST/'migration_20261002'
SKIP={'.git','venv','.venv','agent-venv','__pycache__','.pytest_cache','.cache',
      'node_modules','discovery_tool_deps_20260926'}
BROWSE={'.py','.c','.h','.sh','.toml','.yaml','.yml','.md','.txt','.json','.csv','.svg','.png','.pdf','.ini','.cfg','.patch'}
spec=importlib.util.spec_from_file_location('old_backup',SRC/'prepare_git_backup.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
excluded=[]
MARKERS=(b'hf_',b'ghp_',b'gho_',b'ghu_',b'ghs_',b'ghr_',b'sk-',
         b'private key',b'akia',b'apikey',b'api_key',b'api-key',b'password',
         b'accesstoken',b'access_token',b'access-token')


def check_content(data,path):
    # All existing credential regexes require one of these literal markers.
    # A C-level byte prefilter avoids expensive regex scans of numeric traces;
    # candidate files still run the complete original checker unchanged.
    lower=data.lower()
    if any(marker in lower for marker in MARKERS):old.check_content(data,path)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def git(*args):
    return subprocess.check_output(['git','-C',str(DEST),*args],text=True).strip()


def inventory(roots):
    rows=[];seen=set()
    for root in roots:
        root=Path(root)
        if not root.exists():
            excluded.append(dict(path=str(root),reason='missing optional path'));continue
        paths=[]
        if root.is_dir():
            for d,dirs,files in os.walk(root,followlinks=False):
                for name in list(dirs):
                    p=Path(d)/name
                    if name in SKIP:
                        dirs.remove(name);excluded.append(dict(path=str(p),reason='environment/cache/git history'));continue
                    if p.is_symlink():paths.append(p);dirs.remove(name)
                paths.extend(Path(d)/f for f in files)
        else:paths=[root]
        for p in paths:
            if p in seen:continue
            seen.add(p)
            if old.sensitive_name(p):
                excluded.append(dict(path=str(p),reason='credential filename'));continue
            st=p.lstat()
            if p.is_symlink():
                rows.append(dict(source=str(p),path=str(p).lstrip('/'),kind='symlink',target=os.readlink(p),mtime_ns=st.st_mtime_ns));continue
            if not p.is_file():
                excluded.append(dict(path=str(p),reason='non-regular file'));continue
            # Scan the exact uncompressed bytes, including raw logs and datasets.
            data=p.read_bytes()
            try:check_content(data,str(p))
            except ValueError:
                excluded.append(dict(path=str(p),reason='possible credential content; source retained locally'));continue
            after=p.stat()
            if (st.st_size,st.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise RuntimeError('Source changed while scanning: '+str(p))
            rows.append(dict(source=str(p),path=str(p).lstrip('/'),kind='file',bytes=st.st_size,
                mtime_ns=st.st_mtime_ns,sha256=hashlib.sha256(data).hexdigest()))
            if len(rows)%5000==0:print(f'Scanned {len(rows)} files in component',flush=True)
    return sorted(rows,key=lambda r:r['path'])


def archive(name,rows):
    temp=DEST.parent/(DEST.name+'-'+name+'.tar.zst.partial')
    with temp.open('xb') as output:
        proc=subprocess.Popen(['zstd','-q','-3','-T4','--long=27','-c'],stdin=subprocess.PIPE,stdout=output)
        try:
            with tarfile.open(fileobj=proc.stdin,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for r in rows:
                    p=Path(r['source']);st=p.lstat()
                    if st.st_mtime_ns!=r['mtime_ns'] or (r['kind']=='file' and st.st_size!=r['bytes']):
                        raise RuntimeError('Source changed before archive: '+str(p))
                    tar.add(p,arcname=r['path'],recursive=False)
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('zstd compression failed')
        except BaseException:
            proc.kill();proc.wait();raise
    # Verify the archived contents against hashes collected before packing.
    expected={r['path']:r for r in rows};verified=set()
    proc=subprocess.Popen(['zstd','-q','-d','-c',str(temp)],stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
        for member in tar:
            row=expected[member.name]
            if member.issym():assert row['kind']=='symlink' and member.linkname==row['target']
            elif member.islnk():assert row['sha256']==expected[member.linkname]['sha256']
            else:
                assert member.isfile() and member.size==row['bytes']
                stream=tar.extractfile(member);digest=hashlib.file_digest(stream,'sha256').hexdigest()
                assert digest==row['sha256'],member.name
            verified.add(member.name)
    # Drain tar padding before waiting for the decompressor.
    while proc.stdout.read(2**20):pass
    assert proc.wait()==0 and verified==set(expected)
    out=META/'archives'/name;out.mkdir(parents=True)
    pieces=[]
    with temp.open('rb') as f:
        i=0
        while data:=f.read(64*2**20):
            p=out/f'{name}.tar.zst.part{i:04d}';p.write_bytes(data)
            pieces.append(dict(path=str(p.relative_to(DEST)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));i+=1
    result=dict(name=name,files=len(rows),uncompressed_bytes=sum(r.get('bytes',0) for r in rows),
        archive_bytes=temp.stat().st_size,archive_sha256=hashlib.file_digest(temp.open('rb'),'sha256').hexdigest(),
        parts=pieces,verified_file_hashes=True)
    temp.unlink()  # Only this newly generated intermediate; verified parts remain.
    print(f'{name}: verified {len(rows)} files, archive {result["archive_bytes"]/2**20:.1f} MiB',flush=True)
    return result


def main():
    os.umask(0o077)
    assert not DEST.exists()
    assert shutil.disk_usage(DEST.parent).free>80*2**30
    subprocess.run(['git','clone','--no-checkout','--no-hardlinks',str(OLD),str(DEST)],check=True)
    git('remote','set-url','origin','https://github.com/alsswl/daos-dram-prefetch.git')
    git('branch','migration-backup-20261002',git('rev-parse','HEAD'))
    git('symbolic-ref','HEAD','refs/heads/migration-backup-20261002')
    git('config','user.name','Local Backup');git('config','user.email','backup@localhost')
    META.mkdir()
    project=inventory([SRC])
    runtime_roots=['/root/discos-daos-src','/root/discos-gdrcopy-src','/root/discos',
        '/opt/daos-gds-gpu','/opt/ofi-cuda','/opt/discos-daos-gdr','/opt/discos-gdrcopy',
        '/etc/daos/daos_agent.yml','/root/discos-python','/tmp/dkey_group_plot_deps',
        '/usr/lib64/libgdrapi.so','/usr/lib64/libgdrapi.so.2','/usr/lib64/libgdrapi.so.2.6',
        '/etc/systemd/system/daos_agent.service',
        str(SRC/'venv/lib/python3.12/site-packages/lmcache'),
        str(SRC/'venv/lib/python3.12/site-packages/lmcache-0.5.2.dist-info'),
        str(SRC/'venv/lib/python3.12/site-packages/vllm'),
        str(SRC/'venv/lib/python3.12/site-packages/vllm-0.25.1.dist-info')]
    runtime_roots.extend(str(p) for p in Path('/root').glob('discos-*') if p.is_file() and p.suffix in {'.sh','.md','.patch','.lock','.yml','.json'})
    runtime=inventory(runtime_roots)
    for row in project:
        p=Path(row['source'])
        if row['kind']=='file' and row['bytes']<=10*2**20 and (p.suffix in BROWSE or p.name in {'Makefile','LICENSE','Dockerfile'}):
            target=DEST/p.relative_to(SRC);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
            assert hashlib.file_digest(target.open('rb'),'sha256').hexdigest()==row['sha256']
    # Everything approved, including large files/binaries/raw logs, goes into
    # split archives. The browsable tree is a convenience, not the full restore.
    for name,rows in [('project',project),('runtime',runtime)]:
        dump(META/(name+'_manifest.json'),rows)
    dump(META/'excluded.json',excluded)
    packages={}
    for name in ('venv','agent-venv'):
        python=SRC/name/'bin/python3'
        raw=subprocess.check_output([str(python),'-c','import importlib.metadata as m,json,sys; print(json.dumps(dict(python=sys.version,packages=sorted((d.metadata["Name"],d.version) for d in m.distributions() if d.metadata["Name"]))))'],text=True)
        packages[name]=json.loads(raw)
        (META/(name+'_versions.txt')).write_text('\n'.join(n+'=='+v for n,v in packages[name]['packages'])+'\n')
    dump(META/'python_packages.json',packages)
    (META/'os-release.txt').write_text(Path('/etc/os-release').read_text())
    (META/'rpm_versions.txt').write_bytes(subprocess.check_output(['rpm','-qa']))
    (META/'uname.txt').write_bytes(subprocess.check_output(['uname','-a']))
    shutil.copy2(__file__,META/'prepare_discos_migration_git.py')
    (DEST/'.gitignore').write_text('venv/\nagent-venv/\n__pycache__/\n*.pyc\n.env\n.env.*\n*.pem\n*.key\n**/api_config.json\n')
    (META/'RESTORE_KO.txt').write_text('''discos_minji 서버 이전 백업 (2026-10-02)

Git 루트에는 탐색용 코드/설정/10MiB 이하 결과를 배치했다.
전체 선별 파일은 project/runtime 분할 압축본에 있으며, 원본 로그와 큰 JSON도 포함한다.
excluded.json에 가상환경/캐시/인증정보 등 제외 경로를 정확히 기록했다.
credential content 판정 파일은 자동 업로드에서 제외했으며 원본 서버에 남아 있다.

복구 (빈 디렉터리에서 먼저 확인):
  python3 migration_20261002/restore.py /충분한공간/recovered
파일별 SHA-256과 압축 조각을 검사한 뒤 recovered 아래에 원래 절대경로 구조를 복원한다.
root/discos_minji, opt, etc/daos 등을 검토한 뒤 실제 경로로 옮긴다.
기존 파일 위에 바로 덮어쓰지 않는다. 심볼릭 링크는 원래 경로를 가리킬 수 있다.

Python 패키지 버전은 python_packages.json 및 *_versions.txt에 있다.
가상환경 자체는 포함하지 않지만 현재 lmcache/vllm 패키지는 runtime에 보존했다.
OS/CUDA/드라이버/RDMA 구성이 다른 서버에서는 네이티브 바이너리 호환을 검증하고 필요 시 재빌드한다.
DAOS/GDR 빌드 소스·패치·스크립트와 /opt 클라이언트 라이브러리를 runtime에 포함했다.
모델 가중치(/home/hf/hf_cache)는 포함하지 않는다. Qwen 모델을 다시 내려받거나 별도 전송한다.
DAOS 서버 안의 KV 데이터는 로컬 파일 백업에 포함되지 않는다.
새 서버에서 DAOS endpoint, pool/container 접근, agent 설정, NIC/GPU 장치와 경로를 확인한다.
최근 실험은 cxs_async_drop2g_c8s10_20261002이며 결과와 원본 trace가 project에 보존된다.
''')
    shutil.copy2('/root/restore_discos_migration.py',META/'restore.py')
    git('add','--all');git('commit','-m','Snapshot current discos_minji code, results and migration manifests')
    baseline=git('rev-parse','HEAD');commits=[baseline]
    archives=[]
    for name,rows in [('project',project),('runtime',runtime)]:
        result=archive(name,rows);archives.append(result)
        pieces=[p['path'] for p in result['parts']]
        for at in range(0,len(pieces),6):
            git('add','--',*pieces[at:at+6]);git('commit','-m',f'Preserve {name} archive parts {at} through {min(at+6,len(pieces))-1}')
            commits.append(git('rev-parse','HEAD'))
    dump(META/'archives.json',archives)
    dump(META/'backup_status.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(SRC),project_files=len(project),runtime_files=len(runtime),
        excluded_by_reason=dict(collections.Counter(r['reason'] for r in excluded)),
        archives_verified=True,upload_completed=False))
    git('add','--all');git('commit','-m','Record verified migration archive hashes and restoration instructions')
    commits.append(git('rev-parse','HEAD'))
    assert not git('status','--porcelain')
    git('fsck','--full')
    dump(DEST.parent/(DEST.name+'-upload-plan.json'),dict(repository=str(DEST),remote=git('remote','get-url','origin'),
        branch='migration-backup-20261002',commits_in_push_order=commits,head=commits[-1],uploaded=False))
    print(json.dumps(dict(repository=str(DEST),commits=len(commits),head=commits[-1],archives=archives,
        credential_exclusions=sum(r['reason'].startswith('possible credential') for r in excluded)),indent=2),flush=True)


if __name__=='__main__':main()
