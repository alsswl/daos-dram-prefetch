"""Verify split archives, safely extract to a fresh directory, verify file hashes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import threading

base=Path(__file__).resolve().parent
repo=base.parent
dest=Path(sys.argv[1]).resolve()
if dest.exists():raise SystemExit('Destination must not exist; use a fresh directory')
archives=json.loads((base/'archives.json').read_text())
for a in archives:
    combined=hashlib.sha256()
    for row in a['parts']:
        p=repo/row['path'];h=hashlib.sha256()
        with p.open('rb') as f:
            while data:=f.read(8*2**20):h.update(data);combined.update(data)
        assert h.hexdigest()==row['sha256'] and p.stat().st_size==row['bytes'],str(p)
    assert combined.hexdigest()==a['archive_sha256']
dest.mkdir(parents=True)
for a in archives:
    proc=subprocess.Popen(['zstd','-q','-d','-c'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    errors=[]
    def feed():
        try:
            for row in a['parts']:
                with (repo/row['path']).open('rb') as f:
                    while data:=f.read(8*2**20):proc.stdin.write(data)
        except BaseException as e:errors.append(e)
        finally:proc.stdin.close()
    t=threading.Thread(target=feed);t.start()
    with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
        for member in tar:
            # Restore absolute symlink text after regular-file verification;
            # never follow such links while extracting other archive entries.
            if member.issym():continue
            tar.extract(member,dest,filter='data')
    while proc.stdout.read(2**20):pass
    t.join();assert proc.wait()==0 and not errors
    rows=json.loads((base/(a['name']+'_manifest.json')).read_text())
    for row in rows:
        p=dest/row['path']
        if row['kind']=='file':
            assert p.stat().st_size==row['bytes']
            with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256'],row['path']
    print(a['name']+': file hashes verified')
# Create symlinks only after all regular content has been extracted and checked.
for a in archives:
    for row in json.loads((base/(a['name']+'_manifest.json')).read_text()):
        if row['kind']=='symlink':
            p=dest/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
            p.symlink_to(row['target'])
print('Verified restore completed at '+str(dest))
