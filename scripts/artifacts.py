"""Проверка или загрузка доверенного архива весов из Release этого проекта."""
from pathlib import Path
import argparse, hashlib, json, shutil, tarfile, urllib.request

ROOT=Path(__file__).resolve().parents[1]

def verify():
    manifest=json.loads((ROOT/'reports/artifact_manifest.json').read_text())
    for entry in manifest['files']:
        path=ROOT/entry['path']
        if not path.is_file():raise FileNotFoundError(f"Missing artifact: {entry['path']}. Run scripts/artifacts.py fetch.")
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=entry['sha256']:raise ValueError(f"SHA-256 mismatch: {entry['path']}")
    return {'verified_files':len(manifest['files']),'bytes':sum(x['size'] for x in manifest['files'])}

def fetch(archive=None):
    manifest=json.loads((ROOT/'reports/artifact_manifest.json').read_text())
    target=ROOT/'artifacts'/manifest['archive'];target.parent.mkdir(parents=True,exist_ok=True)
    if archive is None:
        url=manifest.get('download_url')
        if not url:raise ValueError('Release URL is not configured; pass --archive /path/to/rwb-final-artifacts.tar.gz')
        request=urllib.request.Request(url,headers={'User-Agent':'rwb-final-project'})
        partial=target.with_suffix('.partial')
        with urllib.request.urlopen(request,timeout=120) as response,partial.open('wb') as dest:shutil.copyfileobj(response,dest)
        partial.replace(target)
    else:target=Path(archive)
    if manifest.get('archive_sha256'):
        with target.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=manifest['archive_sha256']:raise ValueError('Archive SHA-256 mismatch')
    allowed={x['path']:x for x in manifest['files']}
    with tarfile.open(target,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=len(allowed) or {m.name for m in members}!=set(allowed):raise ValueError('Unexpected archive contents')
        for member in members:
            if not member.isfile() or member.size!=allowed[member.name]['size']:raise ValueError('Unsafe archive entry')
            dest=ROOT/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as stream,dest.open('wb') as output:shutil.copyfileobj(stream,output)
    return verify()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','fetch']);p.add_argument('--archive',type=Path)
    a=p.parse_args();print(json.dumps(verify() if a.action=='verify' else fetch(a.archive),indent=2))
