"""Создать архив только необходимых двоичных артефактов, без исходных parquet."""
from pathlib import Path
import json, sys, tarfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
from artifacts import ROOT,verify
if __name__=='__main__':
    print(verify(),flush=True)
    m=json.loads((ROOT/'reports/artifact_manifest.json').read_text())
    target=ROOT/'artifacts'/m['archive'];target.parent.mkdir(exist_ok=True)
    with tarfile.open(target,'w:gz',compresslevel=1) as tar:
        for f in m['files']:tar.add(ROOT/f['path'],arcname=f['path'],recursive=False)
    print(json.dumps({'archive':target.name,'bytes':target.stat().st_size}))
