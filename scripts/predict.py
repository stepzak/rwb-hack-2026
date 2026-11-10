"""Полный финальный inference без обучения; исходный submission не перезаписывается."""
from pathlib import Path
import argparse, hashlib, json, os, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from artifacts import verify

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(os.environ.get('RWB_DATA_DIR',str(ROOT/'data'))))
    p.add_argument('--device',default=os.environ.get('RWB_NEURAL_DEVICE','cpu'),choices=['cpu','cuda'])
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/submission_reproduced.csv');a=p.parse_args()
    print(verify())
    expected=json.loads((ROOT/'bundle/input_hashes.json').read_text())
    for name,digest in expected.items():
        with (a.data/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=digest:raise ValueError(f'Input SHA-256 mismatch: {name}')
    sys.path.insert(0,str(ROOT/'bundle'))
    from predict_final30 import Runtime
    r=Runtime(ROOT/'bundle',a.data,ROOT/'artifacts/.cache',neural_device=a.device)
    result=r.predict(39);a.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False)
    print(json.dumps({'rows':len(result),'sum':float(result.y_pred.sum())}))
