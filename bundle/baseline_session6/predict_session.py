"""Восстановить конкурсный CSV выбранной итерации 6 без переобучения.

python predict_session.py --data /path/to/parquet/directory --output submission.csv
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd


def predict(bundle, data, cache):
    bundle = Path(bundle)
    manifest = json.loads((bundle/'manifest.json').read_text())
    reference = bundle/'reference_0328.csv'
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == manifest['reference_sha256']
    sys.path.insert(0, str(bundle/'code'))
    from predict_round100 import Predictor
    predictor = Predictor(bundle, data, reference, cache)
    shape = predictor.core(86)
    old = predictor.old
    weight = manifest['shape_weight']
    assert 0 <= weight <= .75
    out = np.empty(len(old))
    for _, indices in predictor.keys.groupby('timestamp').indices.items():
        assert len(indices) == 1000
        scaled = shape[indices]*old[indices].sum()/shape[indices].sum()
        out[indices] = (1-weight)*old[indices]+weight*scaled
        assert np.isclose(out[indices].sum(), old[indices].sum(), rtol=1e-12)
    details = predictor.keys[['timestamp', 'route_id']].copy()
    details['y_pred'] = out
    submission = predictor.test.merge(details, on=['timestamp', 'route_id'], validate='one_to_one', sort=False)[['id', 'y_pred']]
    assert len(submission) == 8000 and submission.id.is_unique
    assert np.array_equal(submission.id, predictor.test.id)
    assert np.isfinite(submission.y_pred).all() and submission.y_pred.ge(0).all()
    return submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path, default=Path('submission_session6.csv'))
    args = parser.parse_args()
    cache = args.cache or args.bundle/'inference_cache'
    submission = predict(args.bundle, args.data, cache)
    submission.to_csv(args.output, index=False)
    print(json.dumps({'rows': len(submission), 'sum': float(submission.y_pred.sum()),
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'file': str(args.output.resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
