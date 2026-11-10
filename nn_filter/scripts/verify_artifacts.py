#!/usr/bin/env python3
"""Verify the recorded binary, models, parent source, results and raw fields."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=ROOT/'results/crossing_20261109')
    args=p.parse_args(); out=args.study
    manifest=json.loads((out/'manifest.json').read_text())
    expected={'protocol_sha256':ROOT/'protocol.json',
              'binary_sha256':ROOT/'tmp/build/m1pn_nn_filter',
              'model_manifest_sha256':ROOT/'models/manifest.json',
              'parent_manifest_sha256':ROOT/'parent_manifest.json',
              'results_sha256':out/'results.json'}
    for name,path in expected.items():assert sha(path)==manifest[name],str(path)
    models=json.loads((ROOT/'models/manifest.json').read_text())
    upstream=json.loads((ROOT/'upstream/manifest.json').read_text())
    for model in models['models']:
        order=model['order']
        for key,name in [('checkpoint_sha256',f'original_N{order}.pth'),
                         ('export_sha256',f'model_N{order}.txt')]:
            assert sha(ROOT/'models'/name)==model[key],name
        checkpoint=(ROOT/'models'/f'original_N{order}.pth').read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(checkpoint)).encode()+b'\0'+checkpoint).hexdigest()
        original=next(f for f in upstream['files'] if f['path']==f'2D/trained_models/model_N{order}.pth')
        assert original['git_blob_sha1']==blob
    parent=json.loads((ROOT/'parent_manifest.json').read_text())
    for name,digest in parent['sha256'].items():assert sha(ROOT.parent/name)==digest,name
    rows=json.loads((out/'results.json').read_text())
    for row in rows:
        run=ROOT/row['run']
        verification=json.loads((run/'filter_verification.json').read_text())
        assert verification['status']=='passed'
        for name,digest in verification['sha256'].items():assert sha(run/name)==digest,str(run/name)
    files=[p for folder in ('src','scripts','tests') for p in (ROOT/folder).glob('*')
           if p.is_file() and p.suffix in ('.cpp','.hpp','.py','.sh')]
    files += [ROOT/'CMakeLists.txt',ROOT/'README.md',ROOT/'protocol.json']
    artifacts=[p for p in out.iterdir() if p.is_file() and p.name!='final_audit.json']
    audit={'status':'passed','measured_runs':len(rows),
           'parent_files_unchanged':len(parent['sha256']),
           'original_checkpoints_match_github_blobs':True,
           'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},
           'artifact_sha256':{p.name:sha(p) for p in artifacts}}
    (out/'final_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if not k.endswith('sha256')},indent=2))


if __name__=='__main__':main()
