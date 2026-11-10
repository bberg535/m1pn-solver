#!/usr/bin/env python3
"""Compare native preprocessing with the exact two audited upstream functions."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
tree=ast.parse((ROOT/'upstream/funcs_common.py').read_text())
definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
             and n.name in ('preprocess_features','NN_normalization')]
assert len(definitions)==2
scope={'torch':torch}
# Only these two numerical functions; no imports, model deserialization, solver,
# training loop, filesystem/network calls, or other upstream top-level code.
exec(compile(ast.Module(body=definitions,type_ignores=[]),'<audited upstream preprocessing>','exec'),scope)
with tempfile.TemporaryDirectory(prefix='nn_preprocess_') as tmp:
    path=Path(tmp)/'features.txt'
    subprocess.run([sys.argv[1],str(path)],check=True,stdout=subprocess.DEVNULL)
    with path.open() as f:
        nodes,order=map(int,f.readline().split()); data=np.loadtxt(f)
    q=(order+1)*(order+2)//2
    tensors=[torch.tensor(data[:,j*q:(j+1)*q]).reshape(1,1,nodes,q) for j in range(3)]
    scattering=torch.tensor(data[:,3*q]).reshape(1,1,nodes)
    source=torch.tensor(data[:,3*q+1]).reshape(1,1,nodes)
    actual=scope['preprocess_features'](order,*tensors,scattering,source,
             {'num_x':nodes,'num_y':1,'batch_size':1,'device':'cpu'})
    # Upstream allocates its norm buffers in default dtype; use float64 for
    # comparison with native float64, preserving the exact original formulas.
    torch.set_default_dtype(torch.float64)
    actual=scope['preprocess_features'](order,*tensors,scattering,source,
             {'num_x':nodes,'num_y':1,'batch_size':1,'device':'cpu'})
    expected=data[:,3*q+2:]
    error=float(np.max(np.abs(actual.numpy().reshape(nodes,-1)-expected)))
    assert error<1e-11,error
print(json.dumps({'status':'passed','upstream_preprocessing_max_error':error}))
