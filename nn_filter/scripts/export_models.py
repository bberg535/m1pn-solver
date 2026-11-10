#!/usr/bin/env python3
"""Export the authors' checkpoints to a small, audited inference-only format.

No upstream Python is executed. Only explicitly allowed tensor/module classes
are decoded by torch.load(weights_only=True).
"""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class SimpleNN(nn.Module):
    pass


def load_checkpoint(path):
    allowed = [(SimpleNN, 'funcs_common.SimpleNN'), set, nn.Linear, nn.LayerNorm]
    with torch.serialization.safe_globals(allowed):
        model = torch.load(path, map_location='cpu', weights_only=True)
    return model


def forward(model, x):
    for i in range(1, 5):
        y = torch.tanh(getattr(model, f'hidden{i}')(x))
        x = getattr(model, f'bn{i+1}')(y if i == 1 else y + x)
    return torch.relu(model.output(x))


def export(order):
    path = ROOT / f'models/original_N{order}.pth'
    model = load_checkpoint(path)
    model.eval()
    nf, nh = model.hidden1.in_features, model.hidden1.out_features
    assert nf == 2 * order + 4
    values = [f'NNFPN2D 1 {order} {nf} {nh}']
    def put(t):
        values.append(' '.join(format(v, '.17g') for v in t.detach().double().flatten().tolist()))
    for i in range(1, 5):
        layer, norm = getattr(model, f'hidden{i}'), getattr(model, f'bn{i+1}')
        put(layer.weight)
        put(layer.bias)
        values.append(format(norm.eps, '.17g'))
        put(norm.weight)
        put(norm.bias)
    put(model.output.weight)
    put(model.output.bias)
    target = ROOT / f'models/model_N{order}.txt'
    target.write_text('\n'.join(values) + '\n')
    # Fixtures from torch's independent Linear/LayerNorm implementation.
    gen = torch.Generator().manual_seed(1511 + order)
    x = torch.cat((torch.zeros(1, nf), torch.randn(23, nf, generator=gen)), dim=0).double()
    model.double()
    with torch.no_grad():
        y = forward(model, x).squeeze(1)
    fixture = ROOT / f'models/fixture_N{order}.txt'
    fixture.write_text(f'{len(x)} {nf}\n' + '\n'.join(
        ' '.join(format(v, '.17g') for v in list(a) + [b])
        for a, b in zip(x.tolist(), y.tolist())) + '\n')
    return {'order': order, 'input_features': nf, 'hidden_width': nh,
            'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'export_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}


if __name__ == '__main__':
    torch.set_num_threads(1)
    data = {'upstream': 'https://github.com/benplumridge/NN_FPN',
            'commit': 'dba16410a4bbb0749897b12b04119f4ca061f647',
            'torch_version': torch.__version__, 'models': [export(n) for n in (3, 5, 7, 9)]}
    (ROOT / 'models/manifest.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2))
