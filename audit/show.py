#!/usr/bin/env python3
"""Print selected JSONL records compactly: show.py file prefix [key,key,...]"""
import json, sys
path, prefix = sys.argv[1], sys.argv[2]
keys = sys.argv[3].split(',') if len(sys.argv) > 3 else None
for line in open(path):
    d = json.loads(line)
    if not d['test'].startswith(prefix):
        continue
    out = {}
    for k, v in d.items():
        if keys and k not in keys and k not in ('test', 'label', 'variant'):
            continue
        out[k] = float('%.4g' % v) if isinstance(v, float) else v
    print(out)
