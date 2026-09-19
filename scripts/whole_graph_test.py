#!/usr/bin/env python3
"""Create an equivalent candidate for end-to-end kern test without scratch snapshots.

Rename operations in driven programs to form one comparison span per program,
and give workspace buffers candidate-local names. Logits, public inputs/outputs,
weights, and persistent states retain their identities. The reference, kernels,
launch arguments, tensor shapes, workload, and test tolerances do not change.
"""
import argparse
import copy
import json
from pathlib import Path


def rename_refs(value, names):
    if isinstance(value, list):
        return [rename_refs(v, names) for v in value]
    if not isinstance(value, dict):
        return value
    return {k: names.get(v, v) if k in {'buf', 'of', 'index_into'} and isinstance(v, str)
            else rename_refs(v, names) for k, v in value.items()}


def whole_graph(manifest):
    m = copy.deepcopy(manifest)
    ops = {name: 'whole_graph_' + name for name in m['ops']}
    assert not set(ops.values()) & m['ops'].keys()
    m['ops'].update({ops[k]: v for k, v in manifest['ops'].items()})
    for program in m['programs'].values():
        if not program.get('once'):
            for call in program['calls']:
                call['op'] = ops[call['op']]
    buffers = {k: 'candidate_' + k for k, v in m['buffers'].items()
               if v['kind'] == 'workspace' and not k.startswith('logits')}
    assert not set(buffers.values()) & m['buffers'].keys()
    m = rename_refs(m, buffers)
    m['buffers'] = {buffers.get(k, k): v for k, v in m['buffers'].items()}

    reverse = {v: k for k, v in buffers.items()}
    restored = rename_refs(m, reverse)
    restored['buffers'] = {reverse.get(k, k): v for k, v in restored['buffers'].items()}
    restored['ops'] = {k: restored['ops'][k] for k in manifest['ops']}
    reverse_ops = {v: k for k, v in ops.items()}
    for program in restored['programs'].values():
        for call in program['calls']:
            call['op'] = reverse_ops.get(call['op'], call['op'])
    assert restored == manifest, 'Test candidate must differ only in names'
    used = {c['op'] for p in m['programs'].values() for c in p['calls']}
    m['ops'] = {k: v for k, v in m['ops'].items() if k in used}
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(whole_graph(json.loads(args.manifest.read_text())), indent=2)+'\n')


if __name__ == '__main__':
    main()
