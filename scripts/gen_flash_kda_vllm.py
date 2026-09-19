#!/usr/bin/env python3
"""Replace FlashKDA launches with vLLM dev's fixed-length FP32-state ABI."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'flash_kda_vllm_d128'


def generate(manifest, cubin, vsplit=False):
    manifest = copy.deepcopy(manifest)
    symbols = subprocess.check_output(['readelf', '-sW', str(cubin)], text=True)
    symbols = {line.split()[-1] for line in symbols.splitlines()
               if 'FUNC' in line and '_flash_kda' in line}
    prepare, = [s for s in symbols if s.startswith('_Z22_flash_kda_fwd_prepare')]
    vd = 64 if vsplit else 128
    recurrence, = [s for s in symbols if s.startswith('_Z25_flash_kda_fwd_recurrence')
                   and f'ElLi{vd}EEv' in s]
    sha = hashlib.sha256(cubin.read_bytes()).hexdigest()
    manifest['modules'][MODULE] = {'source': MODULE + '.cubin', 'sha256': sha}
    for op in manifest['ops'].values():
        launches = op['impl'].get('launches', [])
        if not any(l.get('module') == 'flash_kda_d128' for l in launches):
            continue
        assert len(launches) == 2
        p, r = copy.deepcopy(launches)
        assert p['block'] == [256, 1, 1] and r['grid'][0] == 1
        wp = [{'param': i} for i in range(10, 16)]
        wt = ['buffer<bf16>'] * 3 + ['buffer<f32>'] + ['buffer<bf16>'] * 2
        p.update(module=MODULE, entry=prepare, block=[128, 1, 1], shared_mem=21504,
                 params=p['params'][:5] + p['params'][11:19] + ['out '+t for t in wt],
                 args=p['args'][:5] + p['args'][11:19] + wp)
        args = r['args']
        state_in, state_out = copy.deepcopy(args[8:10])
        if vsplit:
            for state in [state_in, state_out]:
                state['pack']['fields'][0]['tensormap']['box'][1] = vd
        r.update(module=MODULE, entry=recurrence, shared_mem=68608 if vsplit else 98432,
                 grid=[r['grid'][1] * (2 if vsplit else 1), 1, 1],
                 params=['bytes<256>']*5 + ['out buffer<bf16>', 'i64', 'i64',
                         'i32', 'i32', 'i32', 'i64', 'bytes<4>'] + ['in '+t for t in wt],
                 args=[args[0], args[1], state_in, state_out, args[10], args[11],
                       {'i64': 0}, {'i64': 0}] + args[12:] + wp)
        op['impl']['launches'] = [p, r]
    used = {l['module'] for op in manifest['ops'].values()
            for l in op['impl'].get('launches', []) if 'module' in l}
    manifest['modules'] = {k:v for k,v in manifest['modules'].items() if k in used}
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cubin', type=Path, default=ROOT/'prebuilt'/f'{MODULE}.cubin')
    parser.add_argument('--vsplit', action='store_true')
    args = parser.parse_args()
    manifest = generate(json.loads(args.reference.read_text()), args.cubin, args.vsplit)
    args.out.write_text(json.dumps(manifest, indent=1)+'\n')


if __name__ == '__main__':
    main()
