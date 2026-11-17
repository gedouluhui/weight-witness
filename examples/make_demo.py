"""Generate tiny synthetic checkpoints; no models, downloads or dependencies.

Never overwrites an existing output directory. Compare base with resharded for
an equality control, candidate for actual changes, and inspect broken for a
valid-file / invalid-index counterexample.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import struct


def save(path, tensors):
    header, parts, offset = {}, [], 0
    for name, (dtype, shape, values) in sorted(tensors.items()):
        code = {'F32': 'f', 'F16': 'e'}[dtype]
        assert math.prod(shape) == len(values)
        raw = struct.pack('<' + code * len(values), *values)
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset+len(raw)]}
        parts.append(raw); offset += len(raw)
    raw_header = json.dumps(header, separators=(',', ':')).encode('utf8')
    raw_header += b' ' * ((-len(raw_header)) % 8)
    path.write_bytes(struct.pack('<Q', len(raw_header)) + raw_header + b''.join(parts))


def make(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tensors = {'token_embedding.weight': ('F32', [8, 4], [round(math.sin(i) / 8, 5) for i in range(32)])}
    for layer in range(3):
        for projection in ('q_proj', 'k_proj', 'v_proj'):
            tensors[f'encoder.{layer}.attention.{projection}.weight'] = ('F32', [4, 4], [round(math.cos(i+layer) / 8, 5) for i in range(16)])
    tensors['final_norm.weight'] = ('F32', [4], [1.0]*4)
    tensors['legacy_head.bias'] = ('F32', [4], [0.0]*4)
    base = output/'base'; base.mkdir()
    save(base/'model.safetensors', tensors)
    changed = {k: (dtype, list(shape), list(values)) for k, (dtype, shape, values) in tensors.items()}
    for layer in range(3):
        changed[f'encoder.{layer}.attention.q_proj.weight'][2][5] += .5
    changed['encoder.1.attention.v_proj.weight'][2][11] -= .125
    changed['final_norm.weight'] = ('F16', [4], [1.0]*4)
    del changed['legacy_head.bias']
    changed['classifier.bias'] = ('F32', [4], [.1, .2, .3, .4])
    candidate = output/'candidate'; candidate.mkdir()
    save(candidate/'model.safetensors', changed)
    resharded = output/'resharded'; resharded.mkdir()
    names = sorted(tensors)
    mapping = {}
    for shard_number in range(2):
        file = f'model-0000{shard_number+1}-of-00002.safetensors'
        selected = names[shard_number::2]
        save(resharded/file, {name: tensors[name] for name in selected})
        mapping.update({name: file for name in selected})
    index = {'metadata': {'total_size': sum(len(values)*4 for _, _, values in tensors.values())}, 'weight_map': mapping}
    (resharded/'model.safetensors.index.json').write_text(json.dumps(index, indent=2)+'\n', encoding='utf8')
    broken = output/'broken'; shutil.copytree(resharded, broken)
    index['weight_map'][names[0]] = mapping[names[1]]
    (broken/'model.safetensors.index.json').write_text(json.dumps(index, indent=2)+'\n', encoding='utf8')
    return {'example': 'synthetic, not trained model weights', 'tensors_in_base': len(tensors),
            'controls': {'base_vs_resharded': 'equal', 'base_vs_candidate': 'different', 'broken_index': 'error'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/demo', type=Path)
    args = parser.parse_args()
    try:
        result = make(args.output)
    except FileExistsError:
        parser.error('output already exists; choose a fresh directory (nothing was overwritten)')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
