"""Streaming payload fingerprints and explicitly scoped numerical comparisons."""
import hashlib
import math
import struct

from .checkpoint import (
    DTYPE_BYTES, MAX_CHUNK_BYTES, FormatError, assert_unchanged,
    iter_tensor_bytes, load_checkpoint,
)

FLOAT_DTYPES = ('F32', 'F16', 'BF16')
LIMITATIONS = [
    'Structural validity and weight equality do not establish model quality, provenance, or safety.',
    'Only declared/referenced shards are audited; unrelated files in a directory are not scanned.',
    'F32/F16/BF16 receive numerical scans. Other supported dtypes are bytewise only, including F64/FP8.',
    'Source size/mtime/inode checks detect ordinary changes, not adversarial writes; use immutable snapshots.',
    'Reports omit absolute paths and metadata values, but tensor and shard names can still disclose information.',
    'Payloads are fully read; pure-Python numerical scanning favors portability, not GPU-scale throughput.',
]


def _chunk_check(size):
    if type(size) is not int or not 0 < size <= MAX_CHUNK_BYTES:
        raise FormatError('chunk size must be an integer between 1 and 67108864 bytes')


def _tolerance(value, label):
    if type(value) not in (int, float) or value < 0:
        raise FormatError(f'{label} must be finite and nonnegative')
    try:
        value = float(value)
    except OverflowError as exc:
        raise FormatError(f'{label} must be finite and nonnegative') from exc
    if not math.isfinite(value):
        raise FormatError(f'{label} must be finite and nonnegative')
    return value


def _values(tensor, chunk, hasher):
    size = DTYPE_BYTES[tensor.dtype]
    code = {'F32': '<f', 'F16': '<e', 'BF16': '<H'}[tensor.dtype]
    pending = b''
    for block in iter_tensor_bytes(tensor, chunk):
        hasher.update(block)
        data = pending + block
        end = len(data) - len(data) % size
        for item, in struct.iter_unpack(code, data[:end]):
            if tensor.dtype == 'BF16':
                item = struct.unpack('<f', struct.pack('<I', item << 16))[0]
            yield item
        pending = data[end:]
    if pending:
        raise FormatError('unaligned tensor payload')


class _Sum:
    """Compensated sum with deterministic per-element order across chunk sizes."""
    def __init__(self):
        self.value = 0.0
        self.correction = 0.0

    def add(self, value):
        y = value - self.correction
        total = self.value + y
        self.correction = (total - self.value) - y
        self.value = total


def _descriptor(tensor, digest):
    return {'dtype': tensor.dtype, 'shape': list(tensor.shape),
            'nbytes': tensor.nbytes, 'sha256': digest}


def _scan(tensor, chunk):
    hasher = hashlib.sha256()
    nonfinite = None
    if tensor.dtype in FLOAT_DTYPES:
        nonfinite = sum(not math.isfinite(v) for v in _values(tensor, chunk, hasher))
    else:
        for block in iter_tensor_bytes(tensor, chunk):
            hasher.update(block)
    return _descriptor(tensor, hasher.hexdigest()), nonfinite


def _metrics(left, right, chunk, atol, rtol):
    lh, rh = hashlib.sha256(), hashlib.sha256()
    lv, rv = _values(left, chunk, lh), _values(right, chunk, rh)
    absolute, squared = _Sum(), _Sum()
    count = finite = changed = outside = bad_left = bad_right = 0
    maximum = worst = None
    # Same shape and dtype were verified before entering. zip exhausts both
    # streams on every nonempty equal-length tensor; explicitly drain to run
    # both generators' EOF validation, including the empty case.
    for i, (a, b) in enumerate(zip(lv, rv)):
        count += 1
        fa, fb = math.isfinite(a), math.isfinite(b)
        bad_left += not fa
        bad_right += not fb
        if not (fa and fb):
            continue
        finite += 1
        delta = abs(b - a)
        changed += a != b
        outside += delta > atol + rtol * abs(a)
        absolute.add(delta)
        squared.add(delta * delta)
        if maximum is None or delta > maximum:
            maximum, worst = delta, i
    # A malformed short read raises inside the iterators. Any remaining value
    # would violate the validated shape contract rather than be silently lost.
    if next(lv, None) is not None or next(rv, None) is not None or count != left.numel:
        raise FormatError('tensor stream length changed during comparison')
    result = {
        'count': count, 'finite_pairs': finite, 'changed_values': changed,
        'outside_tolerance': outside, 'nonfinite_left': bad_left,
        'nonfinite_right': bad_right, 'max_abs': maximum,
        'mean_abs': absolute.value / finite if finite else None,
        'rmse': math.sqrt(squared.value / finite) if finite else None,
        'worst_index': worst,
    }
    return result, _descriptor(left, lh.hexdigest()), _descriptor(right, rh.hexdigest())


def _source(cp):
    return {'files': [p.name for p in cp.files], 'header_sha256': dict(sorted(cp.headers.items()))}


def inspect_checkpoint(path, *, chunk_bytes=1048576):
    """Fully hash payloads and numerically inspect supported floating dtypes."""
    _chunk_check(chunk_bytes)
    cp = load_checkpoint(path)
    assert_unchanged(cp)
    rows, total_bad = [], 0
    for name, tensor in sorted(cp.tensors.items()):
        descriptor, bad = _scan(tensor, chunk_bytes)
        total_bad += bad or 0
        rows.append({'name': name, **descriptor, 'nonfinite': bad})
    assert_unchanged(cp)
    limitations = list(LIMITATIONS)
    if not rows:
        limitations.append('EMPTY CHECKPOINT: no tensor payloads were audited.')
    return {
        'schema_version': 1, 'kind': 'inspect',
        'status': 'nonfinite' if total_bad else 'valid',
        'sources': {'input': _source(cp)},
        'summary': {'tensors': len(rows), 'files': len(cp.files),
                    'bytes': sum(t.nbytes for t in cp.tensors.values()), 'nonfinite': total_bad},
        'tensors': rows, 'limitations': limitations,
    }


def compare_checkpoints(left, right, *, atol=0.0, rtol=0.0, chunk_bytes=1048576):
    """Compare tensors by name, independent of shard layout.

Tolerance reference is LEFT: abs(right-left) <= atol + rtol*abs(left).
Exact byte equality and numerical closeness are separate statuses.
"""
    atol, rtol = _tolerance(atol, 'atol'), _tolerance(rtol, 'rtol')
    _chunk_check(chunk_bytes)
    a, b = load_checkpoint(left), load_checkpoint(right)
    assert_unchanged(a)
    assert_unchanged(b)
    rows = []
    summary = dict.fromkeys(('total', 'equal', 'close', 'changed', 'added', 'removed', 'incompatible', 'nonfinite'), 0)
    for name in sorted(set(a.tensors) | set(b.tensors)):
        lt, rt = a.tensors.get(name), b.tensors.get(name)
        ld = rd = metrics = None
        bad_left = bad_right = None
        if lt is None:
            rd, bad_right = _scan(rt, chunk_bytes)
            status = 'added'
        elif rt is None:
            ld, bad_left = _scan(lt, chunk_bytes)
            status = 'removed'
        elif lt.shape != rt.shape or lt.dtype != rt.dtype:
            ld, bad_left = _scan(lt, chunk_bytes)
            rd, bad_right = _scan(rt, chunk_bytes)
            status = 'incompatible'
        elif lt.dtype in FLOAT_DTYPES:
            metrics, ld, rd = _metrics(lt, rt, chunk_bytes, atol, rtol)
            bad_left, bad_right = metrics['nonfinite_left'], metrics['nonfinite_right']
            if bad_left or bad_right or metrics['outside_tolerance']:
                status = 'changed'
            elif ld['sha256'] == rd['sha256']:
                status = 'equal'
            else:
                status = 'close'
        else:
            ld, bad_left = _scan(lt, chunk_bytes)
            rd, bad_right = _scan(rt, chunk_bytes)
            status = 'equal' if ld['sha256'] == rd['sha256'] else 'changed'
        rows.append({'name': name, 'status': status, 'left': ld, 'right': rd, 'metrics': metrics,
                     'nonfinite_left': bad_left, 'nonfinite_right': bad_right})
        summary[status] += 1
        summary['total'] += 1
        summary['nonfinite'] += bool(bad_left or bad_right)
    assert_unchanged(a)
    assert_unchanged(b)
    different = any(summary[k] for k in ('changed', 'added', 'removed', 'incompatible'))
    limitations = list(LIMITATIONS)
    if not rows:
        limitations.append('EMPTY CHECKPOINTS: no tensor payloads were compared.')
    return {
        'schema_version': 1, 'kind': 'diff',
        'status': 'different' if different else ('close' if summary['close'] else 'equal'),
        'policy': {'atol': atol, 'rtol': rtol, 'float_dtypes': list(FLOAT_DTYPES),
                   'other_dtypes': 'exact payload byte equality only'},
        'summary': summary, 'sources': {'left': _source(a), 'right': _source(b)},
        'tensors': rows, 'limitations': limitations,
    }
