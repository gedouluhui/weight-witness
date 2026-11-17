"""Bounded, read-only parsing of a conservative safetensors subset.

This module validates structure, not numerical health or model quality. It never
imports a model or evaluates serialized code. Concurrent hostile writers require
an external immutable snapshot; stat fingerprints are a best-effort race check.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import stat
import struct
from typing import Iterator


class FormatError(ValueError):
    """The input cannot be fully audited under the supported format contract."""


# Packed/sub-byte and unknown future types are deliberately rejected.
DTYPE_BYTES = {
    'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2,
    'I32': 4, 'U32': 4, 'I64': 8, 'U64': 8,
    'F16': 2, 'BF16': 2, 'F32': 4, 'F64': 8,
    'F8_E4M3': 1, 'F8_E5M2': 1,
}
MAX_CHUNK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class TensorInfo:
    name: str
    file: Path
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass
class Checkpoint:
    tensors: dict[str, TensorInfo]
    files: tuple[Path, ...]
    fingerprints: dict[Path, tuple]
    headers: dict[str, str]


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise FormatError(f'{label} must be a positive integer')


def _fingerprint(path: Path) -> tuple:
    try:
        if path.is_symlink():
            raise FormatError(f'symlink input rejected: {path.name}')
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise FormatError(f'input is not a regular file: {path.name}')
        return (info.st_size, info.st_mtime_ns, info.st_ino, info.st_dev)
    except OSError as exc:
        raise FormatError(f'cannot read source: {path.name}') from exc


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise FormatError(f'duplicate JSON key: {key!r}')
        obj[key] = value
    return obj


def _constant(value):
    raise FormatError(f'non-standard JSON constant: {value}')


def _json(raw: bytes, label: str, *, header=False):
    try:
        text = raw.decode('utf8')
        if header:
            if not text.startswith('{'):
                raise FormatError(f'{label}: header must begin with {{')
            decoder = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant)
            obj, end = decoder.raw_decode(text)
            if text[end:].strip(' '):
                raise FormatError(f'{label}: header trailing padding must be spaces')
        else:
            obj = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
        if not isinstance(obj, dict):
            raise FormatError(f'{label}: JSON object required')
        stack = [obj]
        while stack:
            item = stack.pop()
            if isinstance(item, str):
                item.encode('utf8', errors='strict')
            elif isinstance(item, dict):
                stack.extend(item.keys())
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
        return obj
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, FormatError):
            raise
        raise FormatError(f'{label}: invalid or excessively nested JSON') from exc


def _read_file(path, header_budget, tensor_budget):
    before = _fingerprint(path)
    try:
        with path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise FormatError(f'{path.name}: missing 8-byte header length')
            length = struct.unpack('<Q', prefix)[0]
            if length > header_budget or length > before[0] - 8:
                raise FormatError(f'{path.name}: header length exceeds budget or file')
            raw = stream.read(length)
            if len(raw) != length:
                raise FormatError(f'{path.name}: truncated header')
    except OSError as exc:
        raise FormatError(f'{path.name}: header read failed') from exc
    header = _json(raw, path.name, header=True)
    metadata = header.get('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise FormatError(f'{path.name}: metadata must map strings to strings')
    tensors = {}
    spans = []
    origin = 8 + length
    data_size = before[0] - origin
    for name, desc in header.items():
        if name == '__metadata__':
            continue
        if len(tensors) >= tensor_budget:
            raise FormatError('tensor count exceeds budget')
        if not isinstance(desc, dict) or set(desc) != {'dtype', 'shape', 'data_offsets'}:
            raise FormatError(f'{name!r}: tensor requires dtype, shape and data_offsets only')
        dtype, shape, offsets = desc['dtype'], desc['shape'], desc['data_offsets']
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise FormatError(f'{name!r}: unsupported dtype {dtype!r}')
        if not isinstance(shape, list) or len(shape) > 64 or any(type(d) is not int or d < 0 for d in shape):
            raise FormatError(f'{name!r}: shape must contain up to 64 nonnegative integer dimensions')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int or n < 0 for n in offsets):
            raise FormatError(f'{name!r}: offsets require two nonnegative integers')
        start, end = offsets
        if start > end or end > data_size:
            raise FormatError(f'{name!r}: offsets outside payload')
        elements = 1
        for dimension in shape:
            if dimension > 2**64 - 1:
                raise FormatError(f'{name!r}: dimension exceeds unsigned 64-bit range')
            elements *= dimension
            if elements > 2**64 - 1:
                raise FormatError(f'{name!r}: dimension product overflows unsigned 64-bit range')
        byte_count = elements * DTYPE_BYTES[dtype]
        if byte_count > 2**64 - 1 or end - start != byte_count:
            raise FormatError(f'{name!r}: shape/dtype byte count disagrees with offsets')
        tensors[name] = TensorInfo(name, path, dtype, tuple(shape), origin + start, origin + end)
        spans.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(spans):
        if start != cursor:
            raise FormatError(f'{name!r}: payload has overlap or gap')
        cursor = end
    if cursor != data_size:
        raise FormatError(f'{path.name}: unindexed trailing payload')
    if _fingerprint(path) != before:
        raise FormatError(f'{path.name}: source changed during header read')
    return tensors, before, hashlib.sha256(raw).hexdigest()


def load_checkpoint(path, *, max_header_bytes=16777216, max_tensors=100000, max_shards=4096) -> Checkpoint:
    """Validate one file, a declared index, or an unambiguous checkpoint directory."""
    for val, label in ((max_header_bytes, 'header budget'), (max_tensors, 'tensor budget'), (max_shards, 'shard budget')):
        _positive_int(val, label)
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise FormatError('symlink input components are unsupported')
    if path.is_dir():
        index = path / 'model.safetensors.index.json'
        standard = path / 'model.safetensors'
        if index.exists():
            path = index
        elif standard.exists():
            path = standard
        else:
            choices = sorted(path.glob('*.safetensors'))
            if len(choices) != 1:
                raise FormatError('directory must have a standard index or exactly one safetensors file')
            path = choices[0]
    if path.name.endswith('.safetensors'):
        tensors, fp, digest = _read_file(path, max_header_bytes, max_tensors)
        return Checkpoint(tensors, (path,), {path: fp}, {path.name: digest})
    if not path.name.endswith('.safetensors.index.json'):
        raise FormatError('expected .safetensors or .safetensors.index.json input')
    before = _fingerprint(path)
    if before[0] > max_header_bytes:
        raise FormatError('index exceeds header budget')
    try:
        with path.open('rb') as stream:
            raw = stream.read(max_header_bytes + 1)
    except OSError as exc:
        raise FormatError('index read failed') from exc
    if len(raw) > max_header_bytes:
        raise FormatError('index exceeds header budget')
    obj = _json(raw, path.name)
    mapping = obj.get('weight_map')
    if not isinstance(mapping, dict) or not mapping or len(mapping) > max_tensors:
        raise FormatError('index weight_map must be nonempty and within tensor budget')
    for name, shard in mapping.items():
        if (not isinstance(shard, str) or not shard.endswith('.safetensors')
            or any(char in shard for char in ('/', '\\', ':', '\x00')) or shard in ('.', '..')):
            raise FormatError(f'{name!r}: shard must be a simple safetensors basename')
    files = tuple(path.parent / s for s in sorted(set(mapping.values())))
    if len(files) > max_shards:
        raise FormatError('shard count exceeds budget')
    tensors = {}
    fingerprints = {path: before}
    headers = {path.name: hashlib.sha256(raw).hexdigest()}
    for file in files:
        part, fp, digest = _read_file(file, max_header_bytes, max_tensors - len(tensors))
        for name, tensor in part.items():
            if name in tensors:
                raise FormatError(f'{name!r}: duplicate ownership across shards')
            if mapping.get(name) != file.name:
                raise FormatError(f'{name!r}: index ownership differs from actual shard {file.name}')
            tensors[name] = tensor
        fingerprints[file] = fp
        headers[file.name] = digest
    if set(tensors) != set(mapping):
        raise FormatError('index declares tensors absent from their shards')
    metadata = obj.get('metadata', {})
    if not isinstance(metadata, dict):
        raise FormatError('index metadata must be an object')
    if 'total_size' in metadata:
        total = metadata['total_size']
        if type(total) is not int or total != sum(t.nbytes for t in tensors.values()):
            raise FormatError('index metadata.total_size does not equal payload byte total')
    result = Checkpoint(tensors, files, fingerprints, headers)
    assert_unchanged(result)
    return result


def iter_tensor_bytes(tensor: TensorInfo, chunk_bytes=1048576) -> Iterator[bytes]:
    """Yield each payload byte exactly once, bounded by the requested chunk size."""
    _positive_int(chunk_bytes, 'chunk size')
    if chunk_bytes > MAX_CHUNK_BYTES:
        raise FormatError('chunk size exceeds 64 MiB limit')
    remaining = tensor.nbytes
    _fingerprint(tensor.file)
    try:
        with tensor.file.open('rb') as stream:
            stream.seek(tensor.start)
            while remaining:
                block = stream.read(min(chunk_bytes, remaining))
                if not block:
                    raise FormatError(f'{tensor.name!r}: truncated tensor payload')
                remaining -= len(block)
                yield block
    except OSError as exc:
        raise FormatError(f'{tensor.name!r}: tensor read failed') from exc


def assert_unchanged(checkpoint: Checkpoint) -> None:
    for path, expected in checkpoint.fingerprints.items():
        if _fingerprint(path) != expected:
            raise FormatError(f'{path.name}: source changed during scan; use a stable snapshot')
