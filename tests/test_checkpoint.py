import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest


def write_tensor_file(path, entries=None, payload=None, raw_header=None):
    if entries is None:
        entries = {'x': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}}
    header = raw_header if raw_header is not None else json.dumps(entries).encode('utf8')
    payload = struct.pack('<ff', 1, 2) if payload is None else payload
    path.write_bytes(struct.pack('<Q', len(header)) + header + payload)
    return path


class ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert importlib.util.find_spec('weight_witness.checkpoint'), 'reader implementation not present'
        from weight_witness import checkpoint
        cls.api = checkpoint

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'model.safetensors'

    def reject(self, **kwargs):
        write_tensor_file(self.path, **kwargs)
        with self.assertRaises(self.api.FormatError):
            self.api.load_checkpoint(self.path)

    def test_real_bytes_and_absolute_spans(self):
        write_tensor_file(self.path)
        cp = self.api.load_checkpoint(self.path)
        t = cp.tensors['x']
        self.assertEqual((t.nbytes, t.numel, t.shape), (8, 2, (2,)))
        self.assertEqual(t.start, len(self.path.read_bytes()) - 8)
        for size in (1, 3, 8, 2048):
            chunks = list(self.api.iter_tensor_bytes(t, size))
            self.assertTrue(all(len(c) <= size for c in chunks))
            self.assertEqual(b''.join(chunks), struct.pack('<ff', 1, 2))
        self.api.assert_unchanged(cp)
        self.assertEqual(len(cp.headers[self.path.name]), 64)

    def test_empty_and_scalar_tensors(self):
        entries = {'empty': {'dtype': 'F32', 'shape': [2, 0], 'data_offsets': [0, 0]},
                   'scalar': {'dtype': 'F32', 'shape': [], 'data_offsets': [0, 4]}}
        write_tensor_file(self.path, entries, struct.pack('<f', 3))
        cp = self.api.load_checkpoint(self.path)
        self.assertEqual(cp.tensors['empty'].numel, 0)
        self.assertEqual(cp.tensors['scalar'].numel, 1)

    def test_empty_checkpoint_legal(self):
        write_tensor_file(self.path, {}, b'')
        self.assertEqual(self.api.load_checkpoint(self.path).tensors, {})

    def test_duplicate_header_key(self):
        self.reject(raw_header=b'{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}')

    def test_nested_duplicate_key(self):
        self.reject(raw_header=b'{"x":{"dtype":"F32","dtype":"F16","shape":[2],"data_offsets":[0,8]}}')

    def test_bad_metadata(self):
        self.reject(entries={'__metadata__': {'key': 2}})

    def test_header_whitespace_and_nonstandard_constants(self):
        for header in (b' {}', b'{}\xff', b'{"x":NaN}', b'[]', b'{}junk'):
            with self.subTest(header=header):
                self.reject(raw_header=header, payload=b'')
        write_tensor_file(self.path, raw_header=b'{}    ', payload=b'')
        self.assertEqual(self.api.load_checkpoint(self.path).tensors, {})

    def test_invalid_shapes_offsets_and_dtype(self):
        for shape, offsets, dtype in [([True], [0, 4], 'F32'), ([-1], [0, 4], 'F32'),
            ([1.0], [0, 4], 'F32'), ([2], [0, True], 'F32'), ([2], [-1, 7], 'F32'),
            ([2], [0, 4], 'F32'), ([2], [0, 8], 'GUESS'), ([2], [8, 0], 'F32')]:
            with self.subTest(shape=shape, offsets=offsets, dtype=dtype):
                self.reject(entries={'x': {'dtype': dtype, 'shape': shape, 'data_offsets': offsets}})

    def test_gaps_overlap_trailing_and_truncation(self):
        self.reject(entries={'x': {'dtype': 'F32', 'shape': [1], 'data_offsets': [4, 8]}})
        self.reject(entries={'x': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}})
        self.reject(entries={name: {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]} for name in ('x', 'y')})
        self.reject(payload=b'123')
        self.path.write_bytes(struct.pack('<Q', 400) + b'{}')
        with self.assertRaises(self.api.FormatError): self.api.load_checkpoint(self.path)

    def test_unpaired_surrogate_rejected(self):
        for entries in ({'__metadata__': {'key': chr(0xD800)}},
                        {chr(0xD800): {'dtype': 'F32', 'shape': [0], 'data_offsets': [0, 0]}}):
            with self.subTest(entries=repr(entries)):
                self.reject(entries=entries, payload=b'')

    def test_dimension_overflow_before_zero_rejected(self):
        for shape in ([4294967296, 4294967296, 0], [2**64, 0], [2**63]):
            with self.subTest(shape=shape):
                self.reject(entries={'x': {'dtype': 'F32', 'shape': shape, 'data_offsets': [0, 0]}}, payload=b'')

    def test_header_and_tensor_budgets(self):
        write_tensor_file(self.path)
        for kwargs in ({'max_header_bytes': 4}, {'max_tensors': 0}, {'max_shards': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(self.api.FormatError):
                self.api.load_checkpoint(self.path, **kwargs)

    def test_fingerprint_detects_edit(self):
        write_tensor_file(self.path)
        cp = self.api.load_checkpoint(self.path)
        self.path.write_bytes(self.path.read_bytes() + b'x')
        with self.assertRaises(self.api.FormatError): self.api.assert_unchanged(cp)

    def test_short_read_rejected(self):
        write_tensor_file(self.path)
        t = self.api.load_checkpoint(self.path).tensors['x']
        self.path.write_bytes(b'')
        with self.assertRaises(self.api.FormatError): list(self.api.iter_tensor_bytes(t))

    def index(self, mapping, metadata=None):
        index = self.root / 'model.safetensors.index.json'
        obj = {'weight_map': mapping}
        if metadata is not None: obj['metadata'] = metadata
        index.write_text(json.dumps(obj), encoding='utf8')
        return index

    def shards(self):
        for name in ('a', 'b'):
            write_tensor_file(self.root / f'{name}.safetensors',
                {name: {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}})
        return {'a': 'a.safetensors', 'b': 'b.safetensors'}

    def test_shard_index_exact_ownership(self):
        mapping = self.shards()
        index = self.index(mapping, {'total_size': 16})
        cp = self.api.load_checkpoint(index)
        self.assertEqual(set(cp.tensors), {'a', 'b'})
        self.assertEqual(len(cp.files), 2)
        self.assertEqual(set(self.api.load_checkpoint(self.root).tensors), {'a', 'b'})
        self.index({'a': 'b.safetensors', 'b': 'a.safetensors'})
        with self.assertRaisesRegex(self.api.FormatError, 'ownership|declared|index'):
            self.api.load_checkpoint(index)

    def test_index_missing_extra_duplicate_and_total(self):
        mapping = self.shards()
        for bad, metadata in [({'a': 'missing.safetensors'}, None),
            ({'ghost': 'a.safetensors'}, None), (mapping, {'total_size': 1}),
            (mapping, {'total_size': True}), ({}, None)]:
            with self.subTest(bad=bad, metadata=metadata):
                index = self.index(bad, metadata)
                with self.assertRaises(self.api.FormatError): self.api.load_checkpoint(index)
        write_tensor_file(self.root / 'b.safetensors', {'a': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}})
        with self.assertRaises(self.api.FormatError): self.api.load_checkpoint(self.index(mapping))

    def test_index_duplicate_json(self):
        self.shards()
        index = self.root / 'model.safetensors.index.json'
        index.write_text('{"weight_map":{"a":"a.safetensors","a":"b.safetensors"}}')
        with self.assertRaises(self.api.FormatError): self.api.load_checkpoint(index)

    def test_shard_path_escape(self):
        for path in ('../a.safetensors', '/a.safetensors', 'C:a.safetensors', 'a/b.safetensors', 'a\\b.safetensors', '.', '..'):
            with self.subTest(path=path), self.assertRaises(self.api.FormatError):
                self.api.load_checkpoint(self.index({'x': path}))

    def test_directory_ambiguous(self):
        self.shards()
        with self.assertRaises(self.api.FormatError): self.api.load_checkpoint(self.root)

    def test_symlink_rejected_if_available(self):
        real = write_tensor_file(self.root / 'real.safetensors')
        try:
            self.path.symlink_to(real)
        except OSError:
            self.skipTest('OS does not permit unprivileged symlinks')
        with self.assertRaises(self.api.FormatError): self.api.load_checkpoint(self.path)


if __name__ == '__main__':
    unittest.main()
