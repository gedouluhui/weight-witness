"""Independent reference checks. Install safetensors + numpy to run these."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('numpy', 'safetensors'))


@unittest.skipUnless(AVAILABLE, 'optional oracle needs safetensors and numpy')
class NativeOracleTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from safetensors.numpy import save_file, load_file
        from safetensors import safe_open
        self.np, self.save, self.load, self.safe_open = np, save_file, load_file, safe_open
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_native_writer_reader_interoperability(self):
        from weight_witness.checkpoint import load_checkpoint, iter_tensor_bytes
        np = self.np
        for dtype in (np.float16, np.float32, np.float64, np.int8, np.uint8, np.int16, np.int32, np.int64, np.bool_):
            for shape in ((), (0,), (2, 0, 3), (2, 3)):
                with self.subTest(dtype=str(dtype), shape=shape):
                    a = np.ones(shape, dtype=dtype)
                    path = self.root/'model.safetensors'
                    self.save({'tensor': a}, str(path))
                    cp = load_checkpoint(path)
                    native = self.load(str(path))['tensor']
                    self.assertEqual(b''.join(iter_tensor_bytes(cp.tensors['tensor'], 3)), native.tobytes())
                    self.assertEqual(cp.tensors['tensor'].shape, shape)

    def test_random_floats_against_numpy_float64(self):
        from weight_witness.compare import compare_checkpoints
        np = self.np
        rng = np.random.default_rng(20261118)
        for dtype in (np.float16, np.float32):
            for count in (1, 3, 33, 257):
                a = rng.normal(size=count).astype(dtype)
                b = a.copy(); b[count//2] += dtype(.5)
                left, right = self.root/'left.safetensors', self.root/'right.safetensors'
                self.save({'x': a}, str(left)); self.save({'x': b}, str(right))
                aa = self.load(str(left))['x'].astype(np.float64)
                bb = self.load(str(right))['x'].astype(np.float64)
                delta = np.abs(bb-aa)
                result = compare_checkpoints(left, right, chunk_bytes=7)['tensors'][0]['metrics']
                self.assertEqual(result['max_abs'], float(delta.max()))
                self.assertEqual(result['worst_index'], int(delta.argmax()))
                self.assertAlmostEqual(result['mean_abs'], float(delta.mean()), places=14)
                self.assertAlmostEqual(result['rmse'], float(np.sqrt(np.mean(delta**2))), places=14)

    def test_resharding_is_equal_and_wrong_ownership_rejected(self):
        from weight_witness.compare import compare_checkpoints
        from weight_witness.checkpoint import FormatError, load_checkpoint
        np = self.np
        values = {'a': np.arange(6, dtype=np.float32), 'b': np.ones((2, 3), dtype=np.float16)}
        original = self.root/'original.safetensors'; self.save(values, str(original))
        sharded = self.root/'sharded'; sharded.mkdir()
        for key, value in values.items(): self.save({key: value}, str(sharded/f'{key}.safetensors'))
        index = sharded/'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'a':'a.safetensors', 'b':'b.safetensors'}}))
        self.assertEqual(compare_checkpoints(original, sharded)['status'], 'equal')
        index.write_text(json.dumps({'weight_map': {'a':'b.safetensors', 'b':'a.safetensors'}}))
        with self.assertRaisesRegex(FormatError, 'ownership'): load_checkpoint(sharded)

    def test_native_and_ours_reject_unicode_and_overflow(self):
        from weight_witness.checkpoint import FormatError, load_checkpoint
        headers = [{'__metadata__': {'key': chr(0xD800)}},
                   {'x': {'dtype':'F32', 'shape':[4294967296,4294967296,0], 'data_offsets':[0,0]}}]
        path = self.root/'model.safetensors'
        for obj in headers:
            raw = json.dumps(obj).encode()
            path.write_bytes(struct.pack('<Q', len(raw))+raw)
            with self.assertRaises(Exception):
                with self.safe_open(str(path), framework='np') as source: list(source.keys())
            with self.assertRaises(FormatError): load_checkpoint(path)


if __name__ == '__main__': unittest.main()
