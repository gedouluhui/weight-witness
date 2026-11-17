import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest


def fixture(path, values, dtype='F32', name='layer.weight', shape=None):
    codes = {'F32': 'f', 'F16': 'e', 'I32': 'i', 'F64': 'd'}
    if dtype == 'BF16':
        raw = b''.join(struct.pack('<I', struct.unpack('<I', struct.pack('<f', v))[0])[2:] for v in values)
    else:
        raw = struct.pack('<' + codes[dtype] * len(values), *values)
    header = json.dumps({name: {'dtype': dtype, 'shape': shape if shape is not None else [len(values)], 'data_offsets': [0, len(raw)]}}).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header + raw)
    return raw


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert importlib.util.find_spec('weight_witness.compare'), 'compare implementation not present'
        from weight_witness import compare
        cls.api = compare

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.left, self.right = self.root/'left.safetensors', self.root/'right.safetensors'

    def compare(self, a, b, **kwargs):
        fixture(self.left, a)
        fixture(self.right, b)
        return self.api.compare_checkpoints(self.left, self.right, **kwargs)

    def test_hand_computed_metrics(self):
        report = self.compare([1, 2, 3], [1, 4, 3])
        self.assertEqual(report['status'], 'different')
        m = report['tensors'][0]['metrics']
        self.assertEqual((m['count'], m['changed_values'], m['outside_tolerance'], m['max_abs'], m['worst_index']), (3, 1, 1, 2, 1))
        self.assertAlmostEqual(m['mean_abs'], 2/3)
        self.assertAlmostEqual(m['rmse'], math.sqrt(4/3))
        self.assertEqual(report['summary']['changed'], 1)

    def test_exact_hash_and_equal(self):
        payload = fixture(self.left, [1, 2, 3])
        fixture(self.right, [1, 2, 3])
        report = self.api.compare_checkpoints(self.left, self.right)
        self.assertEqual(report['status'], 'equal')
        self.assertEqual(report['tensors'][0]['left']['sha256'], hashlib.sha256(payload).hexdigest())
        self.assertNotIn(str(self.root), json.dumps(report))

    def test_chunk_invariance(self):
        first = self.compare([1, 2, 3, 0, -9], [1, 4, 3, .5, -8], chunk_bytes=1)
        for chunk in (2, 3, 7, 8, 1024):
            self.assertEqual(first, self.api.compare_checkpoints(self.left, self.right, chunk_bytes=chunk))

    def test_tolerance_boundary_and_reference_direction(self):
        self.assertEqual(self.compare([2], [3], atol=1)['status'], 'close')
        self.assertEqual(self.compare([2], [3], rtol=.5)['status'], 'close')
        self.assertEqual(self.compare([3], [2], rtol=.34)['status'], 'close')
        self.assertEqual(self.compare([2], [3], rtol=.34)['status'], 'different')

    def test_signed_zero_is_numerically_close_not_identical(self):
        report = self.compare([0.0], [-0.0])
        self.assertEqual(report['status'], 'close')
        self.assertEqual(report['tensors'][0]['metrics']['changed_values'], 0)
        self.assertNotEqual(report['tensors'][0]['left']['sha256'], report['tensors'][0]['right']['sha256'])

    def test_nonfinite_is_never_green_and_json_is_standard(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            report = self.compare([value, 1], [value, 2])
            self.assertEqual(report['status'], 'different')
            m = report['tensors'][0]['metrics']
            self.assertEqual((m['nonfinite_left'], m['nonfinite_right'], m['finite_pairs']), (1, 1, 1))
            self.assertEqual(report['summary']['nonfinite'], 1)
            json.dumps(report, allow_nan=False)

    def test_all_nonfinite_metrics_null(self):
        report = self.compare([float('nan')], [float('nan')])
        m = report['tensors'][0]['metrics']
        self.assertIsNone(m['max_abs'])
        self.assertIsNone(m['worst_index'])

    def test_f16_and_bf16_decode(self):
        for dtype in ('F16', 'BF16'):
            fixture(self.left, [1, 2, -4], dtype)
            fixture(self.right, [1, 2.5, -4], dtype)
            report = self.api.compare_checkpoints(self.left, self.right, chunk_bytes=1)
            self.assertEqual(report['tensors'][0]['metrics']['max_abs'], .5)
            self.assertEqual(report['tensors'][0]['metrics']['worst_index'], 1)

    def test_integer_and_f64_are_bytewise_not_float_claim(self):
        for dtype in ('I32', 'F64'):
            fixture(self.left, [1, 2], dtype)
            fixture(self.right, [1, 3], dtype)
            r = self.api.compare_checkpoints(self.left, self.right, atol=999)
            self.assertEqual(r['status'], 'different')
            self.assertIsNone(r['tensors'][0]['metrics'])

    def test_added_removed(self):
        fixture(self.left, [1], name='old')
        fixture(self.right, [1], name='new')
        r = self.api.compare_checkpoints(self.left, self.right)
        self.assertEqual([t['status'] for t in r['tensors']], ['added', 'removed'])
        self.assertEqual((r['summary']['added'], r['summary']['removed']), (1, 1))

    def test_dtype_and_shape_incompatible(self):
        fixture(self.left, [1, 2])
        fixture(self.right, [1, 2], 'F16')
        self.assertEqual(self.api.compare_checkpoints(self.left, self.right)['tensors'][0]['status'], 'incompatible')
        fixture(self.right, [1, 2], shape=[1, 2])
        self.assertEqual(self.api.compare_checkpoints(self.left, self.right)['summary']['incompatible'], 1)

    def test_empty_tensor_and_scalar(self):
        r = self.compare([], [])
        self.assertEqual(r['status'], 'equal')
        self.assertEqual(r['tensors'][0]['metrics']['count'], 0)
        fixture(self.left, [3], shape=[])
        fixture(self.right, [5], shape=[])
        self.assertEqual(self.api.compare_checkpoints(self.left, self.right)['tensors'][0]['metrics']['max_abs'], 2)

    def test_invalid_knobs_rejected(self):
        fixture(self.left, [1]); fixture(self.right, [2])
        for kwargs in ({'atol': -1}, {'rtol': float('inf')}, {'atol': float('nan')}, {'atol': True}, {'chunk_bytes': 0}, {'chunk_bytes': True}, {'chunk_bytes': 67108865}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.api.compare_checkpoints(self.left, self.right, **kwargs)

    def test_inspect_real_payload_and_nonfinite(self):
        payload = fixture(self.left, [1, float('nan')])
        r = self.api.inspect_checkpoint(self.left)
        self.assertEqual(r['status'], 'nonfinite')
        self.assertEqual(r['summary'], {'tensors': 1, 'files': 1, 'bytes': 8, 'nonfinite': 1})
        self.assertEqual(r['tensors'][0]['sha256'], hashlib.sha256(payload).hexdigest())
        json.dumps(r, allow_nan=False)

    def test_f32_extremes_metrics_stay_finite(self):
        r = self.compare([3e38, -3e38], [-3e38, 3e38])
        self.assertTrue(math.isfinite(r['tensors'][0]['metrics']['rmse']))
        json.dumps(r, allow_nan=False)


if __name__ == '__main__': unittest.main()
