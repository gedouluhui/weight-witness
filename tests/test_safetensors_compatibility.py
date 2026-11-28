"""Compatibility corpus for the official safetensors 0.8 format contract.

``classification`` describes the documented contract, not every permissive JSON
behavior of the native reader. ``unsupported`` marks a native-reader extension
that WeightWitness deliberately rejects. The optional native check records
reader behavior separately so parser quirks cannot silently redefine the format.

Contract: https://docs.rs/safetensors/0.8.0/safetensors/#format
Reader: https://docs.rs/safetensors/0.8.0/src/safetensors/tensor.rs.html
"""
from contextlib import redirect_stdout
from dataclasses import dataclass
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

from weight_witness.cli import main
from weight_witness.checkpoint import FormatError, load_checkpoint


PAYLOAD = struct.pack('<f', 1.0)


def descriptor(**changes):
    value = {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}
    value.update(changes)
    return value


def header(value):
    return json.dumps(value, separators=(',', ':')).encode('utf8')


@dataclass(frozen=True)
class Case:
    name: str
    raw_header: bytes
    classification: str
    native_accepts: bool
    witness_accepts: bool
    rationale: str
    payload: bytes = PAYLOAD
    truncated_header: bool = False


CANONICAL = descriptor()
CORPUS = (
    Case('canonical', header({'x': CANONICAL}), 'accepted', True, True,
         'Required tensor fields and a fully indexed payload.'),
    Case('trailing_space_padding', header({'x': CANONICAL}) + b'    ', 'accepted', True, True,
         'The contract permits trailing 0x20 padding.'),
    Case('leading_whitespace', b' ' + header({'x': CANONICAL}), 'rejected', True, False,
         'The header MUST begin with {; native serde_json currently tolerates leading JSON whitespace.'),
    Case('trailing_newline', header({'x': CANONICAL}) + b'\n', 'rejected', True, False,
         'The contract permits trailing 0x20 padding, not a newline; native serde_json currently tolerates it.'),
    Case('metadata_null', header({'__metadata__': None, 'x': CANONICAL}), 'unsupported', True, False,
         'Native Option deserialization treats null as absent; it is outside the documented string-to-string map contract.'),
    Case('extra_tensor_field', header({'x': descriptor(future='ignored')}), 'unsupported', True, False,
         'Native serde ignores unknown tensor fields; WeightWitness intentionally requires the exact supported descriptor.'),
    Case('duplicate_tensor_key',
         b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
         b'"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
         'rejected', True, False,
         'Duplicate keys are explicitly disallowed even though the native map deserializer currently keeps one value.'),
    Case('duplicate_nested_key',
         b'{"x":{"dtype":"F32","dtype":"F16","shape":[1],"data_offsets":[0,4]}}',
         'rejected', False, False,
         'Duplicate keys are explicitly disallowed; native struct deserialization also rejects a repeated dtype field.'),
    Case('reversed_offsets', header({'x': descriptor(data_offsets=[4, 0])}), 'rejected', False, False,
         'BEGIN must not exceed END.'),
    Case('offset_byte_count_mismatch', header({'x': descriptor(data_offsets=[0, 8])}), 'rejected', False, False,
         'END minus BEGIN must equal shape times dtype width.'),
    Case('offset_gap', header({'x': descriptor(data_offsets=[4, 8])}), 'rejected', False, False,
         'The byte buffer must be fully indexed without holes.'),
    Case('truncated_header', header({'x': CANONICAL}), 'rejected', False, False,
         'The declared header length must fit within the file.', truncated_header=True),
    Case('truncated_payload', header({'x': CANONICAL}), 'rejected', False, False,
         'The indexed payload must fit within the file.', payload=b'\x00\x00'),
)


class SafetensorsCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_case(self, case, index=0):
        path = self.root / f'{index}.safetensors'
        length = len(case.raw_header) + (1 if case.truncated_header else 0)
        path.write_bytes(struct.pack('<Q', length) + case.raw_header + case.payload)
        return path

    def test_corpus_classification_and_public_verdicts_are_stable(self):
        self.assertEqual({case.classification for case in CORPUS}, {'accepted', 'rejected', 'unsupported'})
        for index, case in enumerate(CORPUS):
            with self.subTest(case=case.name):
                path = self.write_case(case, index)
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    result = main(['inspect', str(path)])
                verdict = json.loads(stdout.getvalue())
                if case.witness_accepts:
                    self.assertEqual((result, verdict['status']), (0, 'valid'))
                    self.assertEqual(load_checkpoint(path).tensors['x'].shape, (1,))
                else:
                    self.assertEqual((result, verdict['status']), (2, 'error'))
                    with self.assertRaises(FormatError):
                        load_checkpoint(path)

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ('numpy', 'safetensors')),
        'optional native compatibility check needs safetensors and numpy',
    )
    def test_native_reader_observations_match_corpus(self):
        from safetensors import SafetensorError, safe_open

        for index, case in enumerate(CORPUS):
            with self.subTest(case=case.name):
                path = self.write_case(case, index)
                try:
                    with safe_open(str(path), framework='np') as source:
                        list(source.keys())
                    accepted = True
                except SafetensorError:
                    accepted = False
                self.assertEqual(accepted, case.native_accepts, case.rationale)


if __name__ == '__main__':
    unittest.main()
