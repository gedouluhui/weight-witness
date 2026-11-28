import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    header = b'{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    path.write_bytes(struct.pack('<Q', len(header)) + header + struct.pack('<f', value))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.a, self.b = self.root/'a.safetensors', self.root/'b.safetensors'
        write(self.a, 1); write(self.b, 2)
        self.env = {**os.environ, 'PYTHONPATH': str(ROOT/'src')}

    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'weight_witness', *map(str, args)],
            cwd=self.root, env=self.env, capture_output=True, text=True, encoding='utf8', timeout=30)

    def test_inspect_and_diff_exit_codes(self):
        r = self.run_cli('inspect', self.a)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)['status'], 'valid')
        r = self.run_cli('diff', self.a, self.b)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertEqual(json.loads(r.stdout)['status'], 'different')
        r = self.run_cli('diff', self.a, self.b, '--atol', 1)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)['status'], 'close')

    def test_json_and_html_are_real_outputs(self):
        out, html = self.root/'result.json', self.root/'result.html'
        r = self.run_cli('diff', self.a, self.b, '--json', out, '--html', html)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertEqual(json.loads(out.read_text()), json.loads(r.stdout))
        self.assertIn('Differences found', html.read_text(encoding='utf8'))

    def test_corrupt_input_returns_error_not_pass(self):
        self.a.write_bytes(b'bad')
        r = self.run_cli('inspect', self.a)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertEqual(json.loads(r.stdout)['status'], 'error')

    def test_invalid_tolerance_returns_error(self):
        r = self.run_cli('diff', self.a, self.b, '--atol', 'nan')
        self.assertEqual(r.returncode, 2)
        self.assertEqual(json.loads(r.stdout)['status'], 'error')

    def test_output_cannot_overwrite_source_or_partially_write(self):
        before = self.a.read_bytes()
        output = self.root/'should-not-exist.json'
        r = self.run_cli('diff', self.a, self.b, '--json', output, '--html', self.a)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertEqual(self.a.read_bytes(), before)
        self.assertFalse(output.exists())

    def test_output_alias_is_rejected(self):
        out = self.root/'same.json'
        r = self.run_cli('inspect', self.a, '--json', out, '--html', out)
        self.assertEqual(r.returncode, 2)
        self.assertFalse(out.exists())

    def test_output_shard_and_index_are_protected(self):
        index = self.root/'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'weight': 'a.safetensors'}}))
        for target in (index, self.a):
            before = target.read_bytes()
            r = self.run_cli('inspect', index, '--json', target)
            self.assertEqual(r.returncode, 2)
            self.assertEqual(target.read_bytes(), before)

    def test_missing_output_directory_is_error(self):
        r = self.run_cli('inspect', self.a, '--json', self.root/'absent'/'report.json')
        self.assertEqual(r.returncode, 2)
        self.assertEqual(json.loads(r.stdout)['status'], 'error')

    def test_nonfinite_inspect_exits_one(self):
        write(self.a, float('inf'))
        r = self.run_cli('inspect', self.a)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(json.loads(r.stdout)['status'], 'nonfinite')

    def test_malformed_arguments_are_structured_errors(self):
        for args in (('inspect', self.a, '--chunk-bytes', 'abc'),
                     ('diff', self.a, self.b, '--atol', 'abc'), ('unknown',), ()):
            with self.subTest(args=args):
                r = self.run_cli(*args)
                self.assertEqual(r.returncode, 2)
                self.assertTrue(r.stdout.strip(), 'usage error must produce JSON')
                self.assertEqual(json.loads(r.stdout)['status'], 'error')

    def test_version_and_help(self):
        self.assertEqual(self.run_cli('--help').returncode, 0)
        from weight_witness import __version__
        self.assertIn(__version__, self.run_cli('--version').stdout)


if __name__ == '__main__': unittest.main()
