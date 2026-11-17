import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DemoTests(unittest.TestCase):
    def test_generated_demo_is_a_real_checkpoint_not_prefilled_report(self):
        from weight_witness.checkpoint import FormatError, load_checkpoint
        from weight_witness.compare import compare_checkpoints
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'demo'
            result = subprocess.run([sys.executable, str(ROOT/'examples/make_demo.py'), '--output', str(output)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            equal = compare_checkpoints(output/'base', output/'resharded')
            changed = compare_checkpoints(output/'base', output/'candidate')
            self.assertEqual(equal['status'], 'equal')
            self.assertEqual(changed['status'], 'different')
            self.assertEqual(changed['summary']['added'], 1)
            self.assertEqual(changed['summary']['removed'], 1)
            self.assertEqual(changed['summary']['incompatible'], 1)
            with self.assertRaisesRegex(FormatError, 'ownership'):
                load_checkpoint(output/'broken')
            pages = Path(folder)/'pages'
            render = subprocess.run([sys.executable, str(ROOT/'examples/render_demo.py'), '--data', str(output), '--output', str(pages)],
                env={**os.environ, 'PYTHONPATH': str(ROOT/'src')}, capture_output=True, text=True, timeout=30)
            self.assertEqual(render.returncode, 0, render.stderr)
            self.assertIn('Synthetic weights', (pages/'index.html').read_text(encoding='utf8'))
            self.assertTrue((pages/'report.json').is_file())
            repeat = subprocess.run([sys.executable, str(ROOT/'examples/make_demo.py'), '--output', str(output)], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(repeat.returncode, 0, 'demo must not overwrite an existing directory')


if __name__ == '__main__': unittest.main()
