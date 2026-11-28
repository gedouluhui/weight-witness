import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weight_witness.cli import _write_outputs


class OutputTransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.first = self.root / 'report.json'
        self.second = self.root / 'report.html'

    def fail_second_replace(self):
        real_replace = os.replace
        failed = False

        def replace(source, target):
            nonlocal failed
            if Path(target) == self.second and not failed:
                failed = True
                raise OSError('injected')
            return real_replace(source, target)

        return patch('weight_witness.cli.os.replace', side_effect=replace)

    def test_second_replace_failure_preserves_first_report(self):
        self.first.write_text('old-json')
        self.second.write_text('old-html')
        with self.fail_second_replace():
            with self.assertRaises(OSError):
                _write_outputs([(self.first, 'new-json'), (self.second, 'new-html')])
        assert self.first.read_text() == 'old-json'
        assert self.second.read_text() == 'old-html'
        assert not list(self.root.glob('.weight-witness-*'))

    def test_second_replace_failure_removes_first_new_report(self):
        with self.fail_second_replace():
            with self.assertRaises(OSError):
                _write_outputs([(self.first, 'new-json'), (self.second, 'new-html')])
        assert not self.first.exists()
        assert not self.second.exists()
        assert not list(self.root.glob('.weight-witness-*'))

    def test_all_outputs_are_staged_before_first_replace(self):
        real_open = Path.open
        real_replace = os.replace
        events = []

        def open_file(path, *args, **kwargs):
            if path.name == 'new' and args and args[0] == 'w':
                events.append('stage')
            return real_open(path, *args, **kwargs)

        def replace(source, target):
            events.append('replace')
            return real_replace(source, target)

        with patch('weight_witness.cli.Path.open', open_file), \
                patch('weight_witness.cli.os.replace', side_effect=replace):
            _write_outputs([(self.first, 'new-json'), (self.second, 'new-html')])
        self.assertEqual(events[:2], ['stage', 'stage'])
        self.assertEqual(events.count('stage'), 2)

    def test_rollback_failure_retains_recovery_backups(self):
        self.first.write_text('old-json')
        self.second.write_text('old-html')
        real_replace = os.replace
        failed_second = False
        failed_rollback = False

        def replace(source, target):
            nonlocal failed_second, failed_rollback
            if Path(target) == self.second and not failed_second:
                failed_second = True
                raise OSError('injected publication failure')
            if (failed_second and Path(target) == self.first
                    and Path(source).name == 'old' and not failed_rollback):
                failed_rollback = True
                raise OSError('injected rollback failure')
            return real_replace(source, target)

        with patch('weight_witness.cli.os.replace', side_effect=replace):
            with self.assertRaisesRegex(OSError, 'rollback failed.*recovery files retained'):
                _write_outputs([(self.first, 'new-json'), (self.second, 'new-html')])
        assert self.first.read_text() == 'new-json'
        assert self.second.read_text() == 'old-html'
        backups = list(self.root.glob('.weight-witness-*/old'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), 'old-json')

    def test_successful_publication_replaces_both_reports(self):
        self.first.write_text('old-json')
        self.second.write_text('old-html')
        _write_outputs([(self.first, 'new-json'), (self.second, 'new-html')])
        assert self.first.read_text() == 'new-json'
        assert self.second.read_text() == 'new-html'
        assert not list(self.root.glob('.weight-witness-*'))


if __name__ == '__main__':
    unittest.main()
