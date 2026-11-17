"""Run known-bad variants in disposable copies, never in the working tree.

Every mutation must change the bytes and trigger its named assertion (not just a
crash). The unmodified selected suites are the positive controls. This is a small
falsification suite, not a coverage percentage or proof of universal detection.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = [
    ('index-ownership-disabled', 'checkpoint.py', 'mapping.get(name) != file.name', 'False', 'test_checkpoint.py', 'test_shard_index_exact_ownership'),
    ('duplicate-key-disabled', 'checkpoint.py', 'if key in obj:', 'if False:', 'test_checkpoint.py', 'test_duplicate_header_key'),
    ('numeric-threshold-disabled', 'compare.py', 'outside += delta > atol + rtol * abs(a)', 'outside += False', 'test_compare.py', 'test_hand_computed_metrics'),
    ('bf16-wrong-bit-shift', 'compare.py', 'item << 16', 'item << 15', 'test_compare.py', 'test_f16_and_bf16_decode'),
    ('html-escaping-disabled', 'report.py', 'return html.escape(str(value), quote=True)', 'return str(value)', 'test_report.py', 'test_untrusted_strings_cannot_become_markup'),
    ('source-write-guard-disabled', 'cli.py', '        _destinations(args, inputs)', '        pass  # mutation removes entire guard capability', 'test_cli.py', 'test_output_cannot_overwrite_source_or_partially_write'),
]


def run(directory, suite):
    env = {**os.environ, 'PYTHONPATH': str(directory/'src'), 'PYTHONUTF8': '1'}
    return subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', suite, '-v'],
                          cwd=directory, env=env, capture_output=True, text=True, encoding='utf8', timeout=90)


def main():
    baseline = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/weight_witness').glob('*.py')}
    results = []
    with tempfile.TemporaryDirectory(prefix='weight-witness-mutations-') as tmp:
        copy = Path(tmp)/'repo'; copy.mkdir()
        for folder in ('src/weight_witness', 'tests'):
            shutil.copytree(ROOT/folder, copy/folder, ignore=shutil.ignore_patterns('__pycache__'))
        for suite in sorted({row[4] for row in MUTATIONS}):
            proc = run(copy, suite)
            if proc.returncode != 0:
                raise AssertionError(f'positive control failed: {suite}\n{proc.stdout}\n{proc.stderr}')
        for name, file, old, new, suite, assertion in MUTATIONS:
            target = copy/'src/weight_witness'/file
            original = target.read_text(encoding='utf8')
            occurrences = original.count(old)
            if occurrences < 1:
                raise AssertionError(f'mutation anchor missing: {name}')
            changed = original.replace(old, new)
            if changed == original:
                raise AssertionError(f'no-op mutation: {name}')
            try:
                target.write_text(changed, encoding='utf8', newline='\n')
                shutil.rmtree(target.parent/'__pycache__', ignore_errors=True)
                proc = run(copy, suite)
                output = proc.stdout + proc.stderr
                caught = proc.returncode == 1 and f'FAIL: {assertion}' in output
                results.append({'name':name, 'changed_occurrences':occurrences, 'named_assertion':assertion, 'caught':caught})
                if not caught:
                    raise AssertionError(f'mutation did not fail its named assertion: {name}\n{output}')
            finally:
                target.write_text(original, encoding='utf8', newline='\n')
                shutil.rmtree(target.parent/'__pycache__', ignore_errors=True)
    current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/weight_witness').glob('*.py')}
    if current != baseline:
        raise AssertionError('source changed while verifying; evidence is stale')
    report = {'source_sha256':baseline, 'results':results, 'caught':sum(r['caught'] for r in results), 'total':len(results)}
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/mutation-verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
