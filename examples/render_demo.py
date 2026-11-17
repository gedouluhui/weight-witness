"""Rebuild the shareable example from actual generated checkpoint bytes."""
import argparse
import json
from pathlib import Path
from weight_witness.compare import compare_checkpoints
from weight_witness.report import render_html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('artifacts/demo'))
    parser.add_argument('--output', type=Path, default=Path('docs/demo'))
    args = parser.parse_args()
    report = compare_checkpoints(args.data/'base', args.data/'candidate')
    report['example'] = 'Synthetic weights — reproducible demo, not a trained model or a performance benchmark.'
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf8')
    (args.output/'index.html').write_text(render_html(report), encoding='utf8')
    print(json.dumps({'status': report['status'], 'summary': report['summary'], 'synthetic': True}))


if __name__ == '__main__': main()
