"""CLI with machine-readable outcomes and source-overwrite guards."""
import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from . import __version__
from .checkpoint import FormatError, load_checkpoint
from .compare import compare_checkpoints, inspect_checkpoint
from .report import render_html


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise FormatError('arguments: ' + message)


def _parser():
    parser = _Parser(prog='weight-witness', description='Offline safetensors shard audit and checkpoint diff; no model loading.')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='validate shards, hash payloads and scan supported floats')
    inspect.add_argument('input', type=Path)
    diff = commands.add_parser('diff', help='compare tensors by name, independent of shard layout')
    diff.add_argument('left', type=Path)
    diff.add_argument('right', type=Path)
    diff.add_argument('--atol', type=float, default=0.0, help='absolute tolerance (default: exact)')
    diff.add_argument('--rtol', type=float, default=0.0, help='relative tolerance; left is reference')
    for command in (inspect, diff):
        command.add_argument('--chunk-bytes', type=int, default=1048576, help='streaming chunk size; default 1 MiB, maximum 64 MiB')
        command.add_argument('--json', dest='json_path', type=Path, help='also save JSON to this file')
        command.add_argument('--html', dest='html_path', type=Path, help='save a self-contained offline HTML report')
    return parser


def _destinations(args, inputs):
    protected = set()
    for source in inputs:
        cp = load_checkpoint(source)
        protected.update(p.resolve() for p in cp.fingerprints)
    destinations = []
    for path in (args.json_path, args.html_path):
        if path is None:
            continue
        if path.is_symlink():
            raise FormatError('output symlinks are rejected')
        resolved = path.resolve()
        if resolved in protected or any(resolved.exists() and os.path.samefile(resolved, p) for p in protected):
            raise FormatError('output would overwrite a checkpoint source or index')
        if resolved in destinations or any(resolved.exists() and p.exists() and os.path.samefile(resolved, p) for p in destinations):
            raise FormatError('JSON and HTML outputs must be different files')
        if not resolved.parent.is_dir() or (resolved.exists() and not resolved.is_file()):
            raise FormatError('output parent must exist and destination must be a regular file')
        destinations.append(resolved)


def _write_outputs(outputs):
    """Publish paired reports with best-effort rollback on ordinary I/O failures.

    Each output is staged and backed up in a transaction directory on its own
    filesystem. A failed publication restores earlier outputs, but the pair is
    never atomically visible: cross-filesystem atomic rename is not promised.
    If restoration itself fails, transaction directories are retained so the
    original reports remain available for manual recovery.
    """
    transactions = []
    try:
        for path, content in outputs:
            if path is None:
                continue
            target = Path(path).resolve()
            directory = Path(tempfile.mkdtemp(dir=target.parent, prefix='.weight-witness-'))
            state = {'directory': directory, 'target': target, 'staged': directory / 'new',
                     'backup': directory / 'old', 'backed_up': False,
                     'prepared': False, 'published': False}
            transactions.append(state)
            with state['staged'].open('w', encoding='utf8', newline='\n') as stream:
                stream.write(content)
            # No user-controlled file is opened in write/truncate mode.
        for state in transactions:
            if state['target'].exists() or state['target'].is_symlink():
                os.replace(state['target'], state['backup'])
                state['backed_up'] = True
            state['prepared'] = True
        for state in transactions:
            os.replace(state['staged'], state['target'])
            state['published'] = True
    except BaseException as exc:
        rollback_errors = []
        for state in reversed(transactions):
            if not state['prepared']:
                continue
            try:
                if state['backed_up']:
                    os.replace(state['backup'], state['target'])
                elif state['published']:
                    state['target'].unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"{state['target']}: {rollback_exc}")
        if rollback_errors:
            retained = ', '.join(str(state['directory']) for state in transactions)
            raise OSError(errno.EIO,
                          f'output rollback failed ({"; ".join(rollback_errors)}); '
                          f'recovery files retained in {retained}') from exc
        for state in transactions:
            shutil.rmtree(state['directory'], ignore_errors=True)
        raise
    else:
        for state in transactions:
            shutil.rmtree(state['directory'], ignore_errors=True)


def main(argv=None):
    args = None
    try:
        args = _parser().parse_args(argv)
        inputs = [args.input] if args.command == 'inspect' else [args.left, args.right]
        _destinations(args, inputs)
        if args.command == 'inspect':
            report = inspect_checkpoint(args.input, chunk_bytes=args.chunk_bytes)
        else:
            report = compare_checkpoints(args.left, args.right, atol=args.atol, rtol=args.rtol, chunk_bytes=args.chunk_bytes)
        data = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + '\n'
        page = render_html(report) if args.html_path is not None else ''
        # Revalidate destinations against fresh source ownership immediately
        # before writing; immutable inputs are still required against hostile races.
        _destinations(args, inputs)
        _write_outputs([(args.json_path, data), (args.html_path, page)])
        sys.stdout.write(data)
        return 1 if report['status'] in ('different', 'nonfinite') else 0
    except (FormatError, OSError, ValueError) as exc:
        message = (exc.strerror or type(exc).__name__) if isinstance(exc, OSError) else str(exc)
        sys.stdout.write(json.dumps({'schema_version': 1, 'kind': getattr(args, 'command', 'cli'), 'status': 'error',
                                    'error': message}, ensure_ascii=True, allow_nan=False) + '\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
