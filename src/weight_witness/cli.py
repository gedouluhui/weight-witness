"""CLI with machine-readable outcomes and source-overwrite guards."""
import argparse
import json
import os
from pathlib import Path
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
    # Stage all artifacts before replacing any. Each replace is atomic, but the
    # pair is not a filesystem transaction; an OS failure during replace can
    # leave an earlier output updated. Inputs are never outputs.
    staged = []
    try:
        for path, content in outputs:
            if path is None:
                continue
            path = path.resolve()
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', newline='\n', dir=path.parent,
                                             prefix='.weight-witness-', suffix='.tmp', delete=False) as stream:
                staged.append((Path(stream.name), path))
                stream.write(content)
            # No user-controlled file is opened in write/truncate mode.
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


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
