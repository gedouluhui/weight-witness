"""Install the built wheel into a clean venv and run outside the source tree."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    wheels = sorted((ROOT/'dist').glob('weight_witness-*.whl'))
    if len(wheels) != 1:
        raise AssertionError('expected exactly one release wheel in dist')
    wheel = wheels[0]
    with tempfile.TemporaryDirectory(prefix='weight-witness-install-') as temp:
        work = Path(temp)
        venv.EnvBuilder(with_pip=True).create(work/'env')
        python = work/'env'/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        env = {k: v for k,v in os.environ.items() if k != 'PYTHONPATH'}
        def run(*args, expected=0):
            result = subprocess.run([str(python), '-I', *map(str,args)], cwd=work, env=env, capture_output=True, text=True, encoding='utf8', timeout=120)
            if result.returncode != expected:
                raise AssertionError(f'isolated command rc={result.returncode}, expected={expected}\n{result.stdout}\n{result.stderr}')
            return result.stdout
        run('-m', 'pip', 'install', '--no-index', '--no-deps', wheel)
        run('-c', 'import importlib.util; assert importlib.util.find_spec("torch") is None; assert importlib.util.find_spec("numpy") is None; assert importlib.util.find_spec("safetensors") is None')
        package_path = run('-c', 'import weight_witness; print(weight_witness.__file__)').strip()
        if str(work/'env') not in package_path:
            raise AssertionError('import came from source rather than installed wheel')
        generator = subprocess.run([sys.executable, str(ROOT/'examples/make_demo.py'), '--output', str(work/'demo')], cwd=work, env=env, capture_output=True, text=True, timeout=30)
        if generator.returncode: raise AssertionError(generator.stderr)
        a, b, c = work/'demo/base', work/'demo/candidate', work/'demo/resharded'
        equal = json.loads(run('-m', 'weight_witness', 'diff', a, c))
        changed = json.loads(run('-m', 'weight_witness', 'diff', a, b, '--html', work/'report.html', expected=1))
        invalid = json.loads(run('-m', 'weight_witness', 'inspect', work/'demo/broken', expected=2))
        assert equal['status'] == 'equal' and changed['status'] == 'different' and invalid['status'] == 'error'
        assert 'Synthetic' not in (work/'report.html').read_text(encoding='utf8'), 'ordinary CLI report must not invent an example label'
        report = {'wheel':wheel.name, 'sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  'clean_venv':True, 'runtime_third_party_dependencies':0,
                  'outside_source_tree':True, 'isolated_python':True,
                  'equal_exit':0, 'different_exit':1, 'invalid_index_exit':2, 'html_written':True}
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/install-verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
