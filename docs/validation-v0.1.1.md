# Validation — v0.1.1

Recorded 2026-11-28 on Windows x64 / Python 3.12.3. These are local results; hosted CI is not claimed.

## Source and distribution gate

The release suite exercises real safetensors files, shard ownership, streaming comparisons, paired-output rollback, the official compatibility corpus, CLI exit codes, escaping and package guards. The current wheel and sdist are built and installed outside the checkout for smoke verification.

```text
python -m unittest discover -s tests -v
Ran 65 tests in 7.290s
OK (skipped=1)

python scripts/check_docs.py
docs: OK (8 files, 12 local links)

python -m build
```

The sole skip is the existing Windows symlink-permission case; it is not counted as a pass.

## Installed wheel gate

The 0.1.1 wheel was installed into a separate target outside the checkout; imports resolved only to that target. The installed synthetic demo generated 12-tensor base, resharded, candidate and broken-index controls. The installed `diff` command reported all 12 tensors equal between base and resharded layouts with exit 0.

## Release boundaries

A CLEAR result covers only the declared payload write set and compared bytes. It does not prove package trustworthiness, model quality, provenance, installed-environment safety or absence of malware. Historical v0.1.0 evidence remains in [VERIFICATION.md](VERIFICATION.md).
