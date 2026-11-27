# Contributing

Contributions of focused fixes, reproducible counterexamples, and documentation improvements are welcome. Read the README support status and limitations before proposing work.

## Before you start

- Use synthetic, self-generated, or fully sanitized inputs. Never post credentials, private datasets, model weights, prompts, databases, videos, or personal filesystem paths.
- Search existing issues and pull requests before opening a duplicate.
- For a behavior change, add a failing regression test first and show the named assertion fail before the fix.
- Keep claims bounded by executable evidence. Do not invent benchmarks, users, adoption, release URLs, or compatibility results.

## Development setup

Supported version: **0.1.0**. Install and run the primary test command:

```sh
python -m pip install .
python -m unittest discover -s tests -v
python scripts/check_docs.py
```

The install and test commands above are the documented source-checkout workflow. If an optional integration dependency is unavailable, report the exact skip or failure instead of treating it as a pass.

## Pull requests

- Keep the change scoped and explain the problem, contract, and evidence.
- Preserve existing CLI/API/report semantics unless the pull request explicitly revises the contract and updates tests and documentation.
- Add positive and negative controls for security, correctness, or output-publication changes.
- Do not add runtime network access, telemetry, hidden dependencies, or unsupported compatibility claims without an explicit design discussion.
- Run the relevant tests plus `python scripts/check_docs.py`, and state any environment you could not verify.
- Contributions are provided under the repository's MIT license; third-party dependencies retain their own licenses.

## Reporting issues

Use the bug or feature template. Include Python/Node and OS versions, package version or commit, the exact command, minimal sanitized input, expected and actual results, and whether the issue reproduces from an installed package. For vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
