#!/usr/bin/env python3
"""Validate required public docs, local Markdown links, and obvious placeholders."""
from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "README.md",
    "README.zh-CN.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
]
REQUIRED_HEADINGS = {
    "README.md": [
        "## Quickstart and support", "### Problem", "### Install", "### Success",
        "### Deliberate failure", "### Evidence and coverage boundaries",
        "### Support status", "### Limitations",
    ],
    "README.zh-CN.md": [
        "## 快速开始与支持状态", "### 问题", "### 安装", "### 成功命令",
        "### 故意失败命令", "### 证据与覆盖边界", "### 支持状态", "### 限制",
    ],
    "SECURITY.md": ["# Security", "## Supported versions", "## Reporting a vulnerability", "## Security boundaries"],
    "CONTRIBUTING.md": ["# Contributing", "## Before you start", "## Development setup", "## Pull requests", "## Reporting issues"],
    "CODE_OF_CONDUCT.md": ["# Code of Conduct", "## Our pledge", "## Our standards", "## Enforcement responsibilities", "## Scope", "## Enforcement", "## Attribution"],
    "CHANGELOG.md": ["# Changelog", "## Unreleased"],
    ".github/ISSUE_TEMPLATE/bug_report.md": ["# Bug report", "## Environment", "## Exact command", "## Minimal sanitized input", "## Expected result", "## Actual result", "## Installed package reproduction"],
    ".github/ISSUE_TEMPLATE/feature_request.md": ["# Feature request", "## Problem", "## Proposed behavior", "## Alternatives", "## Additional context"],
}
PLACEHOLDER = re.compile(
    r"\b(?:TODO|TBD|FIXME|XXX)\b|your[-_ ]?(?:username|repo|repository)|<your\b|\{\{[^}]+\}\}|\[insert[^\]]*\]",
    re.IGNORECASE,
)
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)(?:\s+[\"'][^)]*[\"'])?\s*\)")
REFERENCE_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)", re.MULTILINE)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


def without_fences(text: str) -> str:
    text = re.sub(r"(?ms)^```.*?^```[ \t]*$", "", text)
    return re.sub(r"(?ms)^~~~.*?^~~~[ \t]*$", "", text)


def headings(text: str) -> set[str]:
    return {f"{len(m.group(1)) * '#'} {m.group(2).strip()}" for m in HEADING.finditer(text)}


def local_target(raw: str) -> str | None:
    target = raw.strip()
    if target.startswith("<") and ">" in target:
        target = target[1:target.index(">")]
    else:
        target = target.split()[0] if target else ""
    target = urllib.parse.unquote(target)
    if not target or target.startswith("#") or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
        return None
    target = target.split("#", 1)[0].split("?", 1)[0]
    return target or None


def main() -> int:
    errors: list[str] = []
    checked_links = 0
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"{relative}: missing required file")
            continue
        text = path.read_text(encoding="utf-8")
        present = headings(text)
        for required in REQUIRED_HEADINGS.get(relative, []):
            if required not in present:
                errors.append(f"{relative}: missing heading {required!r}")
        prose = without_fences(text)
        match = PLACEHOLDER.search(prose)
        if match:
            errors.append(f"{relative}: obvious placeholder {match.group(0)!r}")
        targets = [m.group(1) for m in INLINE_LINK.finditer(prose)]
        targets.extend(m.group(1) for m in REFERENCE_DEF.finditer(prose))
        for raw in targets:
            target = local_target(raw)
            if target is None:
                continue
            checked_links += 1
            candidate = (path.parent / target).resolve()
            if not candidate.exists():
                errors.append(f"{relative}: broken local link {target!r}")
    if errors:
        for error in errors:
            print(f"docs: {error}", file=sys.stderr)
        print(f"docs: FAILED ({len(errors)} errors)", file=sys.stderr)
        return 1
    print(f"docs: OK ({len(REQUIRED_FILES)} files, {checked_links} local links)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
