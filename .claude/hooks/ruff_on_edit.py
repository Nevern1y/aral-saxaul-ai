#!/usr/bin/env python3
"""PostToolUse hook: format + lint a just-edited Python file with ruff.

Reads the Claude Code hook payload from stdin, extracts the edited file path,
and (only for .py files) runs `ruff format` then `ruff check --fix`. Non-blocking:
PostToolUse cannot block, and any ruff finding is surfaced as context, never an
error that interrupts the session. Silently no-ops if ruff isn't installed.

Written in Python (not bash+jq) on purpose: this is a Windows/PowerShell box and
the jq-based examples in the docs are fragile here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool_input = payload.get("tool_input", {}) or {}
    file_path = tool_input.get("file_path") or tool_input.get("path")
    if not file_path or not str(file_path).endswith(".py"):
        return 0

    ruff = shutil.which("ruff")
    if ruff is None:
        # Not installed (pip install -r requirements-dev.txt). Stay quiet.
        return 0

    # Format first, then autofix safe lint issues (import sorting, unused imports).
    subprocess.run([ruff, "format", file_path], capture_output=True, text=True)
    proc = subprocess.run(
        [ruff, "check", "--fix", "--exit-zero", file_path],
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "").strip()
    if out:
        # Surface remaining lint notes to the transcript without blocking.
        print(f"ruff: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
