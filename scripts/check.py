"""Convenience quality-check runner."""

import subprocess
import sys


COMMANDS = [
    ["ruff", "check", "."],
    ["mypy", "src"],
    ["pytest"],
]


def main() -> int:
    for command in COMMANDS:
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
