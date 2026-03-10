#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    udocker_root = project_root / "udocker-1.3.17"
    udocker_main = udocker_root / "udocker" / "maincmd.py"

    if not udocker_main.is_file():
        print(f"udocker main command not found: {udocker_main}", file=sys.stderr)
        return 1

    preferred_python = Path(
        os.environ.get("UDOCKER_PYTHON", "/home/alonso/anaconda3/envs/ccrcc/bin/python")
    )
    python_bin = preferred_python if preferred_python.is_file() else Path(sys.executable)

    env = os.environ.copy()
    env.setdefault("PROOT_NO_SECCOMP", "1")
    env.setdefault("UDOCKER_DIR", str(project_root / ".udocker"))

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{udocker_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(udocker_root)
    )

    command = [str(python_bin), str(udocker_main), *sys.argv[1:]]
    return subprocess.call(command, cwd=str(project_root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
