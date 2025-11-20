#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test built wheel in a fresh virtual environment using uv."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def find_latest_wheel():
    """Find the latest wheel file in dist/."""
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("Error: dist/ directory not found. Run 'pixi run build' first.")
        sys.exit(1)

    wheels = list(dist_dir.glob("*.whl"))
    if not wheels:
        print("Error: No wheel file found in dist/. Run 'pixi run build' first.")
        sys.exit(1)

    # Sort by modification time, newest first
    latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
    return latest_wheel.absolute()


def test_wheel(wheel_path):
    """Test wheel in a fresh virtual environment."""
    print(f"Testing wheel: {wheel_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        venv_dir = tmpdir / ".venv"

        # Create virtual environment
        print("Creating fresh virtual environment with uv...")
        subprocess.run(
            ["uv", "venv", str(venv_dir)],
            check=True,
            cwd=tmpdir,
        )

        # Python executable path (Unix/Linux/macOS)
        python_exe = venv_dir / "bin" / "python"

        # Install wheel
        print(f"Installing wheel in virtual environment...")
        subprocess.run(
            ["uv", "pip", "install", str(wheel_path)],
            check=True,
            cwd=tmpdir,
            env={**os.environ, "VIRTUAL_ENV": str(venv_dir)},
        )

        # Test import
        print("Testing import...")
        result = subprocess.run(
            [
                str(python_exe),
                "-c",
                "import mhr; print(f'Successfully imported mhr version {mhr.__version__}')",
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=tmpdir,
        )

        print(result.stdout.strip())
        print("Wheel test passed!")


def main():
    """Main entry point."""
    wheel_path = find_latest_wheel()
    test_wheel(wheel_path)


if __name__ == "__main__":
    main()
