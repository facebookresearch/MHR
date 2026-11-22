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

"""Check that the installed pymomentum is new enough for MHR."""

from __future__ import annotations

import importlib.metadata
import json
import re
import sys

from pathlib import Path

# Older builds fail or give wrong vertices with the MHR assets:
# - 0.1.86 and older have no ParameterTransform.add_parameter_set.
# - 0.1.87 and 0.1.88 load the rig, but their per-vertex offsets from the
#   TorchScript reference model reach 11 cm.
# 0.1.89 is the first version that matches the reference. CI tests 0.1.114.
MIN_PYMOMENTUM_VERSION = (0, 1, 89)


def installed_pymomentum_version() -> str | None:
    """Return the installed pymomentum version, or None if it is unknown.

    A PyPI wheel record wins over a conda package record.
    """

    # PyPI wheels record their real version in their metadata.
    for distribution in ("pymomentum-cpu", "pymomentum-gpu", "pymomentum-core"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass

    # conda-forge builds record 0.1.0 in their Python metadata, so read the
    # version from the conda package record instead.
    for record in sorted(Path(sys.prefix, "conda-meta").glob("pymomentum-[0-9]*.json")):
        try:
            return str(json.loads(record.read_text())["version"])
        except (OSError, ValueError, KeyError):
            pass

    return None


def _parse_version(version: str) -> tuple[int, int, int] | None:
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def check_pymomentum_version(version: str | None = None) -> None:
    """Raise ImportError if pymomentum is older than MIN_PYMOMENTUM_VERSION.

    An unknown version passes, so that builds without version records still
    load.
    """

    if version is None:
        version = installed_pymomentum_version()
    if version is None:
        return

    parsed = _parse_version(version)
    if parsed is None or parsed >= MIN_PYMOMENTUM_VERSION:
        return

    minimum = ".".join(str(part) for part in MIN_PYMOMENTUM_VERSION)
    raise ImportError(
        f"MHR requires pymomentum >= {minimum}, but pymomentum {version} is "
        f"installed. Older versions fail to load the MHR model (0.1.86 and "
        f"older) or give wrong vertices (0.1.87 and 0.1.88). To fix it, run "
        f'conda install -c conda-forge "pymomentum>=0.1.114", or '
        f"pip install --upgrade pymomentum-core and see the MHR README for a "
        f"known issue with PyPI builds on Linux."
    )
