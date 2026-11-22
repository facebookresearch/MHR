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

import importlib
import importlib.metadata
import json
import sys

import pytest

from mhr import _pymomentum_version


@pytest.mark.parametrize("version", ["0.1.86.post0", "0.1.87", "0.1.88.post0", "0.0.9"])
def test_old_version_raises_import_error_with_fix(version):
    with pytest.raises(ImportError) as excinfo:
        _pymomentum_version.check_pymomentum_version(version)

    message = str(excinfo.value)
    assert f"pymomentum {version} is installed" in message
    assert ">= 0.1.89" in message
    assert 'conda install -c conda-forge "pymomentum>=0.1.114"' in message
    assert "pip install --upgrade pymomentum-core" in message


@pytest.mark.parametrize("version", ["0.1.89", "0.1.90.post0", "0.1.95", "0.2.0", "1.0.0", "dev"])
def test_supported_or_unparsable_version_passes(version):
    _pymomentum_version.check_pymomentum_version(version)


def test_unknown_version_passes(monkeypatch):
    monkeypatch.setattr(
        _pymomentum_version, "installed_pymomentum_version", lambda: None
    )
    _pymomentum_version.check_pymomentum_version()


def _fake_metadata_version(versions):
    def version(distribution):
        if distribution in versions:
            return versions[distribution]
        raise importlib.metadata.PackageNotFoundError(distribution)

    return version


def test_version_from_pip_wheel_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        _fake_metadata_version({"pymomentum-gpu": "0.1.86.post0"}),
    )
    monkeypatch.setattr(sys, "prefix", str(tmp_path))

    assert _pymomentum_version.installed_pymomentum_version() == "0.1.86.post0"


def test_version_from_conda_package_record(monkeypatch, tmp_path):
    # conda-forge builds report 0.1.0 as the "pymomentum" distribution
    # version, so the conda package record is the source of truth.
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        _fake_metadata_version({"pymomentum": "0.1.0"}),
    )
    conda_meta = tmp_path / "conda-meta"
    conda_meta.mkdir()
    (conda_meta / "pymomentum-0.1.88-cpu_py312_h0_0.json").write_text(
        json.dumps({"name": "pymomentum", "version": "0.1.88"})
    )
    monkeypatch.setattr(sys, "prefix", str(tmp_path))

    assert _pymomentum_version.installed_pymomentum_version() == "0.1.88"


def test_version_unknown_without_records(monkeypatch, tmp_path):
    monkeypatch.setattr(importlib.metadata, "version", _fake_metadata_version({}))
    monkeypatch.setattr(sys, "prefix", str(tmp_path))

    assert _pymomentum_version.installed_pymomentum_version() is None


def test_importing_mhr_with_old_pymomentum_raises_import_error(monkeypatch):
    monkeypatch.setattr(
        _pymomentum_version, "installed_pymomentum_version", lambda: "0.1.86.post0"
    )
    monkeypatch.delitem(sys.modules, "mhr.mhr", raising=False)

    with pytest.raises(ImportError, match="requires pymomentum >= 0.1.89"):
        importlib.import_module("mhr.mhr")
