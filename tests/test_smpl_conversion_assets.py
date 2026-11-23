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

"""Tests that the SMPL-MHR conversion tool finds its asset files from any directory."""

import importlib.util
from pathlib import Path

import pytest

_TOOL_DIR = Path(__file__).resolve().parents[1] / "tools" / "mhr_smpl_conversion"


def _load_file_assets():
    spec = importlib.util.spec_from_file_location(
        "mhr_smpl_file_assets", _TOOL_DIR / "file_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _asset_paths(module) -> dict[str, str]:
    return {
        name: getattr(module, name) for name in dir(module) if name.endswith("_FILE")
    }


def test_asset_paths_resolve_outside_the_tool_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = _asset_paths(_load_file_assets())

    assert len(paths) == 7
    for name, path in paths.items():
        assert Path(
            path
        ).is_absolute(), f"{name} depends on the working directory: {path}"
        assert Path(path).is_file(), f"{name} points to a missing file: {path}"
        assert Path(path).parent == _TOOL_DIR / "assets"


@pytest.mark.parametrize("cwd", ["repo_root", "tool_dir"])
def test_asset_paths_do_not_change_with_the_working_directory(cwd, monkeypatch):
    monkeypatch.chdir(_TOOL_DIR.parents[1] if cwd == "repo_root" else _TOOL_DIR)
    paths = _asset_paths(_load_file_assets())

    assert all(Path(path).is_file() for path in paths.values())
