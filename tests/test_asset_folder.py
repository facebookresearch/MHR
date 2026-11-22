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

"""Tests for finding the MHR asset folder. No real assets are needed."""

import os
from pathlib import Path

import pytest

from mhr import download_assets, io


def _make_assets(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / io.MODEL_FILE_NAME).write_text("")
    return folder


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Run with no package assets, no MHR_ASSETS_DEST and an empty working directory."""

    monkeypatch.setattr(io, "_package_asset_folder", lambda: tmp_path / "pkg" / "assets")
    monkeypatch.delenv(io.ASSETS_DEST_ENV, raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return tmp_path


def test_package_folder_wins_when_it_has_assets(isolated, monkeypatch):
    package = _make_assets(isolated / "pkg" / "assets")
    _make_assets(isolated / "work" / "assets")
    monkeypatch.setenv(io.ASSETS_DEST_ENV, str(isolated / "dest"))
    _make_assets(isolated / "dest" / "assets")

    assert io.get_default_asset_folder() == package


def test_package_folder_without_model_file_is_skipped(isolated):
    (isolated / "pkg" / "assets").mkdir(parents=True)
    work = _make_assets(isolated / "work" / "assets")

    assert io.get_default_asset_folder() == work


def test_env_dest_is_used(isolated, monkeypatch):
    monkeypatch.setenv(io.ASSETS_DEST_ENV, str(isolated / "dest"))
    dest = _make_assets(isolated / "dest" / "assets")
    _make_assets(isolated / "work" / "assets")

    assert io.get_default_asset_folder() == dest


def test_relative_env_dest_resolves_against_cwd(isolated, monkeypatch):
    monkeypatch.setenv(io.ASSETS_DEST_ENV, "downloads")
    dest = _make_assets(isolated / "work" / "downloads" / "assets")

    assert io.get_default_asset_folder() == dest


def test_cwd_is_used_when_env_dest_has_no_assets(isolated, monkeypatch):
    monkeypatch.setenv(io.ASSETS_DEST_ENV, str(isolated / "dest"))
    work = _make_assets(isolated / "work" / "assets")

    assert io.get_default_asset_folder() == work


def test_cwd_is_used_without_env(isolated):
    work = _make_assets(isolated / "work" / "assets")

    assert io.get_default_asset_folder() == work


@pytest.mark.parametrize("env_dest", [None, "", "dest"])
def test_fallback_matches_download_destination(isolated, monkeypatch, env_dest):
    if env_dest is not None:
        monkeypatch.setenv(io.ASSETS_DEST_ENV, env_dest)

    expected = (download_assets.parse_args([]).dest / "assets").absolute()
    assert io.get_default_asset_folder() == expected


def test_candidates_are_ordered_and_unique(isolated, monkeypatch):
    monkeypatch.setenv(io.ASSETS_DEST_ENV, str(isolated / "dest"))

    assert io.get_asset_folder_candidates() == [
        isolated / "pkg" / "assets",
        isolated / "dest" / "assets",
        isolated / "work" / "assets",
    ]

    monkeypatch.delenv(io.ASSETS_DEST_ENV)
    assert io.get_asset_folder_candidates() == [
        isolated / "pkg" / "assets",
        isolated / "work" / "assets",
    ]


@pytest.mark.parametrize("make_folder", [str, Path, lambda p: _PathLikeOnly(p)])
def test_path_helpers_accept_str_and_pathlike(tmp_path, make_folder):
    folder = make_folder(str(tmp_path))

    assert io.get_mhr_fbx_path(folder, 1) == str(tmp_path / "lod1.fbx")
    assert io.get_mhr_model_path(folder) == str(tmp_path / io.MODEL_FILE_NAME)
    assert io.get_mhr_blendshapes_path(folder, 2) == str(
        tmp_path / "corrective_blendshapes_lod2.npz"
    )
    assert io.get_corrective_activation_path(folder) == str(
        tmp_path / "corrective_activation.npz"
    )


class _PathLikeOnly(os.PathLike):
    def __init__(self, path: str) -> None:
        self._path = path

    def __fspath__(self) -> str:
        return self._path


@pytest.mark.parametrize("make_folder", [str, Path])
def test_from_files_reports_missing_assets_in_given_folder(tmp_path, make_folder):
    mhr_module = pytest.importorskip("mhr.mhr")
    folder = tmp_path / "my_assets"

    with pytest.raises(FileNotFoundError) as excinfo:
        mhr_module.MHR.from_files(folder=make_folder(str(folder)), device="cpu", lod=1)

    message = str(excinfo.value)
    assert str(folder / "lod1.fbx") in message
    assert str(folder / io.MODEL_FILE_NAME) in message
    assert str(folder / "corrective_blendshapes_lod1.npz") in message
    assert str(folder / "corrective_activation.npz") in message
    assert "mhr-download-assets" in message
    assert io.ASSETS_DEST_ENV in message
    assert "folder=" in message


def test_from_files_default_lists_every_searched_folder(isolated, monkeypatch):
    mhr_module = pytest.importorskip("mhr.mhr")
    monkeypatch.setenv(io.ASSETS_DEST_ENV, str(isolated / "dest"))

    with pytest.raises(FileNotFoundError) as excinfo:
        mhr_module.MHR.from_files(device="cpu", lod=1, wants_pose_correctives=False)

    message = str(excinfo.value)
    for folder in io.get_asset_folder_candidates():
        assert str(folder) in message
    assert str(isolated / "dest" / "assets" / "lod1.fbx") in message
    assert "corrective_activation.npz" not in message


def test_from_files_default_is_resolved_at_call_time(isolated, monkeypatch):
    mhr_module = pytest.importorskip("mhr.mhr")
    other = isolated / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)

    with pytest.raises(FileNotFoundError) as excinfo:
        mhr_module.MHR.from_files(device="cpu", lod=1)

    assert str(other / "assets" / "lod1.fbx") in str(excinfo.value)
