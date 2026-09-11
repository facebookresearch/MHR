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

import hashlib
import json
import shutil
import zipfile

from pathlib import Path

import pytest

from mhr import download_assets
from mhr import io as asset_io


def _archive(path: Path, member: str, contents: bytes) -> str:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, contents)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_download_converted_lod(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    common = source / "mhr-assets-v1-common.zip"
    lod = source / "mhr-assets-v1-lod6.zip"
    manifest = source / "mhr-assets-v1.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "common": {
                    "filename": common.name,
                    "sha256": _archive(common, "rig.npz", b"rig"),
                },
                "lods": {
                    "6": {
                        "filename": lod.name,
                        "sha256": _archive(lod, "lod6.npz", b"lod"),
                    }
                },
            }
        )
    )

    def copy_download(url: str, output: Path, retries: int) -> None:
        del retries
        shutil.copyfile(source / url.rsplit("/", 1)[-1], output)

    monkeypatch.setattr(download_assets, "_download", copy_download)
    download_assets.main(
        [
            "--repo",
            "example/repo",
            "--release",
            "v1",
            "--dest",
            str(destination),
            "--lod",
            "6",
        ]
    )

    assert (destination / "rig.npz").read_bytes() == b"rig"
    assert (destination / "lod6.npz").read_bytes() == b"lod"


def test_download_rejects_bad_checksum(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    common = source / "mhr-assets-v1-common.zip"
    lod = source / "mhr-assets-v1-lod6.zip"
    _archive(common, "rig.npz", b"rig")
    manifest = source / "mhr-assets-v1.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "common": {"filename": common.name, "sha256": "0" * 64},
                "lods": {"6": {"filename": lod.name, "sha256": "0" * 64}},
            }
        )
    )

    def copy_download(url: str, output: Path, retries: int) -> None:
        del retries
        shutil.copyfile(source / url.rsplit("/", 1)[-1], output)

    monkeypatch.setattr(download_assets, "_download", copy_download)
    with pytest.raises(SystemExit, match="checksum mismatch"):
        download_assets.main(["--dest", str(destination), "--lod", "6"])
    assert not (destination / common.name).exists()


def test_download_legacy_archive_strips_assets_directory(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    archive = source / "assets.zip"
    _archive(archive, "assets/compact_v6_1.model", b"model")

    def copy_download(url: str, output: Path, retries: int) -> None:
        del url, retries
        shutil.copyfile(archive, output)

    monkeypatch.setattr(download_assets, "_download", copy_download)
    download_assets.main(["--dest", str(destination)])

    assert (destination / "compact_v6_1.model").read_bytes() == b"model"
    assert not (destination / "assets").exists()


def test_installed_default_uses_user_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MHR_ASSETS_DIR", raising=False)
    monkeypatch.delenv("MHR_ASSETS_DEST", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(asset_io, "__file__", str(tmp_path / "site-packages/mhr/io.py"))

    assert asset_io.get_default_asset_folder() == tmp_path / "cache/mhr/assets"
    assert download_assets.parse_args([]).dest == tmp_path / "cache/mhr/assets"
