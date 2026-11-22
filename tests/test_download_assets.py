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

import argparse
import hashlib
import json
import shutil
import urllib.error
import zipfile

from pathlib import Path

import pytest

from mhr import download_assets
from mhr.download_assets import (
    _asset_url,
    _extract,
    _positive_int,
    _safe_members,
    _validate_zip,
)


# ---------------------------------------------------------------------------
# _asset_url
# ---------------------------------------------------------------------------
def test_asset_url_latest():
    assert _asset_url("facebookresearch/MHR", "latest", "assets.zip") == (
        "https://github.com/facebookresearch/MHR/releases/latest/download/assets.zip"
    )


def test_asset_url_versioned():
    assert _asset_url("facebookresearch/MHR", "v1.0.1", "assets.zip") == (
        "https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip"
    )


# ---------------------------------------------------------------------------
# _positive_int
# ---------------------------------------------------------------------------
def test_positive_int_accepts_positive():
    assert _positive_int("1") == 1
    assert _positive_int("42") == 42


@pytest.mark.parametrize("value", ["0", "-1", "-100"])
def test_positive_int_rejects_non_positive(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int(value)


def test_positive_int_rejects_non_integer():
    with pytest.raises(ValueError):
        _positive_int("abc")


# ---------------------------------------------------------------------------
# _validate_zip
# ---------------------------------------------------------------------------
def test_validate_zip_accepts_valid_archive(tmp_path):
    archive = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.txt", "hello")
    _validate_zip(archive)  # must not raise


def test_validate_zip_rejects_non_zip(tmp_path):
    archive = tmp_path / "not_a_zip.zip"
    archive.write_bytes(b"this is definitely not a zip archive")
    with pytest.raises(RuntimeError):
        _validate_zip(archive)


def test_validate_zip_rejects_corrupt_archive(tmp_path):
    archive = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.txt", "hello")
    # Corrupt the stored payload without updating the CRC so that testzip() detects it.
    data = bytearray(archive.read_bytes())
    data[data.index(b"hello")] ^= 0xFF
    archive.write_bytes(bytes(data))
    with pytest.raises(RuntimeError):
        _validate_zip(archive)


# ---------------------------------------------------------------------------
# _safe_members
# ---------------------------------------------------------------------------
def test_safe_members_accepts_normal_members(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sub/a.txt", "hi")
        zf.writestr("b.txt", "world")
    with zipfile.ZipFile(archive) as zf:
        members = list(_safe_members(zf, tmp_path))
    assert len(members) == 2


def test_safe_members_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "boom")
    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(RuntimeError):
            list(_safe_members(zf, tmp_path))


# ---------------------------------------------------------------------------
# _extract
# ---------------------------------------------------------------------------
def test_extract_full_archive(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sub/a.txt", "hello")
        zf.writestr("b.txt", "world")
    dest = tmp_path / "out"
    _extract(archive, dest, member=None, output=None)
    assert (dest / "sub" / "a.txt").read_text() == "hello"
    assert (dest / "b.txt").read_text() == "world"


def test_extract_single_member_flattens_to_dest(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("assets/mhr_model.pt", "model-bytes")
    dest = tmp_path / "out"
    _extract(archive, dest, member="assets/mhr_model.pt", output=None)
    assert (dest / "mhr_model.pt").read_text() == "model-bytes"


def test_extract_single_member_with_explicit_output(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("assets/mhr_model.pt", "model-bytes")
    output = tmp_path / "custom.pt"
    _extract(archive, tmp_path / "out", member="assets/mhr_model.pt", output=output)
    assert output.read_text() == "model-bytes"


def test_extract_missing_member_raises(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.txt", "hi")
    with pytest.raises(RuntimeError):
        _extract(archive, tmp_path / "out", member="missing.txt", output=None)


def test_extract_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "boom")
    with pytest.raises(RuntimeError):
        _extract(archive, tmp_path / "out", member=None, output=None)


# ---------------------------------------------------------------------------
# _download
# ---------------------------------------------------------------------------
class _FakeResponse:
    """Minimal response object compatible with urllib.request.urlopen."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if self._pos >= len(self._data):
            return b""
        if n < 0:
            chunk = self._data[self._pos :]
            self._pos = len(self._data)
            return chunk
        chunk = self._data[self._pos : self._pos + n]
        self._pos += n
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _no_sleep(seconds: int) -> None:
    return None


def test_download_writes_content(tmp_path, monkeypatch):
    output = tmp_path / "assets.zip"
    monkeypatch.setattr(
        download_assets.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(b"filedata"),
    )
    download_assets._download("http://example/x.zip", output, retries=2)
    assert output.read_bytes() == b"filedata"


def test_download_retries_then_succeeds(tmp_path, monkeypatch):
    output = tmp_path / "assets.zip"
    calls = {"n": 0}

    def fake_urlopen(request, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("transient failure")
        return _FakeResponse(b"okdata")

    monkeypatch.setattr(download_assets.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(download_assets.time, "sleep", _no_sleep)

    download_assets._download("http://example/x.zip", output, retries=3)

    assert output.read_bytes() == b"okdata"
    assert calls["n"] == 2


def test_download_raises_after_retries_exhausted(tmp_path, monkeypatch):
    output = tmp_path / "assets.zip"

    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("always fails")

    monkeypatch.setattr(download_assets.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(download_assets.time, "sleep", _no_sleep)

    with pytest.raises(RuntimeError):
        download_assets._download("http://example/x.zip", output, retries=3)

    assert not output.exists()


def test_download_rejects_empty_response(tmp_path, monkeypatch):
    output = tmp_path / "assets.zip"
    monkeypatch.setattr(
        download_assets.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(b""),
    )
    monkeypatch.setattr(download_assets.time, "sleep", _no_sleep)

    with pytest.raises(RuntimeError):
        download_assets._download("http://example/x.zip", output, retries=2)

    assert not output.exists()


def test_download_passes_a_timeout(tmp_path, monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["timeout"] = timeout
        return _FakeResponse(b"data")

    monkeypatch.setattr(download_assets.urllib.request, "urlopen", fake_urlopen)
    download_assets._download("http://example/x.zip", tmp_path / "x", retries=1)
    assert seen["timeout"] == download_assets.TIMEOUT_SECONDS


def test_download_does_not_retry_a_missing_file(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(request, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(download_assets.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(download_assets.time, "sleep", _no_sleep)
    with pytest.raises(RuntimeError, match="404"):
        download_assets._download("http://example/x.zip", tmp_path / "x", retries=3)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Member names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["", ".", "..", "./a.txt", "a/../b.txt", "/abs.txt", "a\\b.txt", "C:a.txt"]
)
def test_unsafe_member_names_are_rejected(tmp_path, name):
    with pytest.raises(RuntimeError, match="unsafe archive member name|escapes"):
        download_assets._member_target(tmp_path, name)


def test_nothing_is_extracted_when_one_member_is_unsafe(tmp_path):
    archive = tmp_path / "mixed.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("good.txt", "ok")
        zf.writestr("../evil.txt", "boom")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        _extract(archive, out, member=None, output=None)
    assert not (out / "good.txt").exists()
    assert not (tmp_path / "evil.txt").exists()


# ---------------------------------------------------------------------------
# Converted assets (--lod)
# ---------------------------------------------------------------------------
def _bundle(path: Path, files: dict) -> dict:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A fake release with converted bundles for LOD 1 and 6; _download copies
    from it, and the returned dict is the manifest it serves."""

    source = tmp_path / "release"
    source.mkdir()
    manifest = {
        "schema_version": 1,
        "common": _bundle(
            source / "mhr-assets-v1-common.zip",
            {"rig.npz": b"rig", "LICENSE.txt": b"l"},
        ),
        "lods": {
            "1": _bundle(source / "mhr-assets-v1-lod1.zip", {"lod1.npz": b"lod1"}),
            "6": _bundle(source / "mhr-assets-v1-lod6.zip", {"lod6.npz": b"lod6"}),
        },
    }

    def copy_download(url, output, retries, timeout=None):
        name = url.rsplit("/", 1)[-1]
        if name == "mhr-assets-v1.json":
            output.write_text(json.dumps(manifest))
        else:
            shutil.copyfile(source / name, output)

    monkeypatch.setattr(download_assets, "_download", copy_download)
    return manifest


def test_converted_lods_are_extracted_into_assets(tmp_path, release):
    dest = tmp_path / "dest"
    download_assets.main(["--dest", str(dest), "--lod", "6", "--lod", "1"])

    assert (dest / "assets" / "rig.npz").read_bytes() == b"rig"
    assert (dest / "assets" / "lod1.npz").read_bytes() == b"lod1"
    assert (dest / "assets" / "lod6.npz").read_bytes() == b"lod6"
    assert (dest / "mhr-assets-v1-lod6.zip").is_file()
    assert not [p for p in dest.iterdir() if p.name.startswith(".mhr-assets.")]


def test_missing_lod_names_the_available_ones(tmp_path, release):
    with pytest.raises(
        SystemExit, match="no converted assets for LOD 3; it has LOD 1, 6"
    ):
        download_assets.main(["--dest", str(tmp_path / "dest"), "--lod", "3"])


def test_bad_checksum_extracts_nothing(tmp_path, release):
    release["lods"]["6"]["sha256"] = "0" * 64
    dest = tmp_path / "dest"
    with pytest.raises(
        SystemExit, match="checksum mismatch for mhr-assets-v1-lod6.zip"
    ):
        download_assets.main(["--dest", str(dest), "--lod", "1", "--lod", "6"])
    assert not (dest / "assets").exists()
    assert not (dest / "mhr-assets-v1-common.zip").exists()


def test_unsafe_bundle_member_extracts_nothing(tmp_path, release):
    release["lods"]["6"] = _bundle(
        tmp_path / "release" / "mhr-assets-v1-lod6.zip", {"../lod6.npz": b"x"}
    )
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit, match="unsafe archive member name"):
        download_assets.main(["--dest", str(dest), "--lod", "1", "--lod", "6"])
    assert not (dest / "assets").exists()


@pytest.mark.parametrize(
    "entry",
    [
        {"filename": "../x.zip", "sha256": "0" * 64},
        {"filename": "assets", "sha256": "0" * 64},
        {"filename": "x.zip"},
    ],
)
def test_invalid_manifest_entry_is_rejected(tmp_path, release, entry):
    release["lods"]["6"] = entry
    with pytest.raises(SystemExit, match="invalid LOD 6 entry"):
        download_assets.main(["--dest", str(tmp_path / "dest"), "--lod", "6"])


def test_unsupported_manifest_schema_is_rejected(tmp_path, release):
    release["schema_version"] = 2
    with pytest.raises(SystemExit, match="unsupported MHR asset manifest schema"):
        download_assets.main(["--dest", str(tmp_path / "dest"), "--all-lods"])


def _fail_manifest(monkeypatch, error):
    def fail(url, output, retries, timeout=None):
        raise RuntimeError(f"failed to download {url}: {error}") from error

    monkeypatch.setattr(download_assets, "_download", fail)


def test_release_without_converted_assets_says_so(tmp_path, monkeypatch):
    _fail_manifest(
        monkeypatch, urllib.error.HTTPError("url", 404, "Not Found", None, None)
    )
    with pytest.raises(SystemExit, match="has no converted MHR assets"):
        download_assets.main(
            ["--dest", str(tmp_path), "--release", "v1.0.1", "--lod", "1"]
        )


def test_network_error_is_not_reported_as_missing_assets(tmp_path, monkeypatch):
    _fail_manifest(monkeypatch, urllib.error.URLError("timed out"))
    with pytest.raises(SystemExit) as excinfo:
        download_assets.main(["--dest", str(tmp_path), "--lod", "1"])
    assert "timed out" in str(excinfo.value)
    assert "no converted" not in str(excinfo.value)


def test_rate_limit_is_retried(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(request, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many", None, None)
        return _FakeResponse(b"data")

    monkeypatch.setattr(download_assets.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(download_assets.time, "sleep", _no_sleep)
    download_assets._download("http://example/x.zip", tmp_path / "x", retries=3)
    assert calls["n"] == 2


def test_duplicate_bundle_is_rejected(tmp_path, release):
    release["lods"]["6"] = dict(release["lods"]["1"])
    with pytest.raises(SystemExit, match="lists a bundle twice"):
        download_assets.main(
            ["--dest", str(tmp_path / "dest"), "--lod", "1", "--lod", "6"]
        )


def test_non_numeric_lod_key_is_rejected(tmp_path, release):
    release["lods"]["x"] = release["lods"]["6"]
    with pytest.raises(SystemExit, match="invalid lods entry"):
        download_assets.main(["--dest", str(tmp_path / "dest"), "--lod", "1"])


def test_failed_install_leaves_existing_assets_unchanged(tmp_path, release):
    dest = tmp_path / "dest"
    (dest / "assets" / "lod1.npz").mkdir(parents=True)
    (dest / "assets" / "rig.npz").write_bytes(b"old")
    with pytest.raises(SystemExit, match="is a directory"):
        download_assets.main(["--dest", str(dest), "--lod", "1"])
    assert (dest / "assets" / "rig.npz").read_bytes() == b"old"


@pytest.mark.parametrize(
    "option", [["--member", "assets/mhr_model.pt"], ["--archive", "other.zip"]]
)
def test_legacy_options_are_rejected_with_lod(option):
    with pytest.raises(SystemExit):
        download_assets.parse_args(["--lod", "1", *option])


def test_legacy_archive_keeps_its_assets_folder(tmp_path, monkeypatch):
    source = tmp_path / "assets.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("assets/compact_v6_1.model", "model")

    monkeypatch.setattr(
        download_assets,
        "_download",
        lambda url, output, retries, timeout=None: shutil.copyfile(source, output),
    )
    dest = tmp_path / "dest"
    download_assets.main(["--dest", str(dest)])

    assert (dest / "assets" / "compact_v6_1.model").read_text() == "model"
    assert sorted(p.name for p in dest.iterdir()) == ["assets", "assets.zip"]


@pytest.mark.parametrize("key", ["01", "\u00b2", "-1"])
def test_unusual_lod_keys_are_rejected(tmp_path, release, key):
    release["lods"][key] = release["lods"]["6"]
    with pytest.raises(SystemExit, match="invalid lods entry"):
        download_assets.main(["--dest", str(tmp_path / "dest"), "--lod", "1"])


def test_bundles_that_disagree_on_a_file_are_rejected(tmp_path, release):
    release["lods"]["6"] = _bundle(
        tmp_path / "release" / "mhr-assets-v1-lod6.zip",
        {"lod6.npz": b"lod6", "LICENSE.txt": b"different"},
    )
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit, match="bundles disagree on the contents of"):
        download_assets.main(["--dest", str(dest), "--lod", "1", "--lod", "6"])
    assert list((dest / "assets").iterdir()) == []


def test_assets_folder_may_be_a_symlink(tmp_path, release):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "assets").symlink_to(elsewhere)
    download_assets.main(["--dest", str(dest), "--lod", "1"])
    assert (elsewhere / "lod1.npz").read_bytes() == b"lod1"
    assert not [p for p in tmp_path.iterdir() if ".staging." in p.name]
