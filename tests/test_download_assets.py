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
import urllib.error
import zipfile

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
        lambda request: _FakeResponse(b"filedata"),
    )
    download_assets._download("http://example/x.zip", output, retries=2)
    assert output.read_bytes() == b"filedata"


def test_download_retries_then_succeeds(tmp_path, monkeypatch):
    output = tmp_path / "assets.zip"
    calls = {"n": 0}

    def fake_urlopen(request):
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

    def fake_urlopen(request):
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
        lambda request: _FakeResponse(b""),
    )
    monkeypatch.setattr(download_assets.time, "sleep", _no_sleep)

    with pytest.raises(RuntimeError):
        download_assets._download("http://example/x.zip", output, retries=2)

    assert not output.exists()
