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

import zipfile

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymomentum.geometry")

from scripts.check_release_assets import (
    FLOAT_ATOL,
    check_bundle,
    compare_manifests,
    compare_npz,
)


def _write(path: Path, **arrays) -> Path:
    base = {
        "joint_offsets": np.zeros((2, 3), dtype=np.float32),
        "joint_parents": np.asarray([-1, 0], dtype=np.int32),
        "joint_names": np.asarray(["root", "child"]),
        "source_momentum_version": np.array("0.1.114"),
    }
    base.update(arrays)
    np.savez(path, **base)
    return path


def test_identical_files_match(tmp_path: Path) -> None:
    a, b = _write(tmp_path / "a.npz"), _write(tmp_path / "b.npz")
    assert compare_npz(a, b, "x") == []


def test_float_rounding_and_the_recorded_version_are_allowed(tmp_path: Path) -> None:
    near = np.full((2, 3), FLOAT_ATOL / 2, dtype=np.float32)
    other = _write(
        tmp_path / "b.npz",
        joint_offsets=near,
        source_momentum_version=np.array("0.1.95"),
    )
    assert compare_npz(_write(tmp_path / "a.npz"), other, "x") == []

    # One float32 step at 150 cm is 1.5e-5, more than FLOAT_ATOL.
    big = np.full((2, 3), 150, dtype=np.float32)
    step = np.nextafter(big, np.float32(200))
    a = _write(tmp_path / "c.npz", joint_offsets=big)
    assert compare_npz(a, _write(tmp_path / "d.npz", joint_offsets=step), "x") == []

    inf = np.full((2, 3), np.inf, dtype=np.float32)
    a = _write(tmp_path / "e.npz", joint_offsets=inf)
    assert compare_npz(a, a, "x") == []


@pytest.mark.parametrize(
    "change, problem",
    [
        ({"joint_offsets": np.full((2, 3), 1e-3, dtype=np.float32)}, "differs by up"),
        ({"joint_parents": np.asarray([-1, 1], dtype=np.int32)}, "joint_parents"),
        ({"joint_names": np.asarray(["root", "other"])}, "joint_names"),
        ({"joint_offsets": np.zeros((3, 3), dtype=np.float32)}, "expected float32"),
        ({"extra": np.zeros(1)}, "arrays differ: ['extra']"),
    ],
)
def test_differences_are_reported(tmp_path: Path, change, problem) -> None:
    a, b = _write(tmp_path / "a.npz"), _write(tmp_path / "b.npz", **change)
    problems = compare_npz(a, b, "x")
    assert len(problems) == 1 and problem in problems[0]


def test_manifest_must_list_the_same_sources_and_lods() -> None:
    expected = {"source_sha256": {"lod1.fbx": "a"}, "lods": {"1": {}, "6": {}}}
    assert compare_manifests(expected, dict(expected)) == []
    assert compare_manifests(expected, {**expected, "source_sha256": {}})
    assert compare_manifests(expected, {**expected, "lods": {"1": {}}})


def _bundle(path: Path, files: dict) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def test_bundle_must_hold_exactly_its_file_and_the_license(tmp_path: Path) -> None:
    npz = _write(tmp_path / "lod6.npz")
    data = npz.read_bytes()
    ok = _bundle(tmp_path / "ok.zip", {"lod6.npz": data, "LICENSE.txt": b"L"})
    assert check_bundle(ok, "lod6.npz", npz, b"L", tmp_path / "work") == []

    extra = _bundle(
        tmp_path / "extra.zip",
        {"lod6.npz": data, "LICENSE.txt": b"L", "other.npz": data},
    )
    assert "holds" in check_bundle(extra, "lod6.npz", npz, b"L", tmp_path)[0]

    license = _bundle(tmp_path / "lic.zip", {"lod6.npz": data, "LICENSE.txt": b"X"})
    problems = check_bundle(license, "lod6.npz", npz, b"L", tmp_path / "work")
    assert problems == ["lic.zip: LICENSE.txt differs from assets.zip"]
