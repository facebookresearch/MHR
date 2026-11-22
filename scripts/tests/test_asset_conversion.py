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
import os
import zipfile

from pathlib import Path

import numpy as np
import pytest
import torch

pym_geometry = pytest.importorskip("pymomentum.geometry")

from mhr._rig_schema import validate_rig_assets
from mhr.mhr import MHR
from scripts.convert_assets import _write_bundle, _write_npz, convert_assets

ASSETS = os.environ.get("MHR_TEST_ASSETS")
TEST_LODS = (1, 6)


def test_npz_writer_is_reproducible(tmp_path: Path) -> None:
    arrays = {"b": np.arange(5, dtype=np.float32), "a": np.asarray(["x", "y"])}
    _write_npz(tmp_path / "first.npz", arrays)
    _write_npz(tmp_path / "second.npz", dict(reversed(list(arrays.items()))))

    assert (tmp_path / "first.npz").read_bytes() == (
        tmp_path / "second.npz"
    ).read_bytes()
    with zipfile.ZipFile(tmp_path / "first.npz") as archive:
        assert archive.namelist() == ["a.npy", "b.npy"]
        assert {info.date_time for info in archive.infolist()} == {
            (1980, 1, 1, 0, 0, 0)
        }
        assert {info.compress_type for info in archive.infolist()} == {
            zipfile.ZIP_STORED
        }
    with np.load(tmp_path / "first.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["b"], arrays["b"])


def test_bundles_use_zip64(tmp_path: Path) -> None:
    # lod0.npz is 2.8 GB, and zipfile streams a member of 2 GiB or more only
    # with ZIP64.
    source = tmp_path / "lod0.npz"
    source.write_bytes(b"data")
    _write_bundle(tmp_path / "bundle.zip", [source])
    with zipfile.ZipFile(tmp_path / "bundle.zip") as archive:
        (info,) = archive.infolist()
        assert info.extract_version >= zipfile.ZIP64_VERSION
        assert archive.read("lod0.npz") == b"data"


@pytest.mark.parametrize("lods", [[], [7], [-1]])
def test_invalid_lods_are_rejected(tmp_path: Path, lods) -> None:
    with pytest.raises(ValueError, match="LODs must be selected from 0 through 6"):
        convert_assets(tmp_path, tmp_path / "out", lods, "test")


def test_missing_source_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="compact_v6_1.model"):
        convert_assets(tmp_path, tmp_path / "out", [1], "test")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(ASSETS is None, reason="set MHR_TEST_ASSETS to the asset folder")
def test_real_assets_convert_reproducibly_and_match_pymomentum(tmp_path: Path) -> None:
    assets = Path(ASSETS)
    first = convert_assets(assets, tmp_path / "first", TEST_LODS, "test")
    second = convert_assets(assets, tmp_path / "second", TEST_LODS, "test")

    manifest = json.loads(first.read_text())
    assert first.read_bytes() == second.read_bytes()
    for entry in [manifest["common"], *manifest["lods"].values()]:
        bundle = first.parent / entry["filename"]
        assert _sha256(bundle) == entry["sha256"]
        assert bundle.stat().st_size == entry["size"]
        assert bundle.read_bytes() == (second.parent / entry["filename"]).read_bytes()
    with zipfile.ZipFile(first.parent / manifest["common"]["filename"]) as archive:
        assert archive.namelist() == ["LICENSE.txt", "rig.npz"]

    with np.load(first.parent / "rig.npz", allow_pickle=False) as rig:
        rig = dict(rig)
    for lod in TEST_LODS:
        lod_bundle = first.parent / manifest["lods"][str(lod)]["filename"]
        with zipfile.ZipFile(lod_bundle) as archive:
            assert archive.namelist() == ["LICENSE.txt", f"lod{lod}.npz"]
        with np.load(first.parent / f"lod{lod}.npz", allow_pickle=False) as data:
            lod_data = dict(data)
        validate_rig_assets(rig, lod_data, Path("rig.npz"), Path(f"lod{lod}.npz"), lod)

        # The reference is the runtime's own PyMomentum character.
        legacy = MHR.from_files(
            assets, device=torch.device("cpu"), lod=lod, wants_pose_correctives=False
        )
        character, torch_character = legacy.character, legacy.character_torch
        np.testing.assert_array_equal(
            lod_data["base_shape"], character.blend_shape.base_shape
        )
        np.testing.assert_array_equal(
            lod_data["shape_vectors"], character.blend_shape.shape_vectors
        )
        np.testing.assert_array_equal(lod_data["faces"], character.mesh.faces)
        np.testing.assert_array_equal(
            lod_data["skin_indices"], character.skin_weights.index
        )
        np.testing.assert_array_equal(
            lod_data["skin_weights"], character.skin_weights.weight
        )
        np.testing.assert_array_equal(
            rig["parameter_transform"],
            torch_character.parameter_transform.parameter_transform.cpu().numpy(),
        )
        assert list(rig["joint_names"]) == list(character.skeleton.joint_names)
        assert list(rig["parameter_names"]) == list(character.parameter_transform.names)
        with np.load(assets / f"corrective_blendshapes_lod{lod}.npz") as correctives:
            for name in correctives.files:
                np.testing.assert_array_equal(lod_data[name], correctives[name])
