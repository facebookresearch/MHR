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

"""MHR from the converted assets, without PyMomentum (`from mhr import MHR`)."""

import subprocess
import sys

from pathlib import Path

import pytest
import torch

import mhr
from mhr._model import MHR
from tests import rig_assets


def test_package_exports_the_model():
    assert mhr.MHR is MHR


def test_converted_assets_evaluate_with_gradients(tmp_path: Path) -> None:
    rig_assets.write_assets(tmp_path)
    model = MHR.from_files(tmp_path, device="cpu", lod=1)
    assert model.pose_correctives_model is None  # the test rig has none

    identity = torch.zeros(1, 45)
    identity[0, 0] = 0.5
    identity.requires_grad_()
    pose = torch.zeros(1, 204)
    pose[0, 0] = 1
    pose.requires_grad_()
    expression = torch.zeros(1, 72, requires_grad=True)
    vertices, skeleton = model(identity, pose, expression)

    torch.testing.assert_close(
        vertices, torch.tensor([[[1.0, 0.5, 0.0], [3.0, 1.0, 0.0]]])
    )
    torch.testing.assert_close(
        skeleton[0, :, :3], torch.tensor([[1.0, 0, 0], [2.0, 0, 0]])
    )
    torch.testing.assert_close(
        model.faces, torch.tensor([[0, 1, 1]], dtype=torch.int32)
    )
    vertices.sum().backward()
    assert identity.grad is not None and pose.grad is not None
    assert expression.grad is not None


def test_missing_assets_name_the_download_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        MHR.from_files(tmp_path, device="cpu", lod=3)
    message = str(excinfo.value)
    assert str(tmp_path / "lod3.npz") in message
    assert "Run mhr-download-assets --lod 3 in your working directory" in message


def test_default_folder_is_the_first_with_converted_assets(
    tmp_path: Path, monkeypatch
) -> None:
    from mhr import io as asset_io

    # The first candidate holds only the FBX model file, so it is skipped.
    fbx_only = tmp_path / "package" / "assets"
    fbx_only.mkdir(parents=True)
    (fbx_only / asset_io.MODEL_FILE_NAME).write_text("model")
    monkeypatch.setattr(asset_io, "_package_asset_folder", lambda: fbx_only)
    (tmp_path / "assets").mkdir()
    rig_assets.write_assets(tmp_path / "assets")
    monkeypatch.setenv("MHR_ASSETS_DEST", str(tmp_path))

    assert asset_io.get_default_asset_folder() == fbx_only
    model = MHR.from_files(device="cpu", lod=1)
    assert model.faces.shape == (1, 3)


def test_default_folder_contains_the_requested_lod(tmp_path: Path, monkeypatch) -> None:
    from mhr import io as asset_io

    monkeypatch.chdir(tmp_path)
    package_assets = tmp_path / "package" / "assets"
    monkeypatch.setattr(asset_io, "_package_asset_folder", lambda: package_assets)
    monkeypatch.setenv(asset_io.ASSETS_DEST_ENV, "downloaded")
    downloaded_assets = tmp_path / "downloaded" / "assets"
    downloaded_assets.mkdir(parents=True)
    rig_assets.write_assets(downloaded_assets, lod=1)

    (tmp_path / "assets").mkdir()
    rig_assets.write_assets(tmp_path / "assets", lod=2)

    model = MHR.from_files(device="cpu", lod=2)
    assert model.faces.shape == (1, 3)


def test_model_moves_to_the_device(tmp_path: Path) -> None:
    rig_assets.write_assets(tmp_path)
    model = MHR.from_files(tmp_path, device="meta", lod=1)
    tensors = list(model.parameters()) + list(model.buffers())
    assert tensors and all(tensor.device.type == "meta" for tensor in tensors)


def test_pymomentum_model_subclasses_the_model() -> None:
    pymomentum_mhr = pytest.importorskip("mhr.mhr")
    assert issubclass(pymomentum_mhr.MHR, MHR)


def test_import_mhr_does_not_load_torch(tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "import mhr, mhr.download_assets\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'MHR' in dir(mhr)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path)


def test_wrong_blend_shape_count_is_rejected(tmp_path: Path) -> None:
    import numpy as np

    rig_assets.write_assets(tmp_path)
    lod = rig_assets.lod_arrays()
    lod["shape_vectors"] = lod["shape_vectors"][:100]
    np.savez_compressed(tmp_path / "lod1.npz", **lod)
    with pytest.raises(ValueError, match="has 100 blend shapes"):
        MHR.from_files(tmp_path, device="cpu", lod=1)


def test_model_runs_without_pymomentum(tmp_path: Path) -> None:
    # A fresh interpreter in which importing PyMomentum fails.
    rig_assets.write_assets(tmp_path)
    code = (
        "import sys; sys.modules['pymomentum'] = None\n"
        "import torch\n"
        "from mhr import MHR\n"
        f"model = MHR.from_files({str(tmp_path)!r}, device='cpu', lod=1)\n"
        "vertices, _ = model(torch.zeros(1, 45), torch.zeros(1, 204), None)\n"
        "assert vertices.shape == (1, 2, 3)\n"
        "assert not any(name.startswith('pymomentum') and sys.modules[name]\n"
        "               for name in sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path)
