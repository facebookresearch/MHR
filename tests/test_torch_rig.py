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

from pathlib import Path

import numpy as np
import pytest
import torch

from mhr.mhr import MHR


def _write_assets(folder: Path, schema_version: int = 1) -> None:
    parameter_transform = np.zeros((14, 321), dtype=np.float32)
    parameter_transform[0, 0] = 1
    np.savez_compressed(
        folder / "rig.npz",
        schema_version=np.array(schema_version, dtype=np.int64),
        parameter_transform=parameter_transform,
        pose_parameters=np.zeros(321, dtype=np.bool_),
        rigid_parameters=np.zeros(321, dtype=np.bool_),
        scaling_parameters=np.zeros(321, dtype=np.bool_),
        active_joints=np.ones(2, dtype=np.bool_),
        parameter_names=np.asarray([f"parameter_{i}" for i in range(321)]),
        joint_offsets=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        joint_prerotations=np.asarray([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32),
        joint_parents=np.asarray([-1, 0], dtype=np.int32),
        joint_names=np.asarray(["root", "child"]),
        fk_indices=np.asarray([[1], [0]], dtype=np.int64),
        fk_sizes=np.asarray([1], dtype=np.int64),
        inverse_bind_pose=np.asarray(
            [
                [0, 0, 0, 0, 0, 0, 1, 1],
                [-1, 0, 0, 0, 0, 0, 1, 1],
            ],
            dtype=np.float32,
        ),
        limit_min=np.zeros(0, dtype=np.float32),
        limit_max=np.zeros(0, dtype=np.float32),
        limit_weight=np.zeros(0, dtype=np.float32),
        limit_parameter_index=np.zeros(0, dtype=np.int32),
    )

    skin_indices = np.zeros((2, 8), dtype=np.int32)
    skin_indices[1, 0] = 1
    skin_weights = np.zeros((2, 8), dtype=np.float32)
    skin_weights[:, 0] = 1
    shape_vectors = np.zeros((117, 2, 3), dtype=np.float32)
    shape_vectors[0, :, 1] = [1, 2]
    shape_vectors[45, :, 2] = [1, 2]
    np.savez_compressed(
        folder / "lod1.npz",
        schema_version=np.array(schema_version, dtype=np.int64),
        lod=np.array(1, dtype=np.int64),
        base_shape=np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.float32),
        shape_vectors=shape_vectors,
        faces=np.asarray([[0, 1, 1]], dtype=np.int32),
        texcoords=np.zeros((2, 2), dtype=np.float32),
        texcoord_faces=np.asarray([[0, 1, 1]], dtype=np.int32),
        skin_indices=skin_indices,
        skin_weights=skin_weights,
    )


def test_converted_assets_run_without_pymomentum(tmp_path: Path) -> None:
    _write_assets(tmp_path)
    model = MHR.from_files(tmp_path, device="cpu", lod=1)

    identity = torch.zeros(1, 45)
    identity[0, 0] = 0.5
    identity.requires_grad_()
    pose = torch.zeros(1, 204)
    pose[0, 0] = 1
    pose.requires_grad_()
    expression = torch.zeros(1, 72, requires_grad=True)
    vertices, skeleton = model(identity, pose, expression)

    torch.testing.assert_close(
        vertices,
        torch.tensor([[[1.0, 0.5, 0.0], [3.0, 1.0, 0.0]]]),
    )
    torch.testing.assert_close(
        skeleton[0, :, :3], torch.tensor([[1.0, 0, 0], [2.0, 0, 0]])
    )
    torch.testing.assert_close(
        model.faces, torch.tensor([[0, 1, 1]], dtype=torch.int32)
    )
    with pytest.warns(DeprecationWarning, match="MHR.character"):
        assert model.character.mesh.faces.shape == (1, 3)
    vertices.sum().backward()
    assert identity.grad is not None
    assert pose.grad is not None
    assert expression.grad is not None


def test_converted_asset_schema_is_checked(tmp_path: Path) -> None:
    _write_assets(tmp_path, schema_version=99)
    with pytest.raises(ValueError, match="unsupported MHR asset schema"):
        MHR.from_files(tmp_path, device="cpu", lod=1)


def test_converted_asset_lod_is_checked(tmp_path: Path) -> None:
    _write_assets(tmp_path)
    with np.load(tmp_path / "lod1.npz", allow_pickle=False) as data:
        arrays = {name: np.array(data[name], copy=True) for name in data.files}
    arrays["lod"] = np.array(2, dtype=np.int64)
    np.savez_compressed(tmp_path / "lod1.npz", **arrays)

    with pytest.raises(ValueError, match="wrong LOD"):
        MHR.from_files(tmp_path, device="cpu", lod=1)


def test_incomplete_converted_assets_are_rejected(tmp_path: Path) -> None:
    np.savez(tmp_path / "rig.npz", schema_version=np.array(1))
    with pytest.raises(FileNotFoundError, match="converted MHR assets are incomplete"):
        MHR.from_files(tmp_path, device="cpu", lod=1)
