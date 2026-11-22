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

from mhr._torch_rig import TorchCharacter
from tests import rig_assets


def _character(folder: Path) -> TorchCharacter:
    rig_assets.write_assets(folder)
    return TorchCharacter.from_files(folder / "rig.npz", folder / "lod1.npz", 1)


def test_two_joint_rig_evaluates_and_has_gradients(tmp_path: Path) -> None:
    character = _character(tmp_path)

    coefficients = torch.zeros(1, 117)
    coefficients[0, 0] = 0.5  # first identity shape: +0.5 and +1.0 along y
    coefficients.requires_grad_()
    model_parameters = torch.zeros(1, rig_assets.NUM_PARAMETERS)
    model_parameters[0, 0] = 1  # translate the root by 1 along x
    model_parameters.requires_grad_()

    rest = character.blend_shape(coefficients)
    joint_parameters = character.model_parameters_to_joint_parameters(model_parameters)
    skeleton = character.joint_parameters_to_skeleton_state(joint_parameters)
    vertices = character.skin_points(skeleton, rest)

    torch.testing.assert_close(
        skeleton[0, :, :3], torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        vertices, torch.tensor([[[1.0, 0.5, 0.0], [3.0, 1.0, 0.0]]])
    )
    vertices.sum().backward()
    torch.testing.assert_close(coefficients.grad[0, 0], torch.tensor(3.0))
    torch.testing.assert_close(model_parameters.grad[0, 0], torch.tensor(2.0))


def test_assets_are_checked_against_the_schema(tmp_path: Path) -> None:
    rig_assets.write_assets(tmp_path)
    with pytest.raises(ValueError, match="wrong LOD"):
        TorchCharacter.from_files(tmp_path / "rig.npz", tmp_path / "lod1.npz", 2)


def test_rest_vertex_count_is_checked(tmp_path: Path) -> None:
    character = _character(tmp_path)
    skeleton = character.joint_parameters_to_skeleton_state(torch.zeros(1, 14))
    with pytest.raises(ValueError, match=r"expected \(\.\.\., 2, 3\) rest vertices"):
        character.skin_points(skeleton, torch.zeros(1, 3, 3))


def test_rotation_order_and_prerotation(tmp_path: Path) -> None:
    # The child joint has a 90 degree prerotation about z, and model
    # parameters 1 and 2 drive its rotations about x and y. Its bind pose is
    # translation (1, 0, 0) with that prerotation, so vertex 1 at (2, 0, 0)
    # is (0, -1, 0) in the child's frame.
    half = 0.5**0.5
    rig = rig_assets.rig_arrays()
    rig["parameter_transform"][7 + 3, 1] = 1
    rig["parameter_transform"][7 + 4, 2] = 1
    rig["joint_prerotations"][1] = [0, 0, half, half]
    rig["inverse_bind_pose"][1] = [0, 1, 0, 0, 0, -half, half, 1]
    np.savez(tmp_path / "rig.npz", **rig)
    np.savez(tmp_path / "lod1.npz", **rig_assets.lod_arrays())
    character = TorchCharacter.from_files(
        tmp_path / "rig.npz", tmp_path / "lod1.npz", 1
    )

    model_parameters = torch.zeros(2, rig_assets.NUM_PARAMETERS)
    model_parameters[1, 1:3] = torch.pi / 2
    rest = character.blend_shape(torch.zeros(2, 117))
    skeleton = character.joint_parameters_to_skeleton_state(
        character.model_parameters_to_joint_parameters(model_parameters)
    )
    vertices = character.skin_points(skeleton, rest)

    # At rest the vertices stay in place. The child rotates (0, -1, 0) by
    # 90 degrees about x to (0, 0, -1), then about y to (-1, 0, 0), then by
    # the prerotation about z to (0, -1, 0), so vertex 1 goes to (1, -1, 0).
    torch.testing.assert_close(vertices[0], torch.tensor([[0.0, 0, 0], [2, 0, 0]]))
    torch.testing.assert_close(
        vertices[1], torch.tensor([[0.0, 0, 0], [1, -1, 0]]), atol=1e-6, rtol=0
    )
