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


import unittest

import pymomentum.geometry as pym_geometry

import torch

from mhr.mhr import LOD, MHR
from pymomentum.torch.character import BlendShapeBase


class MHRPoseCorrectivesModelDummy(torch.nn.Module):
    """Non-linear pose correctives model used for tests."""

    def __init__(self, num_verts: int) -> None:
        super().__init__()
        self.num_verts = num_verts

    def forward(self, joint_parameters: torch.Tensor) -> torch.Tensor:
        return torch.ones((joint_parameters.shape[0], self.num_verts, 3)).to(
            joint_parameters
        )


def _build_blend_shape_base(
    c: pym_geometry.Character,
) -> BlendShapeBase:
    torch.manual_seed(0)
    n_pts = c.mesh.n_vertices
    n_blend = 4
    shape_vectors = torch.rand(n_blend, n_pts, 3)
    return BlendShapeBase(shape_vectors)


class TestMHRModel(unittest.TestCase):
    """Test MHR model."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 10

    def _instantiate_model(
        self,
        model: MHR,
        apply_face_expressions: bool = True,
        apply_pose_correctives: bool = True,
    ) -> torch.Tensor:
        """Create random parameters and invoke model forward call."""

        n_id_blendshapes = model.get_num_identity_blendshapes()
        n_params = len(model.character_torch.parameter_transform.parameter_names)

        coeffs = torch.rand(1, n_id_blendshapes).to(self.device)
        params = torch.rand(self.batch_size, n_params).to(self.device)

        face_coeffs = None
        n_face_expr_blendshapes = model.get_num_face_expression_blendshapes()
        if apply_face_expressions and n_face_expr_blendshapes > 0:
            face_coeffs = torch.rand(self.batch_size, n_face_expr_blendshapes).to(
                self.device
            )

        return model(
            identity_coeffs=coeffs,
            model_parameters=params,
            face_expr_coeffs=face_coeffs,
            apply_correctives=apply_pose_correctives,
        )

    def test_model_with_pose_correctives(self):
        """Test body model construction and forward call, applying pose correctives."""

        character = pym_geometry.create_test_character(with_blendshapes=True)
        pose_correctives_model = MHRPoseCorrectivesModelDummy(character.mesh.n_vertices)
        face_expressions_model = _build_blend_shape_base(character)
        mhr_model = MHR(
            character,
            face_expressions_model,
            pose_correctives_model,
            device=self.device,
        )
        res = self._instantiate_model(mhr_model)
        self.assertTrue(res.shape[0] == self.batch_size)

    def test_model_without_loading_pose_correctives(self):
        """Test body model construction and forward call, without loading pose correctives."""

        character = pym_geometry.create_test_character(with_blendshapes=True)
        pose_correctives_model = None
        face_expressions_model = _build_blend_shape_base(character)
        mhr_model = MHR(
            character,
            face_expressions_model,
            pose_correctives_model,
            device=self.device,
        )
        res = self._instantiate_model(mhr_model)
        self.assertTrue(res.shape[0] == self.batch_size)

    def test_model_without_applying_pose_correctives(self):
        """Test body model construction and forward call, without applying pose correctives."""

        character = pym_geometry.create_test_character(with_blendshapes=True)
        pose_correctives_model = MHRPoseCorrectivesModelDummy(character.mesh.n_vertices)
        face_expressions_model = _build_blend_shape_base(character)
        mhr_model = MHR(
            character,
            face_expressions_model,
            pose_correctives_model,
            device=self.device,
        )
        res = self._instantiate_model(mhr_model, apply_pose_correctives=False)
        self.assertTrue(res.shape[0] == self.batch_size)

    def test_model_without_applying_pose_correctives_and_face_expr(self):
        """Test body model construction and forward call, without applying pose correctives and facial expressions."""

        character = pym_geometry.create_test_character(with_blendshapes=True)
        pose_correctives_model = MHRPoseCorrectivesModelDummy(character.mesh.n_vertices)
        face_expressions_model = _build_blend_shape_base(character)
        mhr_model = MHR(
            character,
            face_expressions_model,
            pose_correctives_model,
            device=self.device,
        )
        res = self._instantiate_model(
            mhr_model, apply_face_expressions=False, apply_pose_correctives=False
        )
        self.assertTrue(res.shape[0] == self.batch_size)


if __name__ == "__main__":
    unittest.main()
