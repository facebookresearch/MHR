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

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from mhr.mhr import MHR
from tests.test_torch_rig import _write_assets


class MHRPoseCorrectivesModelDummy(torch.nn.Module):
    """Non-linear pose correctives model used for tests."""

    def __init__(self, num_verts: int) -> None:
        super().__init__()
        self.num_verts = num_verts

    def forward(self, joint_parameters: torch.Tensor) -> torch.Tensor:
        return torch.ones((joint_parameters.shape[0], self.num_verts, 3)).to(
            joint_parameters
        )


class TestMHRModel(unittest.TestCase):
    """Test MHR model."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 10
        self.temp_dir = TemporaryDirectory()
        _write_assets(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _model(self, with_correctives: bool) -> MHR:
        model = MHR.from_files(Path(self.temp_dir.name), device=self.device, lod=1)
        if with_correctives:
            model.pose_correctives_model = MHRPoseCorrectivesModelDummy(
                model.character_torch.mesh.rest_vertices.shape[0]
            ).to(self.device)
        return model

    def _instantiate_model(
        self,
        model: MHR,
        apply_face_expressions: bool = True,
        apply_pose_correctives: bool = True,
    ) -> torch.Tensor:
        """Create random parameters and invoke model forward call."""

        n_id_blendshapes = model.get_num_identity_blendshapes()
        # Only include rigid, pose and scaling parameters in the model parameters to be passed
        n_model_params = (
            model.character_torch.parameter_transform.parameter_transform.shape[1]
            - (n_id_blendshapes + model.get_num_face_expression_blendshapes())
        )

        coeffs = torch.rand(1, n_id_blendshapes).to(self.device)
        params = torch.rand(self.batch_size, n_model_params).to(self.device)

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

        mhr_model = self._model(with_correctives=True)
        res_verts, res_skel = self._instantiate_model(mhr_model)
        self.assertTrue(res_verts.shape[0] == self.batch_size)
        self.assertTrue(res_skel.shape[0] == self.batch_size)

    def test_model_without_loading_pose_correctives(self):
        """Test body model construction and forward call, without loading pose correctives."""

        mhr_model = self._model(with_correctives=False)
        res_verts, res_skel = self._instantiate_model(mhr_model)
        self.assertTrue(res_verts.shape[0] == self.batch_size)
        self.assertTrue(res_skel.shape[0] == self.batch_size)

    def test_model_without_applying_pose_correctives(self):
        """Test body model construction and forward call, without applying pose correctives."""

        mhr_model = self._model(with_correctives=True)
        res_verts, res_skel = self._instantiate_model(
            mhr_model, apply_pose_correctives=False
        )
        self.assertTrue(res_verts.shape[0] == self.batch_size)
        self.assertTrue(res_skel.shape[0] == self.batch_size)

    def test_model_without_applying_pose_correctives_and_face_expr(self):
        """Test body model construction and forward call, without applying pose correctives and facial expressions."""

        mhr_model = self._model(with_correctives=True)
        res_verts, res_skel = self._instantiate_model(
            mhr_model, apply_face_expressions=False, apply_pose_correctives=False
        )
        self.assertTrue(res_verts.shape[0] == self.batch_size)
        self.assertTrue(res_skel.shape[0] == self.batch_size)

    def test_model_supports_input_gradients(self):
        """Test gradients with respect to identity, pose, and expression inputs."""

        mhr_model = self._model(with_correctives=True)
        num_model_parameters = (
            mhr_model.character_torch.parameter_transform.parameter_transform.shape[1]
            - (
                mhr_model.get_num_identity_blendshapes()
                + mhr_model.get_num_face_expression_blendshapes()
            )
        )
        inputs = (
            torch.zeros(
                1,
                mhr_model.get_num_identity_blendshapes(),
                device=self.device,
                requires_grad=True,
            ),
            torch.zeros(
                1,
                num_model_parameters,
                device=self.device,
                requires_grad=True,
            ),
            torch.zeros(
                1,
                mhr_model.get_num_face_expression_blendshapes(),
                device=self.device,
                requires_grad=True,
            ),
        )

        vertices, _ = mhr_model(*inputs)
        vertices.sum().backward()

        self.assertTrue(vertices.requires_grad)
        for value in inputs:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(torch.count_nonzero(value.grad).item(), 0)


if __name__ == "__main__":
    unittest.main()
