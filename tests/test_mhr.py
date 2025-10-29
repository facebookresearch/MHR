# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the TODO license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import get_args

import torch

from mhr.mhr import LOD, MHR


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

        n_id_blendshapes = model.identity_model.blend_shapes.shape[0]
        n_params = len(model.character_gpu.parameter_transform.parameter_names)

        coeffs = torch.rand(1, n_id_blendshapes).to(self.device)
        params = torch.rand(self.batch_size, n_params).to(self.device)

        face_coeffs = None
        if apply_face_expressions:
            n_face_expr_blendshapes = model.face_expressions_model.blend_shapes.shape[0]
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
        """Test body model loading and forward call, applying pose correctives."""

        for lod in get_args(LOD):
            mhr_model = MHR.from_files(device=self.device, lod=lod)
            res = self._instantiate_model(mhr_model)
            self.assertTrue(res.shape[0] == self.batch_size)

    def test_model_without_loading_pose_correctives(self):
        """Test body model loading and forward call, without loading pose correctives."""

        for lod in get_args(LOD):
            mhr_model = MHR.from_files(
                device=self.device, lod=lod, wants_pose_correctives=False
            )
            res = self._instantiate_model(mhr_model)
            self.assertTrue(res.shape[0] == self.batch_size)

    def test_model_without_applying_pose_correctives(self):
        """Test body model loading and forward call, without applying pose correctives."""

        for lod in get_args(LOD):
            mhr_model = MHR.from_files(device=self.device, lod=lod)
            res = self._instantiate_model(mhr_model, apply_pose_correctives=False)
            self.assertTrue(res.shape[0] == self.batch_size)

    def test_model_without_applying_pose_correctives_and_face_expr(self):
        """Test body model loading and forward call, without applying pose correctives and facial expressions."""

        for lod in get_args(LOD):
            mhr_model = MHR.from_files(device=self.device, lod=lod)
            res = self._instantiate_model(
                mhr_model, apply_face_expressions=False, apply_pose_correctives=False
            )
            self.assertTrue(res.shape[0] == self.batch_size)


if __name__ == "__main__":
    unittest.main()
