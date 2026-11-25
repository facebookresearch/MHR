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


"""MHR evaluated with PyTorch alone, from the converted assets.

`from mhr import MHR` gives this class. `mhr.mhr.MHR` subclasses it and loads
the FBX assets with PyMomentum instead.
"""

from __future__ import annotations

import os

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from ._rig_schema import PARAMETERS_PER_JOINT
from ._torch_rig import TorchCharacter
from .face_expression import FACE_EXPRESSION_NAMES
from .io import (
    POSE_CORRECTIVES_COMPONENTS_NAME,
    POSE_CORRECTIVES_SPARSE_MASK_NAME,
    RIG_FILE_NAME,
    asset_not_found_message,
    get_asset_folder_candidates,
    get_default_asset_folder,
    get_mhr_lod_path,
    get_mhr_rig_path,
    has_pose_corrective_blendshapes,
    load_pose_dirs_predictor,
)
from .utils import batch6DFromXYZ

LOD = Literal[0, 1, 2, 3, 4, 5, 6]
NUM_IDENTITY_BLENDSHAPES = 45
NUM_FACE_EXPRESSION_BLENDSHAPES = len(FACE_EXPRESSION_NAMES)


class MHRPoseCorrectivesModel(torch.nn.Module):
    """Non-linear pose correctives model."""

    def __init__(self, pose_dirs_predictor: torch.nn.Sequential) -> None:
        super().__init__()

        # Network to predict pose correctives offsets
        self.pose_dirs_predictor = pose_dirs_predictor

    def _pose_features_from_joint_params(
        self, joint_parameters: torch.Tensor
    ) -> torch.Tensor:
        """Compute pose features, input to the pose correctives network, based on joint parameters."""

        joint_euler_angles = joint_parameters.reshape(
            joint_parameters.shape[0], -1, PARAMETERS_PER_JOINT
        )[
            :, 2:, 3:6
        ]  # Extract rotations (Euler XYZ) from joint parameters, excluding the first two joints (not defining local pose)
        joint_6d_feat = batch6DFromXYZ(joint_euler_angles)
        # Setting also the elements of the matrix diagonal to 0 when there is no rotation (so everything is set to 0)
        joint_6d_feat[:, :, 0] -= 1
        joint_6d_feat[:, :, 4] -= 1
        joint_6d_feat = joint_6d_feat.flatten(1, 2)
        return joint_6d_feat

    def forward(self, joint_parameters: torch.Tensor) -> torch.Tensor:
        """Compute pose correctives given joint parameters (local per-joint transforms)."""

        pose_6d_feats = self._pose_features_from_joint_params(joint_parameters)
        pose_corrective_offsets = self.pose_dirs_predictor(pose_6d_feats).reshape(
            pose_6d_feats.shape[0], -1, 3
        )
        return pose_corrective_offsets


class MHR(torch.nn.Module):
    """MHR body model.

    It evaluates the rig with character_torch, which is a TorchCharacter for
    the converted assets, or a pymomentum.torch.character.Character in
    mhr.mhr.MHR.
    """

    def __init__(
        self,
        character_torch: torch.nn.Module,
        pose_correctives_model: MHRPoseCorrectivesModel | None,
    ) -> None:
        super().__init__()
        self.pose_correctives_model = pose_correctives_model
        self.character_torch = character_torch

    @property
    def faces(self) -> torch.Tensor:
        """Mesh triangles as vertex indices, on the model's device."""

        return self.character_torch.mesh.faces

    @staticmethod
    def from_files(
        folder: str | os.PathLike | None = None,
        device: torch.device | str = "cuda",
        lod: LOD = 1,
        wants_pose_correctives: bool = True,
    ) -> "MHR":
        """Load the converted assets (rig.npz and lod<N>.npz) from folder.

        If folder is None, this uses the first of
        mhr.io.get_asset_folder_candidates() that contains both files.
        Download the converted assets with `mhr-download-assets --lod <N>`.
        """

        if folder is None:
            searched = get_asset_folder_candidates()
            for candidate in searched:
                if (candidate / RIG_FILE_NAME).is_file() and Path(
                    get_mhr_lod_path(candidate, lod)
                ).is_file():
                    folder = candidate
                    break
            else:
                folder = get_default_asset_folder(RIG_FILE_NAME)
        else:
            folder = Path(folder)
            searched = [folder]
        device = torch.device(device)
        rig_path = Path(get_mhr_rig_path(folder))
        lod_path = Path(get_mhr_lod_path(folder, lod))
        missing = [str(path) for path in (rig_path, lod_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                asset_not_found_message(
                    missing, searched, f"mhr-download-assets --lod {lod}"
                )
            )

        character_torch = TorchCharacter.from_files(rig_path, lod_path, lod)
        num_shapes = character_torch.blend_shape.shape_vectors.shape[0]
        if num_shapes != NUM_IDENTITY_BLENDSHAPES + NUM_FACE_EXPRESSION_BLENDSHAPES:
            raise ValueError(
                f"{lod_path} has {num_shapes} blend shapes; expected "
                f"{NUM_IDENTITY_BLENDSHAPES} identity and "
                f"{NUM_FACE_EXPRESSION_BLENDSHAPES} face expression blend shapes"
            )

        pose_correctives_model = None
        if wants_pose_correctives:
            # Read each array once: an NpzFile reads an array again on every
            # access.
            with np.load(lod_path, allow_pickle=False) as lod_data:
                correctives = {
                    name: lod_data[name]
                    for name in lod_data.files
                    if name == POSE_CORRECTIVES_COMPONENTS_NAME
                }
            if has_pose_corrective_blendshapes(correctives):
                with np.load(rig_path, allow_pickle=False) as rig:
                    activations = {
                        name: rig[name]
                        for name in (
                            POSE_CORRECTIVES_SPARSE_MASK_NAME,
                            "0.sparse_indices",
                            "0.sparse_weight",
                        )
                    }
                pose_correctives_model = MHRPoseCorrectivesModel(
                    load_pose_dirs_predictor(
                        correctives,
                        activations,
                        load_with_cuda=device.type == "cuda",
                    )
                )

        return MHR(character_torch, pose_correctives_model).to(device)

    def get_num_identity_blendshapes(self) -> int:
        """Return number of identity blendshapes."""

        return NUM_IDENTITY_BLENDSHAPES

    def get_num_face_expression_blendshapes(self) -> int:
        """Return number of face expression blendshapes."""

        return NUM_FACE_EXPRESSION_BLENDSHAPES

    def forward(
        self,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor | None,
        apply_correctives: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute vertices given input parameters.

        The computation follows standard PyTorch autograd semantics: outputs
        track gradients when grad mode is enabled and relevant inputs require
        gradients.
        """

        # identity_coeffs: [b=batch_size, c=num_shape_coeff]
        # model_parameters: [b=batch_size, c=num_model_params (rigid, pose, scale)]
        # face_expr_coeffs: [b=batch_size, c=num_face_coeff]
        assert (
            len(identity_coeffs.shape) == 2
        ), f"Expected batched (n_rows >= 1) identity coeffs with {self.get_num_identity_blendshapes()} columns, got {identity_coeffs.shape}"
        if face_expr_coeffs is not None:
            # Check batch sizes of face expression coeffs and model parameters are the same
            assert (
                len(face_expr_coeffs.shape) == 2
            ), f"Expected batched (n_rows >= 1) face expressions coeffs with {self.get_num_face_expression_blendshapes()} columns, got {face_expr_coeffs.shape}"
        else:
            # Create zero padding for face expression coeffs
            face_expr_coeffs = torch.zeros(
                model_parameters.shape[0], self.get_num_face_expression_blendshapes()
            ).to(identity_coeffs)
        apply_correctives = (
            apply_correctives and self.pose_correctives_model is not None
        )

        identity_coeffs = identity_coeffs.expand(model_parameters.shape[0], -1)

        coeffs = torch.cat([identity_coeffs, face_expr_coeffs], dim=1)
        # Compute vertices in rest pose
        rest_pose = self.character_torch.blend_shape.forward(coeffs)

        # Compute joint parameters (local) and skeleton state (global)
        # We need to pass as many model parameters as the parameter transform size
        model_padding = (
            torch.zeros(
                model_parameters.shape[0],
                self.get_num_face_expression_blendshapes()
                + self.get_num_identity_blendshapes(),
            )
            .to(model_parameters)
            .requires_grad_(False)
        )
        joint_parameters = self.character_torch.model_parameters_to_joint_parameters(
            torch.concatenate((model_parameters, model_padding), axis=1)
        )
        skel_state = self.character_torch.joint_parameters_to_skeleton_state(
            joint_parameters
        )

        # Apply pose correctives
        linear_model_unposed = rest_pose
        if apply_correctives:
            linear_model_pose_correctives = self.pose_correctives_model.forward(
                joint_parameters=joint_parameters
            )
            linear_model_unposed += linear_model_pose_correctives

        # Compute vertices
        verts = self.character_torch.skin_points(
            skel_state=skel_state, rest_vertex_positions=linear_model_unposed
        )

        return verts, skel_state
