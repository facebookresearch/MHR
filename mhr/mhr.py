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


import os

from pathlib import Path
from typing import Literal

import numpy as np
import pymomentum.geometry as pym_geometry

import pymomentum.torch.character as torch_character

import torch

from .io import (
    get_corrective_activation_path,
    get_default_asset_folder,
    get_mhr_blendshapes_path,
    get_mhr_fbx_path,
    get_mhr_model_path,
    has_pose_corrective_blendshapes,
    load_pose_dirs_predictor,
)
from .utils import batch6DFromXYZ

LOD = Literal[0, 1, 2, 3, 4, 5, 6]
NUM_IDENTITY_BLENDSHAPES = 45
NUM_FACE_EXPRESSION_BLENDSHAPES = 72


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
            joint_parameters.shape[0], -1, pym_geometry.PARAMETERS_PER_JOINT
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
    """MHR body model."""

    def __init__(
        self,
        character: pym_geometry.Character,
        pose_correctives_model: MHRPoseCorrectivesModel | None,
        device: torch.device,
    ) -> None:
        super().__init__()

        # Save pose correctives model
        self.pose_correctives_model = pose_correctives_model

        # Save cpu/gpu characters
        self.character = character
        # Note that this call also instantiates the identity and face expressions model
        self.character_torch = torch_character.Character(character).to(device)

    @staticmethod
    def _create_model(
        character: pym_geometry.Character,
        blendshapes_path: str,
        corrective_activation_path: str | None,
        device: torch.device,
    ) -> "MHR":
        """Create MHR model from the given character and asset paths."""

        blendshapes_data = np.load(blendshapes_path)

        # Pose correctives model
        pose_correctives_model = None
        has_pose_correctives = (
            has_pose_corrective_blendshapes(blendshapes_data)
            and corrective_activation_path is not None
        )
        if has_pose_correctives:
            corrective_activation_data = np.load(corrective_activation_path)
            pose_correctives_model = MHRPoseCorrectivesModel(
                load_pose_dirs_predictor(
                    blendshapes_data,
                    corrective_activation_data,
                    load_with_cuda=device.type == "cuda",
                )
            )

        if pose_correctives_model is not None:
            pose_correctives_model.to(device)

        return MHR(character, pose_correctives_model, device=device)

    @staticmethod
    def from_files(
        folder: Path = get_default_asset_folder(),
        device: torch.device = "cuda",
        lod: LOD = 1,
        wants_pose_correctives: bool = True,
    ) -> "MHR":
        """Load character and model parameterization, and create full model."""

        # Create character by fetching rig and model parameterization paths
        fbx_path = get_mhr_fbx_path(folder, lod)
        model_path = get_mhr_model_path(folder)
        assert os.path.exists(fbx_path), f"FBX file not found at {fbx_path}"
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
        # Load rig and model parameterization
        character = pym_geometry.Character.load_fbx(
            fbx_path, model_path, load_blendshapes=True
        )
        assert (
            character.blend_shape.shape_vectors.shape[0]
            == NUM_IDENTITY_BLENDSHAPES + NUM_FACE_EXPRESSION_BLENDSHAPES
        ), f"Expected {NUM_IDENTITY_BLENDSHAPES} identity and {NUM_FACE_EXPRESSION_BLENDSHAPES} face expression blendshapes, got {character.blend_shape.shape_vectors.shape[0]}"

        n_params = character.parameter_transform.size
        character = character.with_blend_shape(
            character.blend_shape
        )  # update parameter transform to include blendshape coefficients
        # Assert number of parameters now include blendshape coefficients
        assert character.parameter_transform.size == (
            n_params + NUM_IDENTITY_BLENDSHAPES + NUM_FACE_EXPRESSION_BLENDSHAPES
        )
        # Set parameter sets for identity / facial expressions
        set_blendshape_parameter_sets(character)

        # Retrieve correctives paths and create full model
        blendshapes_path = get_mhr_blendshapes_path(folder, lod)
        corrective_activation_path = (
            get_corrective_activation_path(folder) if wants_pose_correctives else None
        )
        assert os.path.exists(
            blendshapes_path
        ), f"Blendshapes file not found at {blendshapes_path}"
        if corrective_activation_path is not None:
            assert os.path.exists(
                corrective_activation_path
            ), f"Corrective activation file not found at {corrective_activation_path}"
        return MHR._create_model(
            character, blendshapes_path, corrective_activation_path, device
        )

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
        """Compute vertices given input parameters."""

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


def set_blendshape_parameter_sets(character: pym_geometry.Character) -> None:
    """Utility function to discriminate between identity/facial expression blendshape parameters of a character."""

    # Check number of blendshapes is as expected
    n_shapes = character.blend_shape.n_shapes
    assert n_shapes == (NUM_IDENTITY_BLENDSHAPES + NUM_FACE_EXPRESSION_BLENDSHAPES)

    # Set parameter set for identity
    identity_parameter_set = torch.zeros(
        character.parameter_transform.size, dtype=torch.bool
    )
    identity_parameter_set[-n_shapes : -n_shapes + NUM_IDENTITY_BLENDSHAPES] = True
    character.parameter_transform.add_parameter_set("identity", identity_parameter_set)

    # Set parameter set for facial expressions
    face_expression_parameter_set = torch.zeros(
        character.parameter_transform.size, dtype=torch.bool
    )
    face_expression_parameter_set[-NUM_FACE_EXPRESSION_BLENDSHAPES:] = True
    character.parameter_transform.add_parameter_set(
        "faceExpression", face_expression_parameter_set
    )
