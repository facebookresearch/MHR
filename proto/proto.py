# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the TODO license found in the
# LICENSE file in the root directory of this source tree.

import os

from typing import Literal

import numpy as np
import pymomentum.geometry as pym_geometry

# import pymomentum.torch.character as gpu_character
import pymomentum.gpu_character as gpu_character

import torch

from .io import (
    get_corrective_activation_path,
    get_proto_blendshapes_path,
    get_proto_fbx_path,
    get_proto_model_path,
    has_face_expression_blendshapes,
    has_pose_corrective_blendshapes,
    load_blendshapes,
    load_pose_dirs_predictor,
)
from .utils import batch6DFromXYZ

LOD = Literal[0, 1, 2, 3, 4, 5, 6]


class PROTOLinearBlendshapeModel(torch.nn.Module):
    """Linear blendshape model used for identity and facial expressions."""

    def __init__(
        self,
        blend_shapes: torch.Tensor,
        mean_shape: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        # mean_shape: [v=nVerts, k=3]
        # blend_shapes: [c=num_coeff, v=nVerts, k=3]
        if mean_shape is not None:
            self.register_buffer(
                "mean_shape", mean_shape.to(torch.float32).detach().clone()
            )

        self.register_buffer(
            "blend_shapes", blend_shapes.to(torch.float32).detach().clone()
        )

    def forward(self, blend_coeffs: torch.Tensor) -> torch.Tensor:
        """Apply coefficients to blendshape and add the mean if needed."""

        assert (
            len(blend_coeffs.shape) == 2
        ), f"Expected batched (n_rows >= 1) blend coeffs with {self.blend_shapes.shape[0]} columns, got {blend_coeffs.shape}"

        # blend_shapes: [c=num_shape_coeff, v=nVerts, k=3]
        # blend_coeffs: [b=batch_size, c=num_shape_coeff]
        batch_size = blend_coeffs.shape[0]
        offsets = torch.einsum("cvk,bc->bvk", self.blend_shapes, blend_coeffs)
        to_return = (
            offsets
            if not hasattr(self, "mean_shape")
            else (offsets + self.mean_shape[None])
        )
        return to_return.reshape(batch_size, -1, 3)


class PROTOPoseCorrectivesModel(torch.nn.Module):
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
        joint_6d_feat[:, :, 0] -= 1  # so all 0 when no rotation.
        joint_6d_feat[:, :, 4] -= 1  # so all 0 when no rotation.
        joint_6d_feat = joint_6d_feat.flatten(1, 2)
        return joint_6d_feat

    def forward(self, joint_parameters: torch.Tensor) -> torch.Tensor:
        """Compute pose correctives given joint parameters (local per-joint transforms)."""

        pose_6d_feats = self._pose_features_from_joint_params(joint_parameters)
        pose_corrective_offsets = self.pose_dirs_predictor(pose_6d_feats).reshape(
            len(pose_6d_feats), -1, 3
        )
        return pose_corrective_offsets


class PROTO(torch.nn.Module):
    """PROTO body model."""

    def __init__(
        self,
        character: pym_geometry.Character,
        identity_model: PROTOLinearBlendshapeModel,
        face_expressions_model: PROTOLinearBlendshapeModel | None,
        pose_correctives_model: PROTOPoseCorrectivesModel | None,
    ) -> None:
        super().__init__()

        # Save linear identity/facial expression models and pose correctives model
        self.identity_model = identity_model
        self.face_expressions_model = face_expressions_model
        self.pose_correctives_model = pose_correctives_model

        # Save cpu/gpu characters
        self.character = character
        self.character_gpu = gpu_character.Character(character).to(
            identity_model.blend_shapes
        )

    @staticmethod
    def _create_model(
        character: pym_geometry.Character,
        blendshapes_path: str,
        corrective_activation_path: str | None,
        device: torch.device,
    ) -> "PROTO":
        """Create PROTO model from the given character and asset paths."""

        blendshapes_data = np.load(blendshapes_path)

        # Identity model
        mean_shape, identity_blendshapes = load_blendshapes(
            blendshapes_data, is_identity=True
        )

        identity_model = PROTOLinearBlendshapeModel(identity_blendshapes, mean_shape)
        identity_model.to(device)

        # Face expressions model
        face_expressions_model = (
            PROTOLinearBlendshapeModel(
                load_blendshapes(blendshapes_data, is_identity=False)
            )
            if has_face_expression_blendshapes(blendshapes_data)
            else None
        )
        if face_expressions_model is not None:
            face_expressions_model.to(device)

        # Pose correctives model
        pose_correctives_model = None
        has_pose_correctives = (
            has_pose_corrective_blendshapes(blendshapes_data)
            and corrective_activation_path is not None
        )
        if has_pose_correctives:
            corrective_activation_data = np.load(corrective_activation_path)
            pose_correctives_model = PROTOPoseCorrectivesModel(
                load_pose_dirs_predictor(
                    blendshapes_data,
                    corrective_activation_data,
                    load_with_cuda=device.type == "cuda",
                )
            )

        if pose_correctives_model is not None:
            pose_correctives_model.to(device)

        return PROTO(
            character, identity_model, face_expressions_model, pose_correctives_model
        )

    @staticmethod
    def from_files(
        device: torch.device = "cuda",
        lod: LOD = 1,
        wants_pose_correctives: bool = True,
    ) -> "PROTO":
        """Load character and model parameterization, and create full model."""

        # Create character
        character = pym_geometry.Character.load_fbx(
            get_proto_fbx_path(lod), get_proto_model_path()
        )

        # Create full model
        blendshapes_path = get_proto_blendshapes_path(lod)
        corrective_activation_path = (
            get_corrective_activation_path() if wants_pose_correctives else None
        )
        assert os.path.exists(
            blendshapes_path
        ), f"Blendshapes file not found at {blendshapes_path}"
        if corrective_activation_path is not None:
            assert os.path.exists(
                corrective_activation_path
            ), f"Corrective activation file not found at {corrective_activation_path}"
        return PROTO._create_model(
            character, blendshapes_path, corrective_activation_path, device
        )

    def forward(
        self,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor | None,
        apply_correctives: bool = True,
    ) -> torch.Tensor:
        """Compute vertices given input parameters."""

        assert (
            len(identity_coeffs.shape) == 2
        ), f"Expected batched (n_rows >= 1) identity coeffs with {self.identity_model.blend_shapes.shape[0]} columns, got {identity_coeffs.shape}"
        apply_face_expressions = (
            self.face_expressions_model is not None and face_expr_coeffs is not None
        )
        apply_correctives = (
            apply_correctives and self.pose_correctives_model is not None
        )

        if apply_face_expressions:
            assert (
                len(face_expr_coeffs.shape) == 2
            ), f"Expected batched (n_rows >= 1) face expressions coeffs with {self.face_expressions_model.blend_shapes.shape[0]} columns, got {face_expr_coeffs.shape}"

        # Compute identity vertices in rest pose
        identity_rest_pose = self.identity_model.forward(identity_coeffs)

        # Compute joint parameters (local) and skeleton state (global)
        joint_parameters = self.character_gpu.model_parameters_to_joint_parameters(
            model_parameters
        )
        skel_state = self.character_gpu.joint_parameters_to_skeleton_state(
            joint_parameters
        )

        # Apply face expressions
        linear_model_unposed = None
        if apply_face_expressions:
            face_expressions = self.face_expressions_model.forward(face_expr_coeffs)
            linear_model_unposed = identity_rest_pose + face_expressions

        # Apply pose correctives
        if apply_correctives:
            linear_model_pose_correctives = self.pose_correctives_model.forward(
                joint_parameters=joint_parameters
            )
            linear_model_unposed = (
                identity_rest_pose + linear_model_pose_correctives
                if linear_model_unposed is None
                else linear_model_unposed + linear_model_pose_correctives
            )

        if linear_model_unposed is None:
            # i.e. (not apply_face_expressions) and (not apply_correctives):
            linear_model_unposed = identity_rest_pose.expand(
                skel_state.shape[0], -1, -1
            )

        # Compute vertices
        verts = self.character_gpu.skin_points(
            skel_state=skel_state, rest_vertex_positions=linear_model_unposed
        )

        return verts
