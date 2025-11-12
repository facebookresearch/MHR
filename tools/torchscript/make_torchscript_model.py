# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Script to convert MHR model to TorchScript format.

Examples usage:
# Custom LOD and model version
buck2 run fbcode//frl/gemini/trinity_linear_model/experimental/mhr:jit_mhr -- \
    convert \
    -o "manifold://body_assets_ca/tree/mhr.pt" \
    -l 2 \

# CPU device
buck2 run fbcode//frl/gemini/trinity_linear_model/experimental/mhr:jit_mhr -- \
    convert \
    -o "manifold://body_assets_ca/tree/mhr.pt" \
"""

import argparse
import logging
import os
from typing import Literal

import torch
from mhr.mhr import MHR

LOD = Literal[0, 1, 2, 3, 4, 5, 6]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MHRDemo(MHR):
    def __init__(
        self,
        character,
        identity_model,
        face_expressions_model,
        pose_correctives_model,
    ):
        super().__init__(
            character=character,
            identity_model=identity_model,
            face_expressions_model=face_expressions_model,
            pose_correctives_model=pose_correctives_model,
        )

        # Store extra information for TorchScript
        self.joint_names_list: list[str] = character.skeleton.joint_names
        self.parameter_names_list: list[str] = (
            self.character_torch.parameter_transform.parameter_names
        )
        self.joint_parameter_mapping: torch.Tensor = (
            self.character.parameter_transform.transform
        )
        parameter_limits = torch.zeros(
            [self.character.parameter_transform.size, 2]
        ).float()
        parameter_limits[:, 0] -= 10.0
        parameter_limits[:, 1] += 10.0
        for limit in self.character.parameter_limits:
            index = limit.data.minmax.model_parameter_index
            parameter_limits[index, 0] = limit.data.minmax.min
            parameter_limits[index, 1] = limit.data.minmax.max
        self.parameter_limits: torch.Tensor = parameter_limits

        self.lbsw_index = torch.from_numpy(self.character.skin_weights.index)
        self.lbsw_weight = torch.from_numpy(self.character.skin_weights.weight)

    @staticmethod
    def from_files(
        device: torch.device = "cuda",
        lod: LOD = 1,
        wants_pose_correctives: bool = True,
    ) -> "MHRDemo":
        """Load character and model parameterization, and create MHRDemo model."""
        # Call parent's from_files to get the base model components
        base_model = MHR.from_files(
            device=device,
            lod=lod,
            wants_pose_correctives=wants_pose_correctives,
        )

        # Create MHRDemo instance with the same components
        return MHRDemo(
            character=base_model.character,
            identity_model=base_model.identity_model,
            face_expressions_model=base_model.face_expressions_model,
            pose_correctives_model=base_model.pose_correctives_model,
        )

    def forward(
        self,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor | None,
        apply_correctives: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        joint_parameters = self.character_torch.model_parameters_to_joint_parameters(
            model_parameters
        )
        skel_state = self.character_torch.joint_parameters_to_skeleton_state(
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
        verts = self.character_torch.skin_points(
            skel_state=skel_state, rest_vertex_positions=linear_model_unposed
        )

        verts_wo_pc = self.character_torch.skin_points(
            skel_state=skel_state, rest_vertex_positions=identity_rest_pose
        )

        return verts, skel_state, linear_model_unposed, verts_wo_pc

    @torch.jit.export
    def get_joint_names(self) -> list[str]:
        """Return the joint names for the skeleton.

        Returns:
            List of joint names corresponding to the skeleton joints.
        """
        return self.joint_names_list

    @torch.jit.export
    def get_parameter_transform(self) -> torch.Tensor:
        """Return the parameter transform matrix.

        Returns:
            Tensor of shape [num_joint_params, num_model_params] that transforms
            model parameters to joint parameters via matrix multiplication.
        """
        return self.joint_parameter_mapping

    @torch.jit.export
    def get_parameter_names(self) -> list[str]:
        """Return the parameter names for model parameters.

        Returns:
            List of parameter names corresponding to the model parameters dimension.
        """
        return self.parameter_names_list

    @torch.jit.export
    def get_parameter_limits(self) -> torch.Tensor:
        """Return the parameter limits for model parameters.

        Returns:
            Tensor of shape [num_model_params, 2] containing the lower and upper
            limits for each model parameter.
        """
        return self.parameter_limits

    @torch.jit.export
    def get_lbsw(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the LBS weights for the model.

        Returns:
            Tensor of shape [num_vertices, num_joint_params] containing the LBS
            weights for each vertex.
        """

        return self.lbsw_index, self.lbsw_weight


def _get_example_inputs(
    mhr_model: MHR | MHRDemo,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Generate example inputs for tracing the MHR model.

    Args:
        mhr_model: The MHR model instance
        batch_size: Batch size for example inputs
        device: Device to create tensors on

    Returns:
        Tuple of (identity_coeffs, model_parameters, face_expr_coeffs)
    """
    num_identity_coeffs = mhr_model.identity_model.blend_shapes.shape[0]
    num_model_params = len(mhr_model.character.parameter_transform.names)

    identity_coeffs = torch.zeros(batch_size, num_identity_coeffs, device=device)
    model_parameters = torch.zeros(batch_size, num_model_params, device=device)

    face_expr_coeffs = None
    if mhr_model.face_expressions_model is not None:
        num_face_expr = mhr_model.face_expressions_model.blend_shapes.shape[0]
        face_expr_coeffs = torch.zeros(batch_size, num_face_expr, device=device)
        logger.info("Model has %d face expression coefficients", num_face_expr)

    logger.info(
        "Generated example inputs: identity_coeffs=%s, model_parameters=%s",
        identity_coeffs.shape,
        model_parameters.shape,
    )

    return identity_coeffs, model_parameters, face_expr_coeffs


def _trace_mhr_model(
    mhr_model: MHR | MHRDemo,
    example_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    strict: bool = False,
) -> torch.jit.ScriptModule:
    """Trace the MHR model to create TorchScript version.

    Args:
        mhr_model: The MHR model to trace
        example_inputs: Tuple of example inputs for tracing
        strict: Whether to use strict tracing mode

    Returns:
        Traced TorchScript model
    """
    logger.info("Starting model tracing with strict=%s", strict)

    with torch.no_grad():
        traced_model = torch.jit.trace(
            mhr_model,
            example_inputs,
            strict=strict,
        )

    logger.info("Model tracing completed successfully")
    return traced_model


def _save_torchscript_model(
    traced_model: torch.jit.ScriptModule,
    output_path: str,
) -> None:
    """Save the traced TorchScript model to disk or Manifold.

    Args:
        traced_model: The traced TorchScript model
        output_path: Path where to save the model (filesystem or manifold:// URL)
    """
    # Create parent directory if output_path is a local filesystem path
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)


    torch.jit.save(traced_model, output_path)


def convert_mhr_to_torchscript(
    output_path: str,
    device: torch.device,
    lod: LOD,
    wants_pose_correctives: bool,
    batch_size: int,
    strict: bool,
    demo: bool,
) -> None:
    """Load MHR model and convert it to TorchScript format.

    Args:
        output_path: Path where to save the TorchScript model (filesystem or manifold:// URL)
        device: Device to load the model on
        lod: Level of detail for the model (0-6)
        wants_pose_correctives: Whether to include pose correctives
        batch_size: Batch size for example inputs during tracing
        strict: Whether to use strict tracing mode
        demo: Whether to create a TorchScript model for demo purpose
    """
    logger.info(
        "Loading MHR model: lod=%d, device=%s",
        lod,
        device,
    )

    if demo:
        logger.info("Creating TorchScript model for demo purpose")
        mhr_model = MHRDemo.from_files(
            device=device,
            lod=lod,
            wants_pose_correctives=wants_pose_correctives,
        )
    else:
        mhr_model = MHR.from_files(
            device=device,
            lod=lod,
            wants_pose_correctives=wants_pose_correctives,
        )

    mhr_model.eval()
    logger.info("Model loaded and set to evaluation mode")

    example_inputs = _get_example_inputs(mhr_model, batch_size, device)
    traced_model = _trace_mhr_model(mhr_model, example_inputs, strict)
    _save_torchscript_model(traced_model, output_path)

    logger.info("Conversion completed successfully")


def main() -> None:
    """Convert MHR model to TorchScript format."""
    parser = argparse.ArgumentParser(
        description="Convert MHR model to TorchScript format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="mhr_model.torchscript",
        help="Output path for the TorchScript model (supports manifold:// URLs)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        help="Device to use for model loading (cuda or cpu)",
    )
    parser.add_argument(
        "--lod",
        "-l",
        type=int,
        default=1,
        help="Level of detail (0-6)",
    )
    parser.add_argument(
        "--no-pose-correctives",
        action="store_true",
        default=False,
        help="Disable pose correctives in the model",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size for example inputs during tracing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Use strict mode for TorchScript tracing",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Create torchscript model for demo purpose",
    )

    args = parser.parse_args()

    if args.lod not in range(7):
        raise ValueError(f"LOD must be between 0 and 6, got {args.lod}")

    torch_device = torch.device(args.device)
    wants_pose_correctives = not args.no_pose_correctives

    logger.info("Starting MHR to TorchScript conversion")
    logger.info("Configuration:")
    logger.info("  Output path: %s", args.output)
    logger.info("  Device: %s", args.device)
    logger.info("  LOD: %d", args.lod)
    logger.info("  Pose correctives: %s", wants_pose_correctives)
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("  Strict mode: %s", args.strict)
    logger.info("  Is for demo: %s", args.demo)

    convert_mhr_to_torchscript(
        output_path=args.output,
        device=torch_device,
        lod=args.lod,
        wants_pose_correctives=wants_pose_correctives,
        batch_size=args.batch_size,
        strict=args.strict,
        demo=args.demo,
    )


if __name__ == "__main__":
    main()
