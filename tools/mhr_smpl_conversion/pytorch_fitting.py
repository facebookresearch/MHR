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

"""PyTorch-based fitting classes for MHR and SMPL model conversion

This module provides PyTorch-based optimization tools for fitting MHR and SMPL models
to target vertex positions. It implements staged optimization approaches using PyTorch's
Adam optimizer and GPU acceleration.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.optim
from mhr.mhr import MHR
from tqdm import tqdm

from pymomentum_fitting import (
    _NUM_BODY_BLENDSHAPES,
    _NUM_HAND_BLENDSHAPES,
    _NUM_HEAD_BLENDSHAPES,
)
from utils import (
    _concat_mhr_lbs_model_parameters,
    ChunkedSequence,
    compute_edge_loss,
    compute_edge_vectors,
    compute_vertex_loss,
    evaluate_model_fitting_error,
    get_batched_parameters,
    load_head_vertex_weights,
)

logger = logging.getLogger(__name__)

# pyre-ignore[21]: Could not find module `smplx`
import smplx


# Constants for optimization
class OptimizationConstants:
    """Constants used throughout the optimization process."""

    # Frame selection and subsampling
    SUBSAMPLING_THRESHOLD_MULTIPLIER = 256
    DEFAULT_IDENTITY_FRAMES = 16
    TRACKING_THRESHOLD_MULTIPLIER = 16

    # Weight thresholds for head optimization
    HEAD_WEIGHT_THRESHOLD = 1.0
    HEAD_VERTEX_THRESHOLD = 0.0

    # Expression regularization
    EXPRESSION_REGULARIZATION_WEIGHT = 1e4
    EXPRESSION_REGULARIZATION_THRESHOLD = 0.3

    # Vertex loss weights
    HEAD_VERTEX_LOSS_WEIGHT = 100.0
    INITIAL_EDGE_WEIGHT = 1.0
    REDUCED_EDGE_WEIGHT = 0.1

    # Learning rate milestones
    SCHEDULER_MILESTONE_1 = 100
    SCHEDULER_MILESTONE_2 = 200
    SCHEDULER_GAMMA = 0.1

    # Optimization stage thresholds
    EDGE_TO_VERTEX_TRANSITION_EPOCH = 50

    # Tracking optimization parameters
    TRACKING_HEAD_FIT_ITERATIONS_STAGE_1 = 20
    TRACKING_HEAD_FIT_ITERATIONS_STAGE_2 = 40
    TRACKING_HEAD_FIT_ITERATIONS_DISABLED = 0
    TRACKING_POSE_STAGE_1_ITERATION = 4
    TRACKING_POSE_STAGE_2_ITERATION = 10
    TRACKING_POSE_STAGE_3_ITERATION = 10
    TRACKING_ALL_PARAMS_ITERATION = 80

    # Tracking learning rates
    TRACKING_HEAD_LR_STAGE_1 = 0.1
    TRACKING_HEAD_LR_STAGE_2 = 0.02
    TRACKING_POSE_LR = 0.01
    TRACKING_ALL_PARAMS_LR = 0.002

    # SMPL optimization constants
    SMPL_TRACKING_COARSE_ITERATION_1 = 4
    SMPL_TRACKING_COARSE_ITERATION_2 = 20
    SMPL_TRACKING_COARSE_ITERATION_3 = 10
    SMPL_TRACKING_FINE_ITERATIONS = 100
    SMPL_TRACKING_LR_COARSE = 0.05
    SMPL_TRACKING_LR_FINE = 0.005
    SMPL_EDGE_TO_VERTEX_TRANSITION_EPOCH = 50


class OptimizationStage(NamedTuple):
    """Configuration for a single optimization stage."""

    parameters: list[str]
    iterations: int
    learning_rate: float
    gradient_masks: dict[str, torch.Tensor]


class BasePyTorchFitting(ABC):
    """Base class for PyTorch-based model fitting.

    Provides common utilities and interface for MHR and SMPL model fitting using PyTorch.
    Subclasses must implement the fit() method for model-specific optimization logic.
    """

    def __init__(self, device: str = "cuda", batch_size: int = 256) -> None:
        """Initialize the base PyTorch fitting instance.

        Args:
            device: Device to use for computation (default: "cuda")
            batch_size: Batch size for processing frames
        """
        self._device = device
        self._batch_size = batch_size

    def _to_tensor(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Convert input data to tensor on the appropriate device.

        Args:
            data: Input data as numpy array or torch tensor

        Returns:
            Tensor on the specified device
        """
        if isinstance(data, torch.Tensor):
            return data.float().to(self._device)
        return torch.from_numpy(data).float().to(self._device)

    def _select_frames_for_identity_estimation(
        self,
        target_vertices: torch.Tensor,
        template_vertices: torch.Tensor,
        faces: np.ndarray,
        num_selected_frames: int = OptimizationConstants.DEFAULT_IDENTITY_FRAMES,
    ) -> np.ndarray:
        """Select frames for identity estimation based on edge distance.

        This method selects frames that are most similar to the template mesh
        based on edge distance, which helps in estimating a stable identity/shape.

        Args:
            target_vertices: Target vertices tensor [B, V, 3]
            template_vertices: Template mesh vertices [V, 3]
            faces: Mesh face connectivity [F, 3]
            num_selected_frames: Number of frames to select

        Returns:
            Array of selected frame indices
        """
        import trimesh

        tmp_mesh = trimesh.Trimesh(
            template_vertices.cpu().numpy(), faces, process=False
        )
        edges = tmp_mesh.edges_unique.copy()

        original_num_frames = target_vertices.shape[0]
        num_selected_frames = min(original_num_frames, num_selected_frames)
        subsampling_threshold = (
            OptimizationConstants.SUBSAMPLING_THRESHOLD_MULTIPLIER * num_selected_frames
        )
        if original_num_frames > subsampling_threshold:
            process_every_n_frames = (
                original_num_frames
                // OptimizationConstants.SUBSAMPLING_THRESHOLD_MULTIPLIER
            )
            logger.info(
                f"There are too many ({original_num_frames}) frames."
                f"Subsampling every {process_every_n_frames} frame for identity estimation."
            )
            target_vertices = target_vertices[::process_every_n_frames]
            logger.info(
                f"This leads to {target_vertices.shape[0]} frames for identity estimation."
            )

        num_frames = target_vertices.shape[0]
        # Keep errors on GPU for efficiency
        errors = torch.full(
            [num_frames], float("inf"), device=self._device, dtype=torch.float32
        )

        source_edges = template_vertices[edges[:, 1]] - template_vertices[edges[:, 0]]

        for batch_start in range(0, num_frames, self._batch_size):
            batch_end = min(batch_start + self._batch_size, num_frames)

            target_verts_batch = target_vertices[batch_start:batch_end]
            target_edge_vecs = (
                target_verts_batch[:, edges[:, 1], :]
                - target_verts_batch[:, edges[:, 0], :]
            )

            dist = torch.norm(source_edges - target_edge_vecs, dim=-1).mean(dim=1)
            errors[batch_start:batch_end] = dist

        # Single CPU transfer at the end
        selected_frame_indices = (
            torch.argsort(errors)[:num_selected_frames].cpu().numpy()
        )

        return selected_frame_indices

    @abstractmethod
    def fit(
        self,
        target_vertices: torch.Tensor,
        single_identity: bool,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Fit model to target vertices.

        This method must be implemented by subclasses to provide model-specific
        fitting logic.

        Args:
            target_vertices: Target vertices to fit to
            single_identity: Whether to use a single identity for all frames
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary containing fitted model parameters
        """
        pass


class PyTorchMHRFitting(BasePyTorchFitting):
    """Class for fitting MHR models using PyTorch optimization.

    This class provides PyTorch-based optimization for fitting MHR body models
    to target vertex positions using Adam optimizer with staged optimization.
    """

    def __init__(
        self,
        mhr_model: MHR,
        mhr_edges: torch.Tensor,
        mhr_vertex_mask: torch.Tensor,
        mhr_param_masks: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
        device: str = "cuda",
        batch_size: int = 256,
    ) -> None:
        """Initialize the PyTorch MHR fitting instance.

        Args:
            mhr_model: The MHR body model to fit
            mhr_edges: Edge connectivity for the MHR mesh
            mhr_vertex_mask: Mask for subsampled vertices
            mhr_param_masks: Dictionary of parameter masks for staged optimization
            device: Device to use for computation (default: "cuda")
            batch_size: Batch size for processing frames
        """
        super().__init__(device=device, batch_size=batch_size)
        self._mhr_model = mhr_model
        self._mhr_edges = mhr_edges
        self._mhr_vertex_mask = mhr_vertex_mask
        self._mhr_param_masks = mhr_param_masks

        # Pre-compute head optimization data for efficiency
        self._head_vertex_weights = self._load_head_vertex_weights()
        self._normalized_head_weights = (
            self._head_vertex_weights / self._head_vertex_weights.sum()
        )
        self._head_edges = self._compute_head_edges(self._head_vertex_weights)
        self._head_vertex_mask = (
            self._head_vertex_weights > OptimizationConstants.HEAD_VERTEX_THRESHOLD
        )

    def _compute_head_edges(self, head_vertex_weights: torch.Tensor) -> torch.Tensor:
        """Pre-compute head edges once at initialization."""
        return self._mhr_edges[
            head_vertex_weights[self._mhr_edges].sum(1)
            > OptimizationConstants.HEAD_WEIGHT_THRESHOLD
        ]

    @torch.no_grad()
    def _get_mhr_vertices_batch(
        self,
        variables: dict[str, torch.Tensor],
        batch_start: int,
        batch_end: int,
    ) -> torch.Tensor:
        """Get MHR vertices for a batch with minimal recomputation."""
        self._concat_mhr_parameters(variables)
        batched_variables = get_batched_parameters(
            variables, batch_start, batch_end, self._device, "mhr"
        )

        verts, _ =  self._mhr_model(
            identity_coeffs=batched_variables["identity_coeffs"],
            model_parameters=batched_variables["lbs_model_params"],
            face_expr_coeffs=batched_variables["face_expr_coeffs"],
            apply_correctives=True,
        )
        return verts

    def _compute_edge_vectors(
        self, vertices: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        """Compute edge vectors efficiently using advanced indexing."""
        return compute_edge_vectors(vertices, edges)

    def _should_disable_progress_messages(self) -> bool:
        """Check if tqdm progress messages should be disabled (nested case)."""
        return len(getattr(tqdm, "_instances", [])) > 0

    def fit(
        self,
        target_vertices: torch.Tensor,
        single_identity: bool,
        is_tracking: bool = False,
        exclude_expression: bool = False,
        known_parameters: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Fit MHR model to target vertices using PyTorch optimizer.

        Args:
            target_vertices: Target vertices in MHR mesh topology for each frame
            single_identity: Whether to use a single identity for all frames
            is_tracking: Whether to use tracking for temporal sequences
            exclude_expression: Whether to exclude expression parameters from fitting
            known_parameters: Dictionary of known parameters to keep constant

        Returns:
            Fitted MHR model parameters
        """
        num_frames = target_vertices.shape[0]
        target_vertices_centers = 0.5 * (
            target_vertices.max(dim=1, keepdim=True)[0]
            + target_vertices.min(dim=1, keepdim=True)[0]
        )
        target_vertices -= target_vertices_centers

        if known_parameters is None:
            known_parameters = {}

        optimized_params = {}
        if single_identity and is_tracking:
            # Select frames for body shape estimation.
            logger.info("Select frames for identity estimation.")
            selected_frame_indices = self._select_frames_for_identity_estimation(
                target_vertices
            )

            logger.info("Optimize the identity...")
            optimized_params = self._optimize_mhr(
                target_vertices[selected_frame_indices],
                single_identity=False,
                known_parameters=known_parameters,
                exclude_expression=exclude_expression,
            )

            fitting_errors = self._evaluate_conversion_error(
                optimized_params, target_vertices[selected_frame_indices]
            )
            weights = self._to_tensor(fitting_errors.max() / fitting_errors)
            weights = weights / weights.sum()

            with torch.no_grad():
                for parameter_name in [
                    "body_identity_coeffs",
                    "head_identity_coeffs",
                    "hand_identity_coeffs",
                    "scale_params",
                ]:
                    known_parameters[parameter_name] = (
                        weights[..., None] * optimized_params[parameter_name]
                    ).sum(0, keepdim=True)
                if "face_expr_coeffs" in known_parameters:
                    del known_parameters["face_expr_coeffs"]

            logger.info("Done optimizing the identity.")

        if is_tracking and target_vertices.shape[0] > 1 * self._batch_size:
            logger.info("Optimize parameters for each frame by tracking...")
            optimized_params = self._track(
                target_vertices,
                single_identity=single_identity,
                known_parameters=known_parameters,
                exclude_expression=exclude_expression,
            )
        else:
            logger.info("Optimize each frames...")
            optimized_params = self._optimize_mhr(
                target_vertices,
                single_identity=single_identity,
                known_parameters=known_parameters,
                exclude_expression=exclude_expression,
            )

        # Organize the parameters for output.
        self._concat_mhr_parameters(optimized_params)
        if single_identity:
            optimized_params["identity_coeffs"] = optimized_params["identity_coeffs"][
                :1
            ].expand(num_frames, -1)
        optimized_params["lbs_model_params"].detach()
        optimized_params["lbs_model_params"][:, 0:3] += (
            0.1 * target_vertices_centers[:, 0, :]
        )
        fitting_parameter_results = {
            "lbs_model_params": optimized_params["lbs_model_params"],
            "identity_coeffs": optimized_params["identity_coeffs"].detach(),
            "face_expr_coeffs": optimized_params["face_expr_coeffs"].detach(),
        }
        target_vertices += target_vertices_centers
        return fitting_parameter_results

    def _select_frames_for_identity_estimation(
        self, target_vertices: torch.Tensor, num_selected_frames: int = 16
    ) -> np.ndarray:
        """Select frames for identity estimation based on edge distance."""
        return super()._select_frames_for_identity_estimation(
            target_vertices=target_vertices,
            template_vertices=self._mhr_model.character_torch.blend_shape.base_shape,
            faces=self._mhr_model.character.mesh.faces,
            num_selected_frames=num_selected_frames,
        )

    def _evaluate_conversion_error(
        self,
        fitting_parameter_results: dict[str, torch.Tensor],
        target_vertices: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute conversion errors in average vertex distance using batch processing.

        Args:
            fitting_parameter_results: Dictionary containing MHR fitting results
            target_vertices: Target vertices tensor in MHR mesh topology [B, V, 3]

        Returns:
            Array of average vertex distance errors for each frame
        """
        return evaluate_model_fitting_error(
            model=self._mhr_model,
            parameters=fitting_parameter_results,
            target_vertices=target_vertices,
            batch_size=self._batch_size,
            device=self._device,
            model_type="mhr",
        )

    def _track(
        self,
        target_vertices: torch.Tensor,
        single_identity: bool,
        known_parameters: dict[str, torch.Tensor] | None = None,
        exclude_expression: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Track MHR model parameters for each frame.

        Args:
            target_vertices: Target vertices tensor in MHR mesh topology [B, V, 3]
            single_identity: If True, uses a single identity parameter for all frames
            known_parameters: Known parameters to be used as constant during optimization
            exclude_expression: If True, excludes facial expression parameters from fitting

        Returns:
            Dictionary containing tracked MHR model parameters
        """
        if known_parameters is None:
            known_parameters = {}

        num_frames = target_vertices.shape[0]
        chunk_iterator = ChunkedSequence(
            num_frames, self._batch_size, num_overlapping_frames=1
        )
        max_chunk_size = chunk_iterator.get_num_iterations()

        parameters = self._define_trainable_variables(
            num_frames=num_frames,
            single_identity=single_identity,
            exclude_expression=exclude_expression,
            known_variables=known_parameters,
        )

        for i in tqdm(range(max_chunk_size)):
            batch_frame_indices, last_batch_frame_indices = (
                chunk_iterator.get_frame_indices(i)
            )

            batch_vertices = target_vertices[batch_frame_indices]

            # Estimate the first frame of each chunk.
            if i == 0:
                # Initialize parameters
                batch_parameters = self._optimize_mhr(
                    batch_vertices,
                    single_identity=single_identity,
                    known_parameters=known_parameters,
                    exclude_expression=exclude_expression,
                )
            else:
                # Track the rest frames of each chunk.
                last_batch_parameters = {}
                for k, v in parameters.items():
                    if k not in known_parameters:
                        last_batch_parameters[k] = v[last_batch_frame_indices]
                head_fit_iterations = [
                    OptimizationConstants.TRACKING_HEAD_FIT_ITERATIONS_STAGE_1,
                    OptimizationConstants.TRACKING_HEAD_FIT_ITERATIONS_STAGE_2,
                ]
                if "head_identity_coeffs" in known_parameters and exclude_expression:
                    head_fit_iterations = [
                        OptimizationConstants.TRACKING_HEAD_FIT_ITERATIONS_DISABLED,
                        OptimizationConstants.TRACKING_HEAD_FIT_ITERATIONS_DISABLED,
                    ]
                batch_parameters = self._optimize_mhr(
                    batch_vertices,
                    single_identity=single_identity,
                    stage_iterations=[
                        head_fit_iterations,
                        [
                            OptimizationConstants.TRACKING_POSE_STAGE_1_ITERATION,
                            OptimizationConstants.TRACKING_POSE_STAGE_2_ITERATION,
                            OptimizationConstants.TRACKING_POSE_STAGE_3_ITERATION,
                        ],
                        [OptimizationConstants.TRACKING_ALL_PARAMS_ITERATION],
                    ],
                    known_parameters=known_parameters,
                    exclude_expression=exclude_expression,
                    initial_parameter_values=last_batch_parameters,
                    learning_rates=[
                        [
                            OptimizationConstants.TRACKING_HEAD_LR_STAGE_1,
                            OptimizationConstants.TRACKING_HEAD_LR_STAGE_2,
                        ],
                        [
                            OptimizationConstants.TRACKING_POSE_LR,
                            OptimizationConstants.TRACKING_POSE_LR,
                            OptimizationConstants.TRACKING_POSE_LR,
                        ],
                        [OptimizationConstants.TRACKING_ALL_PARAMS_LR],
                    ],
                )

            with torch.no_grad():
                for k, _ in parameters.items():
                    if k not in known_parameters:
                        parameters[k][batch_frame_indices] = batch_parameters[k]

        return parameters

    def _optimize_mhr(
        self,
        target_vertices: torch.Tensor,
        single_identity: bool,
        stage_iterations: list[list[int]] | None = None,
        known_parameters: dict[str, torch.Tensor] | None = None,
        exclude_expression: bool = False,
        initial_parameter_values: dict[str, torch.Tensor] | None = None,
        learning_rates: list[list[float]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Optimize MHR model parameters for each frame.

        Args:
            target_vertices: Target vertices tensor in MHR mesh topology [B, V, 3]
            single_identity: If True, uses a single identity parameter for all frames
            stage_iterations: Number of iterations for each optimization stage
            known_parameters: Known parameters to be used as constant during optimization
            exclude_expression: If True, excludes facial expression parameters from fitting
            initial_parameter_values: Initial values for parameters
            learning_rates: Learning rates for each optimization stage

        Returns:
            Dictionary containing optimized MHR model parameters
        """
        if stage_iterations is None:
            stage_iterations = [[20, 200], [20, 80, 80], [200]]
        if learning_rates is None:
            learning_rates = [[0.1, 0.01], [0.1, 0.1, 0.1], [0.01]]
        if known_parameters is None:
            known_parameters = {}

        num_frames = target_vertices.shape[0]

        # Stage 1: Optimize head parameters
        head_variables = self._optimize_head_parameters(
            target_vertices,
            num_frames,
            single_identity,
            exclude_expression,
            known_parameters,
            initial_parameter_values,
            stage_iterations[0],
            learning_rates[0],
        )

        for param_name in ["head_identity_coeffs", "face_expr_coeffs"]:
            if param_name not in known_parameters:
                known_parameters[param_name] = head_variables[param_name]

        # Stage 2: Initial pose optimization
        variables = self._optimize_initial_pose(
            target_vertices,
            num_frames,
            single_identity,
            exclude_expression,
            known_parameters,
            initial_parameter_values,
            stage_iterations[1],
            learning_rates[1],
        )

        # Stage 3: Optimize all parameters
        self._optimize_all_parameters(
            target_vertices,
            num_frames,
            variables,
            known_parameters,
            stage_iterations[2][0],
            learning_rates[2][0],
        )

        return variables

    def _define_trainable_variables(
        self,
        num_frames: int,
        single_identity: bool,
        known_variables: dict[str, torch.Tensor] | None = None,
        exclude_expression: bool = False,
        initial_parameter_values: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Define trainable variables for SMPL to MHR optimization."""
        rots = torch.zeros(num_frames, 3, device=self._device).requires_grad_()
        # Initialize the translation so that the center of the template body is at the origin.
        template_center = 0.5 * (
            self._mhr_model.character_torch.blend_shape.base_shape.max(0)[0]
            + self._mhr_model.character_torch.blend_shape.base_shape.min(0)[0]
        )
        transls = torch.zeros(num_frames, 3, device=self._device)
        transls -= 0.1 * template_center[None, ...]
        transls.requires_grad_()

        # Get pose parameter size from MHR character
        num_pose_param = int(
            self._mhr_model.character_torch.parameter_transform.pose_parameters.sum()
            - 6
        )  # Remove global translation and rotation

        pose_params = torch.zeros(
            num_frames,
            num_pose_param,
            device=self._device,
        )  # Joint angles
        pose_params.requires_grad_()

        # Identity related parameters: scaling and shape blendshapes
        num_identities = 1 if single_identity else num_frames
        # Scaling parameters
        num_scale_params = int(
            self._mhr_model.character_torch.parameter_transform.scaling_parameters.sum()
        )
        scale_params = torch.zeros(
            num_identities, num_scale_params, device=self._device
        )  # 10 scaling parameters
        scale_params.requires_grad_()

        # Identity blendshapes
        body_identity_coeffs = torch.zeros(
            num_identities, _NUM_BODY_BLENDSHAPES, device=self._device
        ).requires_grad_()
        head_identity_coeffs = torch.zeros(
            num_identities, _NUM_HEAD_BLENDSHAPES, device=self._device
        ).requires_grad_()
        hand_identity_coeffs = torch.zeros(
            num_identities, _NUM_HAND_BLENDSHAPES, device=self._device
        ).requires_grad_()

        # Facial expression parameters
        num_face_expr = self._mhr_model.get_num_face_expression_blendshapes()
        face_expr_coeffs = torch.zeros(num_frames, num_face_expr, device=self._device)
        if not exclude_expression:
            face_expr_coeffs.requires_grad_()

        if known_variables is None:
            known_variables = {}

        variables = {
            "rots": known_variables.get("rots", rots),
            "transls": known_variables.get("transls", transls),
            "pose_params": known_variables.get("pose_params", pose_params),
            "scale_params": known_variables.get("scale_params", scale_params),
            "body_identity_coeffs": known_variables.get(
                "body_identity_coeffs", body_identity_coeffs
            ),
            "head_identity_coeffs": known_variables.get(
                "head_identity_coeffs", head_identity_coeffs
            ),
            "hand_identity_coeffs": known_variables.get(
                "hand_identity_coeffs", hand_identity_coeffs
            ),
            "face_expr_coeffs": known_variables.get(
                "face_expr_coeffs", face_expr_coeffs
            ),
        }
        if initial_parameter_values is not None:
            with torch.no_grad():
                for k, v in initial_parameter_values.items():
                    if k not in known_variables:
                        variables[k] = (
                            v[: variables[k].shape[0]]
                            .clone()
                            .detach()
                            .requires_grad_(True)
                        )
        for k, _ in known_variables.items():
            variables[k].requires_grad_(False)
        return variables

    def _concat_mhr_parameters(
        self, separate_parameters: dict[str, torch.Tensor]
    ) -> None:
        """Concatenate separate LBS model parameters and identity coefficients into single tensors."""
        _concat_mhr_lbs_model_parameters(separate_parameters, include_identity=True)

    def _optimize_one_batch(
        self,
        batch_start: int,
        batch_end: int,
        variables: dict[str, torch.Tensor],
        target_edge_vecs: torch.Tensor,
        target_verts_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        vertex_loss_weight: float = 0.0,
        edge_loss_weight: float = 1.0,
        gradient_masks: dict[str, torch.Tensor] | None = None,
        vertex_mask: torch.Tensor | None = None,
        vertex_weights: torch.Tensor | None = None,
        customized_edge_definition: torch.Tensor | None = None,
    ) -> None:
        """Optimize one batch of MHR model parameters against target data."""
        if gradient_masks is None:
            gradient_masks = {}
        self._concat_mhr_parameters(variables)
        batched_mhr_parameters = get_batched_parameters(
            variables, batch_start, batch_end, str(self._device), "mhr"
        )

        mhr_verts, _ = self._mhr_model(
            identity_coeffs=batched_mhr_parameters["identity_coeffs"],
            model_parameters=batched_mhr_parameters["lbs_model_params"],
            face_expr_coeffs=batched_mhr_parameters["face_expr_coeffs"],
            apply_correctives=True,
        )

        if customized_edge_definition is not None:
            mhr_edge_vecs = (
                mhr_verts[:, customized_edge_definition[:, 1], :]
                - mhr_verts[:, customized_edge_definition[:, 0], :]
            )
        else:
            mhr_edge_vecs = (
                mhr_verts[:, self._mhr_edges[:, 1], :]
                - mhr_verts[:, self._mhr_edges[:, 0], :]
            )

        # Compute losses using utility functions
        edge_loss = 0.0
        vertex_loss = 0.0
        if edge_loss_weight > 0.0:
            edge_loss = edge_loss_weight * compute_edge_loss(
                mhr_edge_vecs, target_edge_vecs
            )
        if vertex_loss_weight > 0.0:
            if vertex_mask is None:
                vertex_mask = self._mhr_vertex_mask
            vertex_loss = vertex_loss_weight * compute_vertex_loss(
                mhr_verts, target_verts_batch, vertex_weights, vertex_mask
            )
        expression_regularization_loss = (
            OptimizationConstants.EXPRESSION_REGULARIZATION_WEIGHT
            * (
                (
                    torch.abs(batched_mhr_parameters["face_expr_coeffs"]).clip(
                        OptimizationConstants.EXPRESSION_REGULARIZATION_THRESHOLD
                    )
                    - OptimizationConstants.EXPRESSION_REGULARIZATION_THRESHOLD
                )
                ** 2
            ).mean()
        )

        loss = edge_loss + vertex_loss + expression_regularization_loss

        optimizer.zero_grad()
        loss.backward()

        if gradient_masks:
            for param_name, mask in gradient_masks.items():
                if param_name in variables and variables[param_name].grad is not None:
                    variables[param_name].grad *= mask[None, ...]

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    def _load_head_vertex_weights(self) -> torch.Tensor:
        """Load head vertex weights from the MHR model."""
        face_vertex_weight = load_head_vertex_weights()
        return self._to_tensor(face_vertex_weight)

    def _get_head_optimization_config(
        self,
    ) -> tuple[
        tuple[list[str], list[str]],
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    ]:
        """Get optimization configuration for head parameters.

        Returns:
            Tuple of (optimizable_parameters, parameter_masks) for each stage
        """
        optimizable_parameters = (
            ["rots"],
            [
                "rots",
                "transls",
                "head_identity_coeffs",
                "pose_params",
                "face_expr_coeffs",
            ],
        )
        parameter_masks = (
            {},
            {
                "pose_params": self._mhr_param_masks["head_pose_params"],
            },
        )
        return optimizable_parameters, parameter_masks

    def _optimize_head_batch(
        self,
        head_variables: dict[str, torch.Tensor],
        target_verts_batch: torch.Tensor,
        target_edge_vecs: torch.Tensor,
        batch_start: int,
        batch_end: int,
        known_parameters: dict[str, torch.Tensor],
        optimizable_parameters: tuple[list[str], list[str]],
        parameter_masks: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        iterations: list[int],
        learning_rates: list[float],
        head_vertex_mask: torch.Tensor,
        head_edges: torch.Tensor,
        head_vertex_weights: torch.Tensor,
    ) -> None:
        """Optimize a single batch for head parameters."""
        for stage_idx, (params, num_iters, param_mask, learn_rate) in enumerate(
            zip(optimizable_parameters, iterations, parameter_masks, learning_rates)
        ):
            vertex_loss_weight = (
                0.0 if stage_idx == 0 else OptimizationConstants.HEAD_VERTEX_LOSS_WEIGHT
            )
            optimizer = torch.optim.Adam(
                [head_variables[p] for p in params if p not in known_parameters],
                lr=learn_rate,
            )
            for _ in range(num_iters):
                self._optimize_one_batch(
                    batch_start,
                    batch_end,
                    head_variables,
                    target_edge_vecs,
                    target_verts_batch,
                    optimizer,
                    gradient_masks=param_mask,
                    edge_loss_weight=OptimizationConstants.INITIAL_EDGE_WEIGHT,
                    vertex_loss_weight=vertex_loss_weight,
                    vertex_mask=head_vertex_mask,
                    customized_edge_definition=head_edges,
                )

            if "transls" not in params:
                self._update_translation_from_head(
                    head_variables,
                    target_verts_batch,
                    batch_start,
                    batch_end,
                    head_vertex_weights,
                )

    def _optimize_head_parameters(
        self,
        target_vertices: torch.Tensor,
        num_frames: int,
        single_identity: bool,
        exclude_expression: bool,
        known_parameters: dict[str, torch.Tensor],
        initial_parameter_values: dict[str, torch.Tensor] | None,
        iterations: list[int],
        learning_rates: list[float],
    ) -> dict[str, torch.Tensor]:
        """Optimize head pose, identity, and face expression parameters."""
        disable_inner_message = self._should_disable_progress_messages()

        head_variables = self._define_trainable_variables(
            num_frames=num_frames,
            single_identity=single_identity,
            exclude_expression=exclude_expression,
            known_variables=known_parameters,
            initial_parameter_values=initial_parameter_values,
        )

        if not disable_inner_message:
            logger.info(
                "Optimize head pose, identity, and face expression parameters..."
            )

        optimizable_parameters, parameter_masks = self._get_head_optimization_config()

        for batch_start in tqdm(
            range(0, num_frames, self._batch_size),
            desc="Head pose, identity, and face expression optimization",
            disable=disable_inner_message,
        ):
            batch_end = min(batch_start + self._batch_size, num_frames)
            target_verts_batch = target_vertices[batch_start:batch_end]
            target_edge_vecs = self._compute_edge_vectors(
                target_verts_batch, self._head_edges
            )

            self._optimize_head_batch(
                head_variables,
                target_verts_batch,
                target_edge_vecs,
                batch_start,
                batch_end,
                known_parameters,
                optimizable_parameters,
                parameter_masks,
                iterations,
                learning_rates,
                self._head_vertex_mask,
                self._head_edges,
                self._head_vertex_weights,
            )

        return head_variables

    def _update_translation_from_head(
        self,
        variables: dict[str, torch.Tensor],
        target_verts_batch: torch.Tensor,
        batch_start: int,
        batch_end: int,
        head_vertex_weights: torch.Tensor,
    ) -> None:
        """Update translation based on head vertex alignment."""
        with torch.no_grad():
            head_vertex_weights = head_vertex_weights / head_vertex_weights.sum()
            mhr_verts = self._get_mhr_vertices_batch(variables, batch_start, batch_end)
            # Compute weighted translation update from head vertices
            diff = target_verts_batch - mhr_verts
            update = (diff * head_vertex_weights[None, ..., None]).sum(dim=1)
            variables["transls"][batch_start:batch_end] += update.detach() / 10.0

    def _get_initial_pose_optimization_config(
        self,
    ) -> tuple[
        tuple[list[str], list[str], list[str]],
        tuple[
            dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]
        ],
    ]:
        """Get optimization configuration for initial pose parameters.

        Returns:
            Tuple of (optimizable_parameters, parameter_masks) for each stage
        """
        optimizable_parameters = (
            ["rots"],
            ["rots", "body_identity_coeffs", "pose_params", "scale_params"],
            ["rots", "body_identity_coeffs", "pose_params", "scale_params"],
        )
        parameter_masks = (
            {},
            {
                "pose_params": self._mhr_param_masks["no_hand_param_masks"][0],
                "scale_params": self._mhr_param_masks["no_hand_param_masks"][1],
            },
            {},
        )
        return optimizable_parameters, parameter_masks

    def _optimize_initial_pose_batch(
        self,
        variables: dict[str, torch.Tensor],
        target_verts_batch: torch.Tensor,
        target_edge_vecs: torch.Tensor,
        batch_start: int,
        batch_end: int,
        known_parameters: dict[str, torch.Tensor],
        optimizable_parameters: tuple[list[str], list[str], list[str]],
        parameter_masks: tuple[
            dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]
        ],
        iterations: list[int],
        learning_rates: list[float],
    ) -> None:
        """Optimize a single batch for initial pose parameters."""
        for params, num_iters, param_mask, learn_rate in zip(
            optimizable_parameters, iterations, parameter_masks, learning_rates
        ):
            optimizer = torch.optim.Adam(
                [variables[p] for p in params if p not in known_parameters],
                lr=learn_rate,
            )
            for _ in range(num_iters):
                self._optimize_one_batch(
                    batch_start,
                    batch_end,
                    variables,
                    target_edge_vecs,
                    target_verts_batch,
                    optimizer,
                    gradient_masks=param_mask,
                )

        self._update_translation_from_vertices(
            variables, target_verts_batch, batch_start, batch_end
        )

    def _optimize_initial_pose(
        self,
        target_vertices: torch.Tensor,
        num_frames: int,
        single_identity: bool,
        exclude_expression: bool,
        known_parameters: dict[str, torch.Tensor],
        initial_parameter_values: dict[str, torch.Tensor] | None,
        iterations: list[int],
        learning_rates: list[float],
    ) -> dict[str, torch.Tensor]:
        """Optimize initial pose and body parameters."""
        disable_inner_message = self._should_disable_progress_messages()

        variables = self._define_trainable_variables(
            num_frames=num_frames,
            single_identity=single_identity,
            exclude_expression=exclude_expression,
            known_variables=known_parameters,
            initial_parameter_values=initial_parameter_values,
        )

        if not disable_inner_message:
            logger.info("Initial pose optimization...")

        (
            optimizable_parameters,
            parameter_masks,
        ) = self._get_initial_pose_optimization_config()

        for batch_start in tqdm(
            range(0, num_frames, self._batch_size),
            desc="Initial pose optimization batches",
            disable=disable_inner_message,
        ):
            batch_end = min(batch_start + self._batch_size, num_frames)
            target_verts_batch = target_vertices[batch_start:batch_end]
            target_edge_vecs = self._compute_edge_vectors(
                target_verts_batch, self._mhr_edges
            )

            self._optimize_initial_pose_batch(
                variables,
                target_verts_batch,
                target_edge_vecs,
                batch_start,
                batch_end,
                known_parameters,
                optimizable_parameters,
                parameter_masks,
                iterations,
                learning_rates,
            )

        return variables

    def _update_translation_from_vertices(
        self,
        variables: dict[str, torch.Tensor],
        target_verts_batch: torch.Tensor,
        batch_start: int,
        batch_end: int,
    ) -> None:
        """Update translation based on full vertex alignment."""
        with torch.no_grad():
            mhr_verts = self._get_mhr_vertices_batch(variables, batch_start, batch_end)
            # Compute translation update from all vertices
            update = (target_verts_batch - mhr_verts).mean(dim=1)
            variables["transls"][batch_start:batch_end] += update.detach() / 10.0

    def _optimize_all_parameters(
        self,
        target_vertices: torch.Tensor,
        num_frames: int,
        variables: dict[str, torch.Tensor],
        known_parameters: dict[str, torch.Tensor],
        num_epochs: int,
        learning_rate: float,
    ) -> None:
        """Optimize all parameters with identity-related parameters."""
        disable_inner_message = self._should_disable_progress_messages()

        if not disable_inner_message:
            logger.info("Optimize all parameters...")

        optimizer = torch.optim.Adam(
            [
                v
                for k, v in variables.items()
                if k
                not in (
                    "lbs_model_params",
                    "face_expr_coeffs",
                    "identity_coeffs",
                )  # lbs_model_params and identity_coeffs are concatenated parameters
                and k not in known_parameters
            ],
            lr=learning_rate,
        )
        parameter_mask = {}
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                OptimizationConstants.SCHEDULER_MILESTONE_1,
                OptimizationConstants.SCHEDULER_MILESTONE_2,
            ],
            gamma=OptimizationConstants.SCHEDULER_GAMMA,
        )

        for epoch_id in tqdm(
            range(num_epochs), desc="Optimize all params", disable=disable_inner_message
        ):
            for batch_start in range(0, num_frames, self._batch_size):
                batch_end = min(batch_start + self._batch_size, num_frames)
                target_verts_batch = target_vertices[batch_start:batch_end]
                target_edge_vecs = self._compute_edge_vectors(
                    target_verts_batch, self._mhr_edges
                )

                edge_loss_weight = (
                    OptimizationConstants.INITIAL_EDGE_WEIGHT
                    if epoch_id < OptimizationConstants.EDGE_TO_VERTEX_TRANSITION_EPOCH
                    else OptimizationConstants.REDUCED_EDGE_WEIGHT
                )
                vertex_loss_weight = (
                    OptimizationConstants.REDUCED_EDGE_WEIGHT
                    if epoch_id < OptimizationConstants.EDGE_TO_VERTEX_TRANSITION_EPOCH
                    else OptimizationConstants.INITIAL_EDGE_WEIGHT
                )
                self._optimize_one_batch(
                    batch_start,
                    batch_end,
                    variables,
                    target_edge_vecs,
                    target_verts_batch,
                    optimizer,
                    scheduler=scheduler,
                    edge_loss_weight=edge_loss_weight,
                    vertex_loss_weight=vertex_loss_weight,
                    gradient_masks=parameter_mask,
                )


class PyTorchSMPLFitting(BasePyTorchFitting):
    """Class for fitting SMPL models using PyTorch optimization.

    This class provides PyTorch-based optimization for fitting SMPL(X) body models
    to target vertex positions using Adam optimizer.
    """

    def __init__(
        self,
        smpl_model: smplx.SMPLX,
        smpl_edges: torch.Tensor,
        smpl_model_type: str,
        hand_pose_dim: int,
        device: str = "cuda",
        batch_size: int = 256,
    ) -> None:
        """Initialize the PyTorch SMPL fitting instance.

        Args:
            smpl_model: The SMPL(X) body model to fit
            smpl_edges: Edge connectivity for the SMPL mesh
            smpl_model_type: Type of SMPL model ("smpl" or "smplx")
            hand_pose_dim: Dimension of hand pose parameters
            device: Device to use for computation (default: "cuda")
            batch_size: Batch size for processing frames
        """
        super().__init__(device=device, batch_size=batch_size)
        self._smpl_model = smpl_model
        self._smpl_edges = smpl_edges
        self._smpl_model_type = smpl_model_type
        self._hand_pose_dim = hand_pose_dim

    def fit(
        self,
        target_vertices: torch.Tensor,
        single_identity: bool,
        is_tracking: bool,
        known_parameters: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Fit SMPL model to target vertices using PyTorch optimizer.

        Args:
            target_vertices: Target vertices in SMPL mesh topology for each frame
            single_identity: Whether to use a single identity (betas) for all frames
            is_tracking: Whether to use tracking information to initialize fitting parameters
            known_parameters: Dictionary of known parameters to keep constant

        Returns:
            Dictionary containing fitted SMPL parameters
        """
        num_frames = target_vertices.shape[0]
        target_vertices_center = 0.5 * (
            target_vertices.min(1)[0] + target_vertices.max(1)[0]
        )
        target_vertices -= target_vertices_center[:, None, :]

        if known_parameters is None:
            known_parameters = {}

        if single_identity and is_tracking:
            logger.info("Select frames to estimate body shape...")
            selected_frame_indices = self._select_frames_for_identity_estimation(
                target_vertices
            )

            logger.info("Optimize the identity...")
            variables = self._optimize_smpl(
                target_vertices[selected_frame_indices],
                single_identity=False,
            )

            with torch.no_grad():
                errors = self._evaluate_conversion_error(variables, target_vertices)
                frame_weights = self._to_tensor(errors.max() / errors)
                frame_weights = frame_weights / frame_weights.sum()
                known_parameters["betas"] = (
                    frame_weights[..., None] * variables["betas"]
                ).sum(0, keepdim=True)

        if is_tracking and target_vertices.shape[0] > 16 * self._batch_size:
            logger.info("Optimize parameters for each frame by tracking...")
            variables = self._track_smpl(
                target_vertices, single_identity, known_parameters
            )
        else:
            logger.info("Optimize each frames...")
            variables = self._optimize_smpl(
                target_vertices,
                single_identity=single_identity,
                known_parameters=known_parameters,
            )

        logger.info("Optimization completed, returning results...")

        with torch.no_grad():
            variables["transl"] += target_vertices_center
        target_vertices += target_vertices_center[:, None, :]
        # Return parameters as dictionary
        if single_identity:
            variables["betas"] = variables["betas"][:1].expand(num_frames, -1)
        if self._smpl_model_type == "smplx":
            return variables
        else:
            return {
                "betas": variables["betas"],
                "body_pose": variables["body_pose"],
                "global_orient": variables["global_orient"],
                "transl": variables["transl"],
            }

    def _select_frames_for_identity_estimation(
        self, target_vertices: torch.Tensor, num_selected_frames: int = 16
    ) -> np.ndarray:
        """Select frames for identity estimation based on edge distance."""
        return super()._select_frames_for_identity_estimation(
            target_vertices=target_vertices,
            template_vertices=self._smpl_model.v_template,
            faces=self._smpl_model.faces,
            num_selected_frames=num_selected_frames,
        )

    def _evaluate_conversion_error(
        self,
        fitting_parameter_results: dict[str, torch.Tensor],
        target_vertices: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute conversion errors in average vertex distance.

        Args:
            fitting_parameter_results: Dictionary containing SMPL fitting results
            target_vertices: Target vertices tensor in SMPL mesh topology [B, V, 3]

        Returns:
            Array of average vertex distance errors for each frame
        """
        num_frames = len(fitting_parameter_results["betas"])
        error_lists = []

        # Process frames in batches for better memory efficiency
        for batch_start in range(0, num_frames, self._batch_size):
            batch_end = min(batch_start + self._batch_size, num_frames)
            batch_indices = range(batch_start, batch_end)

            # Prepare batch parameters for SMPL model
            betas_batch = torch.stack(
                [fitting_parameter_results["betas"][i] for i in batch_indices],
                dim=0,
            )
            body_pose_batch = torch.stack(
                [fitting_parameter_results["body_pose"][i] for i in batch_indices],
                dim=0,
            )
            global_orient_batch = torch.stack(
                [fitting_parameter_results["global_orient"][i] for i in batch_indices],
                dim=0,
            )
            transl_batch = torch.stack(
                [fitting_parameter_results["transl"][i] for i in batch_indices],
                dim=0,
            )

            # Handle SMPLX-specific parameters if they exist
            if "left_hand_pose" in fitting_parameter_results:
                left_hand_pose_batch = torch.stack(
                    [
                        fitting_parameter_results["left_hand_pose"][i]
                        for i in batch_indices
                    ],
                    dim=0,
                )
                right_hand_pose_batch = torch.stack(
                    [
                        fitting_parameter_results["right_hand_pose"][i]
                        for i in batch_indices
                    ],
                    dim=0,
                )
            else:
                left_hand_pose_batch = torch.zeros(
                    len(batch_indices), 6, device=self._device
                )
                right_hand_pose_batch = torch.zeros(
                    len(batch_indices), 6, device=self._device
                )

            # Generate additional SMPLX parameters if needed
            current_batch_size = batch_end - batch_start
            jaw_pose_batch = torch.zeros(
                [current_batch_size, 1, 3], device=self._device
            )
            leye_pose_batch = torch.zeros(
                [current_batch_size, 1, 3], device=self._device
            )
            reye_pose_batch = torch.zeros(
                [current_batch_size, 1, 3], device=self._device
            )

            # Handle expression parameters for SMPLX
            if "expression" in fitting_parameter_results:
                expression_batch = torch.stack(
                    [fitting_parameter_results["expression"][i] for i in batch_indices],
                    dim=0,
                )
            else:
                expression_batch = torch.zeros(
                    current_batch_size,
                    self._smpl_model.num_expression_coeffs,
                    device=self._device,
                )

            # Generate vertices for the entire batch using SMPL model
            smpl_output = self._smpl_model(
                betas=betas_batch.to(self._device),
                body_pose=body_pose_batch.to(self._device),
                global_orient=global_orient_batch.to(self._device),
                transl=transl_batch.to(self._device),
                left_hand_pose=left_hand_pose_batch.to(self._device),
                right_hand_pose=right_hand_pose_batch.to(self._device),
                jaw_pose=jaw_pose_batch,
                leye_pose=leye_pose_batch,
                reye_pose=reye_pose_batch,
                expression=expression_batch.to(self._device),
            )
            batch_vertices = smpl_output.vertices

            batch_target_vertices = target_vertices[batch_start:batch_end]
            batch_errors = torch.sqrt(
                ((batch_vertices - batch_target_vertices) ** 2).sum(-1)
            ).mean(1)
            batch_errors = batch_errors.detach().cpu().numpy().tolist()

            error_lists.extend(batch_errors)

        return np.array(error_lists)

    def _track_smpl(
        self,
        target_vertices: torch.Tensor,
        single_identity: bool,
        known_parameters: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Track SMPL(X) model parameters for each frame.

        Args:
            target_vertices: Target vertices tensor in SMPL(X) mesh topology [B, V, 3]
            single_identity: If True, uses a single identity parameter for all frames
            known_parameters: Known parameters to be used as constant during optimization

        Returns:
            Dictionary containing tracked SMPL(X) model parameters
        """
        num_frames = target_vertices.shape[0]
        chunk_iterator = ChunkedSequence(
            num_frames, self._batch_size, num_overlapping_frames=1
        )
        max_chunk_size = chunk_iterator.get_num_iterations()

        parameters = self._define_trainable_variables(
            num_frames, single_identity=single_identity
        )
        if known_parameters is None:
            known_parameters = {}

        for k, v in parameters.items():
            parameters[k] = v.requires_grad_(False)
            if k in known_parameters:
                parameters[k] = known_parameters[k].requires_grad_(False)

        disable_inner = len(getattr(tqdm, "_instances", [])) > 0
        for i in tqdm(range(max_chunk_size), disable=disable_inner):
            batch_frame_indices, previous_frame_indices = (
                chunk_iterator.get_frame_indices(i)
            )

            batch_vertices = target_vertices[batch_frame_indices]

            # Estimate the first frame of each chunk.
            if i == 0:
                # Initialize parameters
                batch_parameters = self._optimize_smpl(
                    batch_vertices,
                    single_identity=single_identity,
                    known_parameters=known_parameters,
                )
            else:
                # Track the rest frames of each chunk.
                last_batch_parameters = {}
                for k, v in parameters.items():
                    if k not in known_parameters:
                        last_batch_parameters[k] = v[previous_frame_indices]
                batch_parameters = self._optimize_smpl(
                    batch_vertices,
                    single_identity=single_identity,
                    iterations=((4, 20, 10), 100),
                    learning_rates=(0.05, 0.005),
                    known_parameters=known_parameters,
                    parameter_initialization=last_batch_parameters,
                )

            for k, _ in parameters.items():
                if k not in known_parameters:
                    parameters[k][batch_frame_indices] = batch_parameters[k]

        return parameters

    def _optimize_smpl(
        self,
        target_vertices: torch.Tensor,
        single_identity: bool,
        iterations: tuple[tuple[int, int, int], int] = ((40, 80, 40), 300),
        learning_rates: tuple[float, float] = (0.1, 0.01),
        known_parameters: dict[str, torch.Tensor] | None = None,
        parameter_initialization: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Optimize SMPL(X) model parameters for each frame.

        Args:
            target_vertices: Target vertices tensor in SMPL(X) mesh topology [B, V, 3]
            single_identity: If True, uses a single identity parameter for all frames
            iterations: Number of iterations for each stage of optimization
            learning_rates: Learning rates for each stage
            known_parameters: Known parameters to be used as constant during optimization
            parameter_initialization: Initial parameter values

        Returns:
            Dictionary containing optimized SMPL(X) model parameters
        """
        disable_inner_message = len(getattr(tqdm, "_instances", [])) > 0
        num_frames = target_vertices.shape[0]

        # Define variables for optimization.
        variables = self._define_trainable_variables(
            num_frames,
            single_identity,
            known_variables=known_parameters,
            variable_initialization=parameter_initialization,
        )
        if not disable_inner_message:
            logger.info("Initial pose optimization...")
        for batch_start in tqdm(
            range(0, num_frames, self._batch_size),
            desc="Initial pose optimization batches",
            disable=disable_inner_message,
        ):
            batch_end = min(batch_start + self._batch_size, num_frames)

            target_verts_batch = target_vertices[batch_start:batch_end]
            target_edge_vecs = (
                target_verts_batch[:, self._smpl_edges[:, 1], :]
                - target_verts_batch[:, self._smpl_edges[:, 0], :]
            )

            optimizable_parameters = [
                ["global_orient"],
                [
                    "global_orient",
                    "body_pose",
                    "betas",
                    "expression",
                ],
                [
                    "global_orient",
                    "body_pose",
                    "betas",
                    "expression",
                    "left_hand_pose",
                    "right_hand_pose",
                ],
            ]
            coarse_stage_iterations = iterations[0]
            for op, it in zip(optimizable_parameters, coarse_stage_iterations):
                optimizer = torch.optim.Adam(
                    [variables[p] for p in op if p in variables], lr=learning_rates[0]
                )
                for _ in range(it):
                    self._optimize_one_batch(
                        batch_start,
                        batch_end,
                        variables,
                        target_edge_vecs,
                        target_verts_batch,
                        optimizer,
                    )

            # Compute the initial global translation for each frame.
            with torch.no_grad():
                batched_smpl_parameters = self._get_batched_body_model_parameters(
                    variables, batch_start, batch_end
                )
                smpl_verts = self._smpl_model(**batched_smpl_parameters).vertices
                variables["transl"][batch_start:batch_end] += (
                    (target_verts_batch - smpl_verts).mean(dim=1).detach()
                )
        variables["transl"].requires_grad_()

        # Optimize all parameters (including shape parameters betas)
        if not disable_inner_message:
            logger.info("Optimize all parameters...")
        optimizer = torch.optim.Adam(
            [v for k, v in variables.items() if k != "lbs_model_params"],
            lr=learning_rates[1],
        )
        fine_stage_iterations = iterations[1]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[fine_stage_iterations // 3, fine_stage_iterations // 3 * 2],
            gamma=0.1,
        )
        for epoch_id in tqdm(
            range(fine_stage_iterations),
            desc="Optimize all parameters in batches",
            disable=disable_inner_message,
        ):
            for batch_start in range(0, num_frames, self._batch_size):
                batch_end = min(batch_start + self._batch_size, num_frames)
                target_verts_batch = target_vertices[batch_start:batch_end]
                target_edge_vecs = (
                    target_verts_batch[:, self._smpl_edges[:, 1], :]
                    - target_verts_batch[:, self._smpl_edges[:, 0], :]
                )

                edge_weight = 1.0 if epoch_id < 50 else 0
                self._optimize_one_batch(
                    batch_start,
                    batch_end,
                    variables,
                    target_edge_vecs,
                    target_verts_batch,
                    optimizer,
                    scheduler,
                    edge_weight=edge_weight,
                )

        # Return parameters as dictionary
        if single_identity:
            variables["betas"] = variables["betas"][:1].expand(num_frames, -1)
        if self._smpl_model_type == "smplx":
            return variables
        else:
            return {
                "betas": variables["betas"],
                "body_pose": variables["body_pose"],
                "global_orient": variables["global_orient"],
                "transl": variables["transl"],
            }

    def _define_trainable_variables(
        self,
        num_frames: int,
        single_identity: bool,
        known_variables: dict[str, torch.Tensor] | None = None,
        variable_initialization: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Define trainable variables for MHR to SMPL optimization."""
        if known_variables is None:
            known_variables = {}
        if variable_initialization is None:
            variable_initialization = {}

        global_orient = torch.zeros(
            num_frames, 3, device=self._device, requires_grad=True
        )
        transl = torch.zeros(num_frames, 3, device=self._device)
        body_pose_dim = 69 if self._smpl_model_type == "smpl" else 63
        body_pose = torch.zeros(
            num_frames, body_pose_dim, device=self._device, requires_grad=True
        )

        left_hand_pose = torch.zeros(
            num_frames, self._hand_pose_dim, device=self._device, requires_grad=True
        )
        right_hand_pose = torch.zeros(
            num_frames, self._hand_pose_dim, device=self._device, requires_grad=True
        )
        expression = torch.zeros(
            num_frames,
            self._smpl_model.num_expression_coeffs,
            device=self._device,
            requires_grad=True,
        )

        num_identities = 1 if single_identity else num_frames
        num_betas = self._smpl_model.num_betas
        betas = torch.zeros(
            num_identities, num_betas, device=self._device, requires_grad=True
        )

        variables = {
            "global_orient": global_orient,
            "transl": transl,
            "body_pose": body_pose,
            "betas": betas,
        }
        if self._smpl_model_type == "smplx":
            variables["left_hand_pose"] = left_hand_pose
            variables["right_hand_pose"] = right_hand_pose
            variables["expression"] = expression

        for k, v in known_variables.items():
            variables[k] = v
            variables[k].requires_grad_(False)
        for k, v in variable_initialization.items():
            if k not in known_variables:
                with torch.no_grad():
                    variables[k] = v.detach().clone().requires_grad_(True)
        return variables

    def _get_batched_body_model_parameters(
        self,
        parameter_dict: dict[str, torch.Tensor],
        batch_start: int,
        batch_end: int,
    ) -> dict[str, torch.Tensor]:
        """Get SMPL parameters for a batch of frames."""
        batched_parameter_dict = {}

        for k, v in parameter_dict.items():
            if v.shape[0] < batch_end:
                batched_parameter_dict[k] = v.expand(batch_end - batch_start, -1)
            else:
                batched_parameter_dict[k] = v[batch_start:batch_end].to(self._device)

        # For SMPLX model, we need to make sure jaw_pose, leye_pose,
        # reye_pose, left_hand_pose, right_hand_pose, and expression exist
        # and have the right batch dimension.
        if self._smpl_model_type == "smplx":
            batch_size = batch_end - batch_start
            if "jaw_pose" not in batched_parameter_dict:
                batched_parameter_dict["jaw_pose"] = torch.zeros(
                    [batch_size, 1, 3], device=self._device
                )
            if "leye_pose" not in batched_parameter_dict:
                batched_parameter_dict["leye_pose"] = torch.zeros(
                    [batch_size, 1, 3], device=self._device
                )
            if "reye_pose" not in batched_parameter_dict:
                batched_parameter_dict["reye_pose"] = torch.zeros(
                    [batch_size, 1, 3], device=self._device
                )
            hand_pose_dim = 6 if self._smpl_model.use_pca else 45
            if "left_hand_pose" not in batched_parameter_dict:
                batched_parameter_dict["left_hand_pose"] = torch.zeros(
                    [batch_size, hand_pose_dim], device=self._device
                )
            if "right_hand_pose" not in batched_parameter_dict:
                batched_parameter_dict["right_hand_pose"] = torch.zeros(
                    [batch_size, hand_pose_dim], device=self._device
                )
            expression_dim = self._smpl_model.num_expression_coeffs
            if "expression" not in batched_parameter_dict:
                batched_parameter_dict["expression"] = torch.zeros(
                    [batch_size, expression_dim], device=self._device
                )

        return batched_parameter_dict

    def _optimize_one_batch(
        self,
        batch_start: int,
        batch_end: int,
        variables: dict[str, torch.Tensor],
        target_edge_vecs: torch.Tensor,
        target_verts_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        edge_weight: float | None = None,
    ) -> None:
        """Optimize one batch of SMPL model parameters against target data."""
        batched_smpl_parameters = self._get_batched_body_model_parameters(
            variables, batch_start, batch_end
        )

        # Compute SMPL vertices and edges.
        smpl_verts = self._smpl_model(**batched_smpl_parameters).vertices
        smpl_edges = (
            smpl_verts[:, self._smpl_edges[:, 1], :]
            - smpl_verts[:, self._smpl_edges[:, 0], :]
        )

        # Compute edge loss using absolute difference
        edge_loss = torch.abs(smpl_edges - target_edge_vecs).mean()

        # Compute total loss
        if edge_weight is not None:
            vertex_loss = torch.square(smpl_verts - target_verts_batch).mean()
            loss = edge_weight * edge_loss + vertex_loss
        else:
            loss = edge_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
