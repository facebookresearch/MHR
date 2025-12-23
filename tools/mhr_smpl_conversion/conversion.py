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

"""
This class provides bidirectional conversion between SMPL/SMPLX body models and MHR
body models. It supports converting either model parameters or vertex positions between
the two representations using optimization-based fitting with barycentric interpolation.

Key Features:
    - Bidirectional conversion: SMPL(X) ↔ MHR
    - Multiple backends: PyMomentum (CPU-only) and PyTorch (GPU-enabled)
    - Supports both SMPL and SMPLX model types
    - Handles temporal sequences with tracking mode
    - Supports single-identity mode for consistent shape across frames
    - Optional expression parameter fitting

Usage Example - SMPL to MHR:
    >>> import smplx
    >>> from arvr.libraries.pymomentum.mhr.mhr import MHR
    >>> from arvr.libraries.pymomentum.mhr.tools.smpl_mhr_conversion.conversion import Conversion
    >>>
    >>> # Initialize models
    >>> mhr_model = MHR(...)
    >>> smpl_model = smplx.SMPLX(...)
    >>>
    >>> # Create converter
    >>> converter = Conversion(
    ...     mhr_model=mhr_model,
    ...     smpl_model=smpl_model,
    ...     method="pytorch",  # or "pymomentum"
    ...     batch_size=256
    ... )
    >>>
    >>> # Convert SMPL vertices to MHR parameters
    >>> result = converter.convert_smpl2mhr(
    ...     smpl_vertices=smpl_verts,  # [B, V, 3] tensor or numpy array
    ...     single_identity=True,       # Use consistent shape across frames
    ...     is_tracking=False,          # Not a temporal sequence
    ...     return_mhr_parameters=True,
    ...     return_mhr_meshes=True
    ... )
    >>>
    >>> # Access results
    >>> mhr_params = result.result_parameters
    >>> mhr_meshes = result.result_meshes
    >>> errors = result.result_errors

Usage Example - MHR to SMPL:
    >>> # Convert MHR vertices to SMPL parameters
    >>> result = converter.convert_mhr2smpl(
    ...     mhr_vertices=mhr_verts,  # [B, V, 3] tensor or numpy array
    ...     single_identity=True,
    ...     is_tracking=False,
    ...     return_smpl_parameters=True,
    ...     return_smpl_meshes=True
    ... )
    >>>
    >>> # Access results
    >>> smpl_params = result.result_parameters
    >>> smpl_meshes = result.result_meshes

Method Selection:
    - "pymomentum": CPU-only, hierarchical optimization, only supports MHR->SMPL(X)
    - "pytorch": GPU-enabled, faster for large batches

Conversion Process:
    1. Barycentric interpolation maps vertices between mesh topologies
    2. Optimization-based fitting matches target vertex positions
    3. Optional failure case reprocessing for improved accuracy (PyMomentum only)
"""

import logging
from functools import cached_property, lru_cache
from typing import Optional, Any

import numpy as np

import torch
import torch.optim

import trimesh
from mhr.mhr import MHR
from tqdm import tqdm

from pymomentum_fitting import _NUM_RIG_PARAMETERS, PyMomentumModelFitting
from pytorch_fitting import PyTorchMHRFitting, PyTorchSMPLFitting
from utils import (
    complete_smplx_parameters,
    ConversionResult,
    evaluate_model_fitting_error,
    FittingMethod,
    get_batched_parameters,
    load_head_vertex_weights,
    load_subsampled_vertex_mask,
    load_surface_mapping,
)

_NUM_VERTICES_SMPL = 6890
_NUM_VERTICES_SMPLX = 10475

logger = logging.getLogger(__name__)


class ConversionConstants:
    """Constants used for SMPL-MHR conversion process.

    This class centralizes all magic numbers and thresholds used throughout
    the conversion pipeline to improve maintainability and documentation.
    """

    # Error thresholds
    # Default threshold (in cm) for detecting failure cases in SMPL->MHR conversion.
    # Values above 0.8cm indicate poor vertex alignment between SMPL and MHR meshes
    # and trigger reprocessing with interpolation-based refinement.
    DEFAULT_ERROR_THRESHOLD = 0.8

    # Interpolation parameters
    # Number of intermediate steps when interpolating SMPL parameters from zero
    # to target values during failure case reprocessing. More steps provide
    # smoother transitions but increase computation time.
    DEFAULT_INTERPOLATION_STEPS = 4

    # Frame selection parameters
    # Multiplier for determining when to subsample frames during identity estimation.
    # If num_frames > (SUBSAMPLING_MULTIPLIER * num_selected_frames), then subsample
    # every nth frame where n = num_frames / SUBSAMPLING_MULTIPLIER.
    # This prevents excessive computation on very long sequences.
    SUBSAMPLING_MULTIPLIER = 256

    # Batch processing parameters
    # Default batch size for processing frames in chunks to manage GPU memory.
    # Larger batches are faster but require more memory.
    DEFAULT_BATCH_SIZE = 256

    # Coordinate system conversion factors
    # MHR uses centimeters while SMPL uses meters, requiring conversion.
    METERS_TO_CENTIMETERS = 100.0
    CENTIMETERS_TO_METERS = 0.01


# pyre-ignore[21]: Could not find module `smplx`
import smplx


class Conversion:
    """Main class for converting between SMPL(X) and MHR model representations.

    This class provides functionality to convert SMPL or SMPLX model parameters
    and vertices to MHR model parameters or the other way round using various
    optimization methods.
    Supports both PyMomentum (CPU-only) and PyTorch (GPU-enabled) backends for
    SMPL(X) -> MHR conversion.

    The conversion process uses barycentric interpolation between the model
    topologies and optimization-based fitting to match target vertex positions.

    Args:
        mhr_model: Initialized MHR body model instance
        smpl_model: Initialized SMPL or SMPLX model instance (from smplx library)
        method: Fitting method - "pymomentum" (CPU) or "pytorch" (GPU). Default: "pymomentum"
        batch_size: Number of frames to process in each batch. Default: 256
    """

    def __init__(
        self,
        mhr_model: MHR,
        smpl_model: smplx.SMPLX,
        method: str = "pymomentum",
        batch_size: int = 256,
    ) -> None:
        """Initialize the conversion instance.

        Args:
            mhr_model: MHR body model for conversion target.
            smpl_model: SMPL or SMPLX model for conversion source.
            method: Fitting method to use ("pymomentum" or "pytorch").
                   Defaults to "pymomentum".

        Raises:
            ValueError: If SMPL model has unsupported number of vertices.
        """
        self._DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self._method = FittingMethod.from_string(method)
        if self._method == FittingMethod.PYMOMENTUM:
            self._DEVICE = "cpu"  # pymomentum only supports cpu

        self._mhr_model = mhr_model.to(self._DEVICE)
        self._smpl_model = smpl_model.to(self._DEVICE)

        if smpl_model.v_template.shape[0] == _NUM_VERTICES_SMPL:
            self.smpl_model_type = "smpl"
            self._hand_pose_dim = 0
        elif smpl_model.v_template.shape[0] == _NUM_VERTICES_SMPLX:
            self.smpl_model_type = "smplx"
            self._hand_pose_dim = 6 if smpl_model.use_pca else 45
        else:
            raise ValueError(
                f"Unsupported SMPL model type! Expected {_NUM_VERTICES_SMPL} or {_NUM_VERTICES_SMPLX} vertices, got {smpl_model.v_template.shape[0]}"
            )
        self.pymomentum_solver = None

        self._batch_size = batch_size

        self._mhr_vertex_mask: torch.Tensor = self._load_subsampled_vertex_mask().bool()
        self._mhr_param_masks: dict[
            str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def convert_smpl2mhr(
        self,
        smpl_vertices: torch.Tensor | np.ndarray | None = None,
        smpl_parameters: dict[str, torch.Tensor] | None = None,
        single_identity: bool = False,
        exclude_expression: bool = False,
        is_tracking: bool = False,
        return_mhr_meshes: bool = False,
        return_mhr_parameters: bool = True,
        return_mhr_vertices: bool = False,
        return_fitting_errors: bool = True,
        batch_size: int | None = None,
    ) -> ConversionResult:
        """
        Convert SMPL(X) meshes or model parameters to MHR model parameters.

        Args:
            smpl_vertices: The vertex positions of the SMPL(X) model. Can be a torch tensor, numpy array or None.
            smpl_parameters: The parameters of the SMPL(X) model. Can be a dictionary of torch tensors or None
            single_identity: Whether to use a single identity for all results. If True, a common identity parameters will be optimized
                for all the input SMPL data. If False, each input SMPL data will have its own identity parameters.
            exclude_expression: Whether to exclude expression parameters from the fitting. If True, the function will optimize for body shape and pose parameters only
            is_tracking: Whether the input SMPL data is a temporal sequence. If True, the function will use the previous frame's parameters as the initial parameters for the next frame.
            return_mhr_meshes: Whether to return the MHR meshes. If True, the function will return a list of MHR meshes.
            return_mhr_parameters: Whether to return the MHR parameters. If True, the function will return a dictionary of MHR parameters.
            return_mhr_vertices: Whether to return the MHR vertices. If True, the function will return a numpy array of MHR vertices.
            return_fitting_errors: Whether to return fitting errors. If True, the function will return errors for each frame.
            batch_size: Number of frames to process in each batch. If None, uses the default batch size.

        Returns:
            ConversionResult containing:
                - result_meshes: List of MHR meshes (if return_mhr_meshes=True)
                - result_vertices: Numpy array of MHR vertices (if return_mhr_vertices=True)
                - result_parameters: Dictionary with 'lbs_model_params', 'identity_coeffs', and 'face_expr_coeffs'
                (if return_mhr_parameters=True)
                - result_errors: Numpy array of fitting errors for each frame (if return_fitting_errors=True)

        Raises:
            ValueError: If neither smpl_vertices nor smpl_parameters are provided,
                       or if an unsupported fitting method is specified.
        """
        if batch_size is not None:
            self._batch_size = batch_size
        # Validate and process input data
        if smpl_vertices is not None:
            # Convert to tensor and reshape to expected format
            smpl_vertices = self._to_tensor(smpl_vertices)
            expected_vertices = (
                _NUM_VERTICES_SMPLX
                if self.smpl_model_type == "smplx"
                else _NUM_VERTICES_SMPL
            )
            smpl_vertices = smpl_vertices.reshape(-1, expected_vertices, 3)
        else:
            if smpl_parameters is None:
                raise ValueError(
                    "smpl_parameters must be provided if smpl_vertices is None."
                )
            _, smpl_vertices = self._smpl_para2mesh(smpl_parameters, return_mesh=False)
            smpl_vertices = self._to_tensor(smpl_vertices)

        target_vertices = self._compute_target_vertices(smpl_vertices, "smpl2mhr")
        if self._method == FittingMethod.PYMOMENTUM:
            fitting_parameter_results = self._s2m_fit_mhr_using_pymomentum(
                target_vertices,
                single_identity,
                is_tracking,
                exclude_expression=exclude_expression,
            )
            # Reprocess failure cases by turning the fitting problem into a tracking problem.
            fitting_parameter_results, errors = self._s2m_reprocess_failure_cases(
                fitting_parameter_results,
                smpl_vertices,
                target_vertices,
                error_threshold=ConversionConstants.DEFAULT_ERROR_THRESHOLD,
                exclude_expression=exclude_expression,
                smpl_parameters=smpl_parameters,
            )
        elif self._method == FittingMethod.PYTORCH:
            self._s2m_load_masks()
            fitting_parameter_results = self._s2m_fit_mhr_using_pytorch(
                target_vertices,
                single_identity,
                is_tracking=is_tracking,
                exclude_expression=exclude_expression,
            )
            errors = self._s2m_evaluate_conversion_error(
                fitting_parameter_results,
                target_vertices,
            )
        else:
            raise ValueError(
                f"Unknown fitting method: {self._method}. Only {FittingMethod.PYMOMENTUM} and {FittingMethod.PYTORCH} are supported."
            )

        result_meshes = None
        mhr_vertices = None
        if return_mhr_meshes or return_mhr_vertices:
            result_meshes, mhr_vertices = self._mhr_para2mesh(
                fitting_parameter_results, return_mesh=return_mhr_meshes
            )

        return ConversionResult(
            result_meshes=result_meshes if return_mhr_meshes else None,
            result_vertices=mhr_vertices if return_mhr_vertices else None,
            result_parameters=(
                None if not return_mhr_parameters else fitting_parameter_results
            ),
            result_errors=errors if return_fitting_errors else None,
        )

    def convert_mhr2smpl(
        self,
        mhr_vertices: torch.Tensor | np.ndarray | None = None,
        mhr_parameters: dict[str, torch.Tensor] | None = None,
        single_identity: bool = False,
        is_tracking: bool = False,
        return_smpl_meshes: bool = False,
        return_smpl_parameters: bool = True,
        return_smpl_vertices: bool = False,
        return_fitting_errors: bool = True,
        batch_size: int = 256,
    ) -> ConversionResult:
        """
        Convert MHR meshes or model parameters to SMPL(X) model parameters.

        Args:
            mhr_vertices: The vertex positions of the MHR model. Can be a torch tensor, numpy array or None.
            mhr_parameters: The parameters of the MHR model. Can be a dictionary of torch tensors or None
            single_identity: Whether to use a single identity for all results. If True, a common identity parameters will be optimized
                for all the input MHR data. If False, each input MHR data will have its own identity parameters.
            is_tracking: Whether the input MHR data is a temporal sequence. If True, the function will use the previous frame's parameters as the initial parameters for the next frame.
            return_smpl_meshes: Whether to return the SMPL(X) meshes. If True, the function will return a list of SMPL(X) meshes.
            return_smpl_parameters: Whether to return the SMPL(X) parameters. If True, the function will return a dictionary of SMPL(X) parameters.
            return_smpl_vertices: Whether to return the SMPL(X) vertices. If True, the function will return a numpy array of SMPL(X) vertices.
            return_fitting_errors: Whether to return fitting errors. If True, the function will return errors for each frame.
            batch_size: Number of frames to process in each batch.

        Returns:
            ConversionResult containing:
                - result_meshes: List of SMPL(X) meshes (if return_smpl_meshes=True)
                - result_vertices: Numpy array of SMPL(X) vertices (if return_smpl_vertices=True)
                - result_parameters: Dictionary of SMPL(X) parameters (if return_smpl_parameters=True)
                - result_errors: Numpy array of fitting errors for each frame (if return_fitting_errors=True)
        """
        if batch_size is not None:
            self._batch_size = batch_size
        if mhr_vertices is not None:
            mhr_vertices = self._to_tensor(mhr_vertices)
        else:
            if mhr_parameters is None:
                raise ValueError(
                    "mhr_parameters must be provided if mhr_vertices is None."
                )
            _, mhr_vertices = self._mhr_para2mesh(mhr_parameters, return_mesh=False)
            mhr_vertices = self._to_tensor(mhr_vertices)

        target_vertices = self._compute_target_vertices(mhr_vertices, "mhr2smpl")

        # For MHR to SMPL, we only support pytorch solution
        if self._method == FittingMethod.PYTORCH:
            fitting_parameter_results = self._m2s_fit_smpl_using_pytorch(
                target_vertices,
                single_identity,
                is_tracking,
            )
            errors = self._m2s_evaluate_conversion_error(
                fitting_parameter_results, target_vertices
            )
        else:
            raise ValueError(
                "We only support pytorch solution for MHR -> SMPL conversion!"
            )

        result_meshes = None
        smpl_vertices = None
        if return_smpl_meshes or return_smpl_vertices:
            result_meshes, smpl_vertices = self._smpl_para2mesh(
                fitting_parameter_results, return_mesh=return_smpl_meshes
            )

        return ConversionResult(
            result_meshes=result_meshes if return_smpl_meshes else None,
            result_vertices=smpl_vertices if return_smpl_vertices else None,
            result_parameters=(
                fitting_parameter_results if return_smpl_parameters else None
            ),
            result_errors=errors if return_fitting_errors else None,
        )

    def convert_sam3d_output_to_smpl(
        self,
        sam3d_outputs: list[dict[str, Any]],
        return_smpl_meshes: bool = False,
        return_smpl_parameters: bool = True,
        return_smpl_vertices: bool = False,
        return_fitting_errors: bool = True,
        batch_size: int = 256,
    ) -> ConversionResult:
        """
        Convert SAM3D output to SMPL model parameters.

        Usage:
            After initializing a coverter object from Conversion class,
            sam3d_outputs = SAM3DBodyEstimator.process_one_image(...)
            conversion_results = coverter.convert_sam3d_output_to_smpl(sam3d_outputs)

        Args:
            sam3d_outputs: List of dictionaries containing the output of the SAM3D model.
            return_smpl_meshes: Whether to return the SMPL meshes. If True, the function will return a list
                                of SMPL meshes.
            return_smpl_parameters: Whether to return the SMPL parameters. If True, the function will return a
                                    dictionary of SMPL parameters.
            return_smpl_vertices: Whether to return the SMPL vertices. If True, the function will return a numpy
                                  array of SMPL vertices.
            return_fitting_errors: Whether to return fitting errors. If True, the function will return errors for
                                   each frame.
            batch_size: Number of frames to process in each batch.

        Returns:
            ConversionResult containing:
                - result_meshes: List of SMPL meshes
                - result_vertices: Numpy array of SMPL vertices
                - result_parameters: Dictionary of SMPL parameters
                - result_errors: Numpy array of fitting errors for each frame
        """

        # These are the keys that are expected in the sam3d_outputs, from the first three we can extract the MHR vertices. The last one are the MHR vertices.
        SAM3D_OUTPUT_VARIABLES = (
            "mhr_model_params",  # Translation is not included in the model params, but in "pred_cam_t".
            "shape_params",
            "expr_params",
            "pred_vertices",  # Translation is not included in the vertices, but in "pred_cam_t".
            "pred_cam_t",  # This is the camera translation w.r.t. each person.
        )

        # The result of SAM3DBodyEstimator.process_one_image() is a list of dictionaries, each dictionary contains the outputs of the SAM3D model for one person.
        # We concatenate all person's outputs into one dictionary, so that we can convert them to SMPL in batch.
        concatenated_sam3d_outputs = {}
        for k in SAM3D_OUTPUT_VARIABLES:
            concatenated_sam3d_outputs[k] = np.stack(
                [
                    sam3d_output[k]
                    for sam3d_output in sam3d_outputs
                    if k in sam3d_output
                ],
                axis=0,
            )
            concatenated_sam3d_outputs[k] = self._to_tensor(
                concatenated_sam3d_outputs[k]
            )

        # If MHR vertices are available, use them to compute the SMPL parameters. If not, use the MHR model, shape, and expressions to compute the SMPL parameters.
        num_people = len(sam3d_outputs)
        mhr_vertices = None
        if (
            "pred_vertices" in concatenated_sam3d_outputs
            and concatenated_sam3d_outputs["pred_vertices"].shape[0] == num_people
        ):
            # If pred_vertices is available, use it to compute the target vertices
            # Note: SAM3D outputs the vertices in meters, so we need to convert them to centimeters before passing them to the conversion function
            mhr_vertices = (
                100.0 * concatenated_sam3d_outputs["pred_vertices"]
                + 100.0 * concatenated_sam3d_outputs["pred_cam_t"][:, None, :]
            )
        elif (
            "mhr_model_params" in concatenated_sam3d_outputs
            and concatenated_sam3d_outputs["mhr_model_params"].shape[0] == num_people
        ):
            # If pred_vertices is not available, use the mhr_model_params to compute the target vertices
            mhr_parameters = {}
            mhr_parameters["lbs_model_params"] = concatenated_sam3d_outputs[
                "mhr_model_params"
            ]

            assert (
                "shape_params" in concatenated_sam3d_outputs
                and concatenated_sam3d_outputs["shape_params"].shape[0] == num_people
            )
            mhr_parameters["identity_coeffs"] = concatenated_sam3d_outputs[
                "shape_params"
            ]
            assert (
                "expr_params" in concatenated_sam3d_outputs
                and concatenated_sam3d_outputs["expr_params"].shape[0] == num_people
            )
            mhr_parameters["face_expr_coeffs"] = concatenated_sam3d_outputs[
                "expr_params"
            ]
            _, mhr_vertices = self._mhr_para2mesh(mhr_parameters, return_mesh=False)
            mhr_vertices = self._to_tensor(mhr_vertices)
            mhr_vertices[..., [1, 2]] *= -1  # Camera system difference in SAM3D-Body
            mhr_vertices += 100.0 * concatenated_sam3d_outputs["pred_cam_t"][:, None, :]
        else:
            raise ValueError(
                "Either pred_vertices or mhr_model_params must be available in the SAM3D output."
            )

        conversion_results = self.convert_mhr2smpl(
            mhr_vertices=mhr_vertices,
            single_identity=False,
            is_tracking=False,
            return_smpl_meshes=return_smpl_meshes,
            return_smpl_parameters=return_smpl_parameters,
            return_smpl_vertices=return_smpl_vertices,
            return_fitting_errors=return_fitting_errors,
            batch_size=batch_size,
        )
        return conversion_results

    def _to_tensor(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Convert input data to tensor on the appropriate device."""
        if isinstance(data, torch.Tensor):
            return data.float().to(self._DEVICE)
        return torch.from_numpy(data).float().to(self._DEVICE)

    def _compute_target_vertices(
        self, source_vertices: torch.Tensor, direction: str = "smpl2mhr"
    ) -> torch.Tensor:
        """
        Compute target vertices using barycentric interpolation from source vertices.

        This method performs coordinate space conversion and barycentric interpolation
        to map vertices from one mesh topology to another. It handles the coordinate
        system differences between SMPL (meters) and MHR (centimeters).

        Uses batch processing with PyTorch GPU operations for efficient computation
        while avoiding GPU memory issues.

        Args:
            source_vertices: Source vertices tensor of shape [B, V, 3] where B is batch size,
                           V is number of vertices, and 3 is spatial dimensions (x,y,z).
            direction: Direction of conversion. Must be either "smpl2mhr" or "mhr2smpl".

        Returns:
            Target vertices tensor in target mesh topology with shape [B, V_target, 3]
            where V_target is the number of vertices in the target mesh topology.

        Raises:
            ValueError: If direction is not "smpl2mhr" or "mhr2smpl".
        """
        # Load appropriate mapping
        if direction == "smpl2mhr":
            mapped_face_id, baryc_coords = self._load_surface_mapping_SMPL2MHR()
            source_faces = self._smpl_model.faces
        elif direction == "mhr2smpl":
            mapped_face_id, baryc_coords = self._load_surface_mapping_MHR2SMPL()
            source_faces = self._mhr_model.character.mesh.faces
        else:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'smpl2mhr' or 'mhr2smpl'"
            )

        # Convert to tensors on device
        mapped_face_id_tensor = torch.from_numpy(mapped_face_id).long().to(self._DEVICE)
        baryc_coords_tensor = torch.from_numpy(baryc_coords).float().to(self._DEVICE)
        # Expand baryc_coords to match batch dimension: [1, N, 3, 1]
        baryc_coords_tensor = baryc_coords_tensor[None, :, :, None]
        source_faces_tensor = torch.from_numpy(source_faces).long().to(self._DEVICE)

        # Handle coordinate system conversion for input
        if direction == "mhr2smpl":
            # MHR uses centimeters but SMPL uses meters
            source_vertices = (
                source_vertices * ConversionConstants.CENTIMETERS_TO_METERS
            )

        # Process in batches to avoid GPU memory issues
        num_frames = source_vertices.shape[0]
        target_num_vertices = len(mapped_face_id_tensor)

        # Pre-allocate output tensor for better memory efficiency
        target_vertices = torch.empty(
            num_frames,
            target_num_vertices,
            3,
            dtype=source_vertices.dtype,
            device=self._DEVICE,
        )

        logger.info(f"Converting meshes from {direction} with barycentric mapping...")
        disable_tqdm = len(getattr(tqdm, "_instances", [])) > 0
        for batch_start in tqdm(
            range(0, num_frames, self._batch_size), disable=disable_tqdm
        ):
            batch_end = min(batch_start + self._batch_size, num_frames)
            source_vertices_batch = source_vertices[batch_start:batch_end]

            # Batch barycentric interpolation using PyTorch
            # Get the triangles: [batch_size, N, 3, 3] where N is number of target vertices
            triangles = source_vertices_batch[
                :, source_faces_tensor[mapped_face_id_tensor], :
            ]

            # Perform batch barycentric interpolation
            # Multiply triangles by barycentric weights and sum across vertex dimension
            # triangles * baryc_expanded: [B, N, 3, 3] * [1, N, 3, 1] = [B, N, 3, 3]
            # Sum across vertex dimension (dim=2): [B, N, 3]
            # Directly assign to pre-allocated tensor instead of appending to list
            target_vertices[batch_start:batch_end] = (
                triangles * baryc_coords_tensor
            ).sum(dim=2)

        # Handle coordinate system conversion for output
        if direction == "smpl2mhr":
            # MHR uses centimeters but SMPL uses meters
            target_vertices = (
                target_vertices * ConversionConstants.METERS_TO_CENTIMETERS
            )

        return target_vertices

    @lru_cache
    def _load_head_vertex_weights(self) -> torch.Tensor:
        """Load head vertex weights from the MHR model.

        Returns:
            Vertex weights for head vertices.
        """
        face_vertex_weight = load_head_vertex_weights()
        return self._to_tensor(face_vertex_weight)

    def _s2m_get_mhr_param_mask(
        self, mask_type: str
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get parameter mask for MHR model."""

        num_pose_param = int(
            self._mhr_model.character.parameter_transform.pose_parameters.sum() - 6
        )
        num_scale_params = int(
            self._mhr_model.character.parameter_transform.scaling_parameters.sum()
        )

        lbs_parameter_names = self._mhr_model.character.parameter_transform.names[
            6 : 6 + num_pose_param + num_scale_params
        ]

        if mask_type == "params_for_no_hand":
            _HAND_PARTS = {"index", "middle", "ring", "pinky", "thumb", "wrist"}

            no_hands_pose_param_mask = (
                torch.ones(num_pose_param).bool().to(self._DEVICE)
            )
            no_hands_scale_param_mask = (
                torch.ones(num_scale_params).bool().to(self._DEVICE)
            )

            for i, name in enumerate(lbs_parameter_names):
                if any(part in name for part in _HAND_PARTS):
                    if i < num_pose_param:
                        no_hands_pose_param_mask[i] = False
                    else:
                        no_hands_scale_param_mask[i - num_pose_param] = False

            return no_hands_pose_param_mask, no_hands_scale_param_mask
        elif mask_type == "pose_params_for_head":
            _HEAD_JOINTS = {"neck", "head"}

            head_pose_param_mask = torch.zeros(num_pose_param).bool().to(self._DEVICE)
            head_scale_param_mask = (
                torch.zeros(num_scale_params).bool().to(self._DEVICE)
            )

            for i, name in enumerate(lbs_parameter_names):
                if any(part in name for part in _HEAD_JOINTS):
                    if i < num_pose_param:
                        head_pose_param_mask[i] = True
                    else:
                        head_scale_param_mask[i - num_pose_param] = True

            return head_pose_param_mask, head_scale_param_mask

    def _get_identity_parameter_mask(self) -> torch.Tensor:
        """Get parameter mask for identity-related parameters (scaling and blendshapes).

        Returns:
            Boolean tensor mask indicating which parameters are identity-related.
        """
        scale_mask = self._mhr_model.character.parameter_transform.scaling_parameters
        identity_blendshape_mask = torch.zeros_like(scale_mask)
        identity_blendshape_mask[
            _NUM_RIG_PARAMETERS : _NUM_RIG_PARAMETERS
            + self._mhr_model.get_num_identity_blendshapes()
        ] = True

        identity_parameter_mask = (scale_mask + identity_blendshape_mask).to(
            self._DEVICE
        )
        return identity_parameter_mask

    def _s2m_fit_mhr_using_pymomentum(
        self,
        target_vertices: torch.Tensor,
        single_identity: bool,
        is_tracking: bool,
        exclude_expression: bool = False,
        known_parameters: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Fit MHR model to target vertices using PyMomentum hierarchical optimization.

        This method uses a two-stage fitting process:
        1. Per-frame fitting to get initial parameters for each frame
        2. Single-identity refinement (if enabled) to ensure consistent shape across frames

        The PyMomentum solver uses hierarchical optimization stages progressing from
        rigid transformations to full body and shape parameters.

        Args:
            target_vertices: Target vertices tensor in MHR mesh topology with shape [B, V, 3]
                           where B is number of frames, V is MHR vertices count.
            single_identity: If True, uses averaged identity parameters across all frames
                           for consistent shape. If False, each frame gets unique identity.
            is_tracking: If True, uses previous frame's parameters to initialize the next
                       frame's optimization (temporal consistency).
            exclude_expression: If True, excludes facial expression parameters from fitting.

        Returns:
            Dictionary containing fitted MHR model parameters:
                - 'lbs_model_params': Linear blend skinning parameters tensor [B, P]
                  where P is number of LBS parameters
                - 'identity_coeffs': Identity shape parameters tensor [B, S]
                  where S is number of identity blendshapes
                - 'face_expr_coeffs': Facial expression coefficients tensor [B, F]
                  where F is number of facial expression blendshapes
        """
        if self.pymomentum_solver is None:
            self.pymomentum_solver = PyMomentumModelFitting(
                self._mhr_model, subsampled_mhr_vertex_mask=self._mhr_vertex_mask
            )

        num_frames = target_vertices.shape[0]
        # Get a sample fitting result to determine tensor shapes
        self.pymomentum_solver.reset()
        self.pymomentum_solver.fit(
            target_vertices[0],
            skip_global_stages=False,
            exclude_expression=exclude_expression,
        )
        sample_result = self.pymomentum_solver.get_fitting_results()

        # Pre-allocate result tensors for better memory efficiency
        fitting_parameter_results = {
            "lbs_model_params": torch.zeros(
                num_frames,
                sample_result["lbs_model_params"].shape[0],
                dtype=sample_result["lbs_model_params"].dtype,
                device=self._DEVICE,
            ),
            "identity_coeffs": torch.zeros(
                num_frames,
                sample_result["identity_coeffs"].shape[0],
                dtype=sample_result["identity_coeffs"].dtype,
                device=self._DEVICE,
            ),
            "face_expr_coeffs": torch.zeros(
                num_frames,
                sample_result["face_expr_coeffs"].shape[0],
                dtype=sample_result["face_expr_coeffs"].dtype,
                device=self._DEVICE,
            ),
        }
        # Store the first frame's result
        fitting_parameter_results["lbs_model_params"][0] = sample_result[
            "lbs_model_params"
        ]
        fitting_parameter_results["identity_coeffs"][0] = sample_result[
            "identity_coeffs"
        ]
        fitting_parameter_results["face_expr_coeffs"][0] = sample_result[
            "face_expr_coeffs"
        ]

        if single_identity:
            # Select frames for body shape estimation.
            logger.info("Select frames for identity estimation.")
            selected_frames = self._select_frames_for_identity_estimation(
                target_vertices,
                self._mhr_model.character.mesh.vertices,
                self._mhr_model.character.mesh.faces,
            )
            target_vertices_for_identity = target_vertices[selected_frames]
            num_identity_frames = len(selected_frames)

            # Pre-allocate tensors for identity estimation frames
            identity_parameter_results = {
                "lbs_model_params": torch.zeros(
                    num_identity_frames,
                    sample_result["lbs_model_params"].shape[0],
                    dtype=sample_result["lbs_model_params"].dtype,
                    device=self._DEVICE,
                ),
                "identity_coeffs": torch.zeros(
                    num_identity_frames,
                    sample_result["identity_coeffs"].shape[0],
                    dtype=sample_result["identity_coeffs"].dtype,
                    device=self._DEVICE,
                ),
                "face_expr_coeffs": torch.zeros(
                    num_identity_frames,
                    sample_result["face_expr_coeffs"].shape[0],
                    dtype=sample_result["face_expr_coeffs"].dtype,
                    device=self._DEVICE,
                ),
            }

            # Fit model per-frame for identity estimation
            logger.info("Fit frames for identity estimation.")
            for frame_idx, target_verts in enumerate(
                tqdm(target_vertices_for_identity)
            ):
                self.pymomentum_solver.reset()
                self.pymomentum_solver.fit(
                    target_verts,
                    skip_global_stages=False,
                    exclude_expression=exclude_expression,
                )
                fitting_result = self.pymomentum_solver.get_fitting_results()
                # Directly assign to pre-allocated tensors
                identity_parameter_results["lbs_model_params"][frame_idx] = (
                    fitting_result["lbs_model_params"]
                )
                identity_parameter_results["identity_coeffs"][frame_idx] = (
                    fitting_result["identity_coeffs"]
                )
                identity_parameter_results["face_expr_coeffs"][frame_idx] = (
                    fitting_result["face_expr_coeffs"]
                )
            # Weighted average the identity parameters.
            logger.info("Get weighted average of body shapes.")
            errors = self._s2m_evaluate_conversion_error(
                identity_parameter_results, target_vertices_for_identity
            )
            weights = self._to_tensor(errors.max() / errors)
            weights /= weights.sum()
            concatenated_parameters = torch.cat(
                [
                    identity_parameter_results["lbs_model_params"],
                    identity_parameter_results["identity_coeffs"],
                    identity_parameter_results["face_expr_coeffs"],
                ],
                dim=-1,
            )
            concatenated_parameters = weights[..., None] * concatenated_parameters
            average_fitting_parameter = concatenated_parameters.sum(dim=0)
            # Get the identity related parameters mask, so that these can be set to constant.
            identity_parameter_mask = self._get_identity_parameter_mask()

        # Fit model to each frame. If single identity, set the pre-estimated identity as constant.
        logger.info("Fit model to all the target frames one by one.")
        for i in tqdm(range(num_frames)):
            target_verts = target_vertices[i]
            if not is_tracking:
                self.pymomentum_solver.reset()
            if known_parameters is not None:
                self.pymomentum_solver.set_constant_parameters(
                    known_parameters[0], known_parameters[1][known_parameters[0]]
                )
            skip_global_stages = False
            if is_tracking and i > 0:
                skip_global_stages = True

            if single_identity:
                self.pymomentum_solver.set_constant_parameters(
                    identity_parameter_mask,
                    average_fitting_parameter[identity_parameter_mask],
                )

            self.pymomentum_solver.fit(
                target_verts,
                skip_global_stages=skip_global_stages,
                exclude_expression=exclude_expression,
            )

            fitting_result = self.pymomentum_solver.get_fitting_results()

            # Directly assign to pre-allocated tensors
            fitting_parameter_results["lbs_model_params"][i] = fitting_result[
                "lbs_model_params"
            ]
            fitting_parameter_results["identity_coeffs"][i] = fitting_result[
                "identity_coeffs"
            ]
            fitting_parameter_results["face_expr_coeffs"][i] = fitting_result[
                "face_expr_coeffs"
            ]

        return fitting_parameter_results

    def _s2m_fit_mhr_using_pytorch(
        self,
        target_vertices: torch.Tensor,
        single_identity: bool,
        is_tracking: bool = False,
        exclude_expression: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Fit MHR model to target vertices using PyTorch optimizer.

        Args:
            target_vertices: Target vertices in MHR mesh topology for each frame
            single_identity: Whether to use a single identity for all frames
            is_tracking: Whether to use tracking for temporal sequences
            exclude_expression: Whether to exclude expression parameters from fitting

        Returns:
            Fitted MHR model parameters [B, num_params]
        """
        pytorch_solver = PyTorchMHRFitting(
            mhr_model=self._mhr_model,
            mhr_edges=self._mhr_edges,
            mhr_vertex_mask=self._mhr_vertex_mask,
            mhr_param_masks=self._mhr_param_masks,
            device=self._DEVICE,
            batch_size=self._batch_size,
        )

        return pytorch_solver.fit(
            target_vertices=target_vertices,
            single_identity=single_identity,
            is_tracking=is_tracking,
            exclude_expression=exclude_expression,
        )

    def _s2m_load_masks(self) -> None:
        """
        Load masks for MHR model fitting.
        """
        self._mhr_param_masks["head_pose_params"] = self._s2m_get_mhr_param_mask(
            "pose_params_for_head"
        )[0]
        self._mhr_param_masks["no_hand_param_masks"] = self._s2m_get_mhr_param_mask(
            "params_for_no_hand"
        )

    def _m2s_fit_smpl_using_pytorch(
        self,
        target_vertices: torch.Tensor,
        single_identity: bool,
        is_tracking: bool,
    ) -> dict[str, torch.Tensor]:
        """
        Fit SMPL model to target vertices using PyTorch optimizer.

        Args:
            target_vertices: Target vertices in SMPL mesh topology for each frame
            single_identity: Whether to use a single identity (betas) for all frames
            is_tracking: Whether to use tracking information to initialize fitting parameters

        Returns:
            Dictionary containing fitted SMPL parameters
        """
        pytorch_solver = PyTorchSMPLFitting(
            smpl_model=self._smpl_model,
            smpl_edges=self._smpl_edges,
            smpl_model_type=self.smpl_model_type,
            hand_pose_dim=self._hand_pose_dim,
            device=self._DEVICE,
            batch_size=self._batch_size,
        )

        return pytorch_solver.fit(
            target_vertices=target_vertices,
            single_identity=single_identity,
            is_tracking=is_tracking,
        )

    def _load_surface_mapping(self, direction: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load precomputed surface mapping data for mesh topology conversion.

        This method loads triangle correspondence and barycentric coordinates that enable
        mapping between different mesh topologies. The mapping files are precomputed
        and stored as .npz files in the package resources.

        Args:
            direction: Conversion direction ("smpl2mhr" or "mhr2smpl")

        Returns:
            Tuple containing:
                - triangle_ids: Array of triangle indices in source mesh
                - baryc_coords: Barycentric coordinates for interpolation [N, 3]

        Raises:
            ValueError: If the mapping file cannot be loaded or is not found.
        """
        return load_surface_mapping(direction, self.smpl_model_type)

    @lru_cache
    def _load_surface_mapping_SMPL2MHR(self) -> tuple[np.ndarray, np.ndarray]:
        """Load SMPL/SMPLX to MHR surface mapping (cached)."""
        return self._load_surface_mapping("smpl2mhr")

    @lru_cache
    def _load_surface_mapping_MHR2SMPL(self) -> tuple[np.ndarray, np.ndarray]:
        """Load MHR to SMPL/SMPLX surface mapping (cached)."""
        return self._load_surface_mapping("mhr2smpl")

    def _mhr_para2mesh(
        self,
        mhr_parameters: dict[str, torch.Tensor],
        return_mesh: bool = True,
        verbose: bool = False,
    ) -> tuple[list[trimesh.Trimesh], np.ndarray]:
        """
        Convert MHR parameters to meshes and vertices.

        Args:
            mhr_parameters: Dictionary containing MHR parameters
            return_mesh: Whether to return meshes (default: True)

        Returns:
            Tuple of (list of meshes, numpy array of vertices)
        """
        faces = self._mhr_model.character.mesh.faces
        meshes = []
        mhr_vertices = []
        num_samples = mhr_parameters["lbs_model_params"].shape[0]

        if verbose:
            logger.info("Converting MHR parameters to meshes...")
        for batch_start in tqdm(range(0, num_samples, self._batch_size)):
            batch_end = min(batch_start + self._batch_size, num_samples)
            batch_parameters = get_batched_parameters(
                mhr_parameters, batch_start, batch_end, str(self._DEVICE), "mhr"
            )
            with torch.no_grad():
                mhr_verts, _ = self._mhr_model(
                    identity_coeffs=batch_parameters["identity_coeffs"],
                    model_parameters=batch_parameters["lbs_model_params"],
                    face_expr_coeffs=batch_parameters["face_expr_coeffs"],
                    apply_correctives=True,
                )
                mhr_vertices.append(mhr_verts.detach().cpu().numpy())
        mhr_vertices = np.concatenate(mhr_vertices, axis=0)
        if return_mesh:
            for vertices in mhr_vertices:
                meshes.append(trimesh.Trimesh(vertices, faces, process=False))

        return meshes, mhr_vertices

    def _smpl_para2mesh(
        self,
        smpl_parameters: dict[str, torch.Tensor],
        return_mesh: bool = True,
        verbose: bool = False,
    ) -> tuple[list[trimesh.Trimesh], np.ndarray]:
        """
        Convert SMPL parameters to meshes and vertices.

        Args:
            smpl_parameters: Dictionary containing SMPL parameters
            return_mesh: Whether to return meshes (default: True)

        Returns:
            Tuple of (list of meshes, numpy array of vertices)
        """
        faces = self._smpl_model.faces
        meshes = []
        smpl_vertices = []
        num_samples = smpl_parameters["betas"].shape[0]

        if verbose:
            logger.info("Converting SMPL parameters to meshes...")
        for batch_start in tqdm(range(0, num_samples, self._batch_size)):
            batch_end = min(batch_start + self._batch_size, num_samples)
            batch_parameters = get_batched_parameters(
                smpl_parameters, batch_start, batch_end, str(self._DEVICE), "smpl"
            )

            with torch.no_grad():
                smpl_output = self._smpl_model(**batch_parameters)
                smpl_vertices.append(smpl_output.vertices.detach().cpu().numpy())
        smpl_vertices = np.concatenate(smpl_vertices, axis=0)
        if return_mesh:
            for vertices in smpl_vertices:
                meshes.append(trimesh.Trimesh(vertices, faces, process=False))

        return meshes, smpl_vertices

    def _s2m_reprocess_failure_cases(
        self,
        fitting_parameter_results: dict[str, torch.Tensor],
        smpl_vertices: torch.Tensor,
        target_vertices: torch.Tensor,
        error_threshold: float = 1.0,
        single_identity: bool = False,
        exclude_expression: bool = False,
        smpl_parameters: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        """
        Reprocess failure cases of SMPL->MHR conversion using PyMomentum approach.

        This method identifies conversion failure cases based on average vertex distance errors
        and reprocesses them using an interpolation-based approach for better results.

        Args:
            fitting_parameter_results: Initial MHR fitting results with 'lbs_model_params',
                                     'identity_coeffs', and 'face_expr_coeffs' tensors
            smpl_vertices: SMPL vertices tensor in SMPL mesh topology [B, V_s, 3]
            target_vertices: Target vertices tensor in MHR mesh topology [B, V_t, 3]
            error_threshold: Threshold for average vertex distance to identify failure cases
                           (default: 0.05)
            smpl_parameters: Optional dictionary containing SMPL parameters for reprocessing

        Returns:
            Updated fitting parameter results with improved parameters for failure cases
        """
        logger.info("Reprocessing failure cases with PyMomentum approach...")

        # Step 1: Compute conversion errors in average vertex distance using batch processing
        errors = self._s2m_evaluate_conversion_error(
            fitting_parameter_results, target_vertices
        )

        # Step 2: Calculate average distance errors per frame and threshold to find failure cases
        failure_mask = errors > error_threshold
        failure_indices = np.where(failure_mask)[0]

        if len(failure_indices) == 0:
            logger.info("No failure cases detected. Returning original results.")
            return fitting_parameter_results, errors
        else:
            logger.info(
                f"Detected {len(failure_indices)} failure cases: {failure_indices}.\nReprocessing..."
            )

        # Step 3: If only SMPL vertices are provided, fit SMPL model to get SMPL parameters
        if smpl_parameters is None:
            smpl_params_failure_cases = self._m2s_fit_smpl_using_pytorch(
                smpl_vertices[failure_indices],
                single_identity=single_identity,
                is_tracking=False,
            )
        else:
            smpl_params_failure_cases = {
                key: smpl_parameters[key][failure_indices] for key in smpl_parameters
            }

        # Step 4: For each failure case, reprocess using interpolation approach
        for i, frame_idx in enumerate(failure_indices):
            logger.info(f"Reprocessing failure case for frame {frame_idx}")
            target_frame_vertices = target_vertices[frame_idx]  # [V, 3]
            smpl_params = {}
            for k, v in smpl_params_failure_cases.items():
                smpl_params[k] = v[i]

            # Step 4.1: Interpolate SMPL parameters from all zero to target with default interpolation steps
            interpolation_steps = ConversionConstants.DEFAULT_INTERPOLATION_STEPS
            interpolated_params = {}
            alphas = torch.linspace(0, 1, interpolation_steps + 1).to(
                self._DEVICE
            )  # [steps+1]
            for key, value in smpl_params.items():
                interpolated_params[key] = alphas[..., None] * value.to(
                    self._DEVICE
                )  # [steps+1, F]

            if self.smpl_model_type == "smplx":
                interpolated_params = complete_smplx_parameters(
                    interpolated_params, interpolation_steps + 1, self._DEVICE
                )

            # Generate SMPL vertices for this interpolation step
            with torch.no_grad():
                smpl_output = self._smpl_model(**interpolated_params)

                # Convert SMPL vertices to MHR space
                sequence_vertices = self._compute_target_vertices(
                    smpl_output.vertices, "smpl2mhr"
                )  # [steps+1, V_mhr, 3]
                # Make sure the last frame is exactly the target frame vertices.
                # Without this, the last frame vertices can be different from the target frame vertices if the SMPL(X) parameters are estimated in Step 3.
                sequence_vertices[-1] = target_frame_vertices

            # Step 4.2: Run _s2m_fit_mhr_using_pymomentum with is_tracking=True
            if not single_identity:
                improved_results = self._s2m_fit_mhr_using_pymomentum(
                    sequence_vertices,
                    single_identity=False,
                    is_tracking=True,
                    exclude_expression=exclude_expression,
                )
            else:
                # Average the per-frame identity parameters.
                full_parameter = torch.cat(
                    [
                        fitting_parameter_results["lbs_model_params"][0],
                        fitting_parameter_results["identity_coeffs"][0],
                        fitting_parameter_results["face_expr_coeffs"][0],
                    ],
                    dim=-1,
                )
                # Get the identity related parameters mask, so that these can be set to constant.
                identity_parameter_mask = self._get_identity_parameter_mask()

                improved_results = self._s2m_fit_mhr_using_pymomentum(
                    sequence_vertices,
                    single_identity=True,
                    is_tracking=True,
                    exclude_expression=exclude_expression,
                    known_parameters=(identity_parameter_mask, full_parameter),
                )

            # Step 4.3 Get the fitting result for the final frame and update the mhr result parameters
            improved_verts, _ = self._mhr_model(
                identity_coeffs=improved_results["identity_coeffs"][-1][None, ...].to(
                    self._DEVICE
                ),
                model_parameters=improved_results["lbs_model_params"][-1][None, ...].to(
                    self._DEVICE
                ),
                face_expr_coeffs=improved_results["face_expr_coeffs"][-1][None, ...].to(
                    self._DEVICE
                ),
                apply_correctives=True,
            )

            error = torch.sqrt(
                ((improved_verts - target_frame_vertices) ** 2).sum(-1)
            ).mean()

            # Update the mhr result parameters if the error is lower than the original error.
            if error < errors[frame_idx]:
                fitting_parameter_results["lbs_model_params"][frame_idx] = (
                    improved_results["lbs_model_params"][-1]
                )
                fitting_parameter_results["identity_coeffs"][frame_idx] = (
                    improved_results["identity_coeffs"][-1]
                )
                fitting_parameter_results["face_expr_coeffs"][frame_idx] = (
                    improved_results["face_expr_coeffs"][-1]
                )
                logger.info(
                    f"Frame {frame_idx}: Error improved from {errors[frame_idx]:.6f} to {error:.6f}"
                )
                errors[frame_idx] = error
            else:
                logger.info(
                    f"Frame {frame_idx}: Error not improved. Original error: {errors[frame_idx]:.6f}, after reprocessing: {error:.6f}"
                )

        # Step 5: Return updated conversion results
        logger.info("Failure case reprocessing completed.")
        return (
            fitting_parameter_results,
            errors,
        )

    def _s2m_evaluate_conversion_error(
        self,
        fitting_parameter_results: dict[str, torch.Tensor],
        target_vertices: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute conversion errors in average vertex distance using batch processing.

        Args:
            fitting_parameter_results: Dictionary containing MHR fitting results with
                                     'lbs_model_params', 'identity_coeffs', and 'face_expr_coeffs'
            target_vertices: Target vertices tensor in MHR mesh topology [B, V, 3]

        Returns:
            Array of average vertex distance errors for each frame
        """
        return evaluate_model_fitting_error(
            model=self._mhr_model,
            parameters=fitting_parameter_results,
            target_vertices=target_vertices,
            batch_size=self._batch_size,
            device=self._DEVICE,
            model_type="mhr",
        )

    def _m2s_evaluate_conversion_error(
        self,
        fitting_parameter_results: dict[str, torch.Tensor],
        target_vertices: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute conversion errors in average vertex distance using batch processing.

        Args:
            fitting_parameter_results: Dictionary containing SMPL fitting results with
                                       'betas', 'body_pose', 'global_orient', 'transl', etc.
            target_vertices: Target vertices tensor in SMPL mesh topology [B, V, 3]

        Returns:
            Array of average vertex distance errors for each frame
        """
        return evaluate_model_fitting_error(
            model=self._smpl_model,
            parameters=fitting_parameter_results,
            target_vertices=target_vertices,
            batch_size=self._batch_size,
            device=self._DEVICE,
            model_type="smpl",
        )

    def _select_frames_for_identity_estimation(
        self,
        target_vertices: torch.Tensor,
        source_vertex: torch.Tensor | np.ndarray,
        faces: np.ndarray,
        num_selected_frames: int = 16,
    ) -> np.ndarray:
        """
        Select frames for identity estimation based on the average edge distance between the source and target meshes.

        Args:
            target_vertices: Target vertices tensor in MHR mesh topology [B, V, 3]
            source_vertex: Source vertices tensor in MHR mesh topology [V, 3]
            faces: Faces tensor in MHR mesh topology [F, 3]
            num_selected_frames: Number of frames to select

        Returns:
            Selected frame indices
        """
        tmp_mesh = trimesh.Trimesh(source_vertex, faces, process=False)
        edges = tmp_mesh.edges_unique.copy()
        source_vertex = self._to_tensor(source_vertex)

        original_num_frames = target_vertices.shape[0]
        num_selected_frames = min(original_num_frames, num_selected_frames)
        subsampling_threshold = (
            ConversionConstants.SUBSAMPLING_MULTIPLIER * num_selected_frames
        )
        if original_num_frames > subsampling_threshold:
            process_every_n_frames = (
                original_num_frames // ConversionConstants.SUBSAMPLING_MULTIPLIER
            )
            logger.info(
                f"There are too many ({original_num_frames}) frames."
                f"Subsampling every {process_every_n_frames} frame for identity estimation.",
            )
            target_vertices = target_vertices[::process_every_n_frames]
            logger.info(
                f"This leads to {target_vertices.shape[0]} frames for identity estimation."
            )

        num_frames = target_vertices.shape[0]
        errors = np.full([num_frames], np.inf, dtype=np.float32)

        source_edges = source_vertex[edges[:, 1]] - source_vertex[edges[:, 0]]

        for batch_start in range(0, num_frames, self._batch_size):
            batch_end = min(batch_start + self._batch_size, num_frames)

            target_verts_batch = target_vertices[batch_start:batch_end]
            target_edges = (
                target_verts_batch[:, edges[:, 1], :]
                - target_verts_batch[:, edges[:, 0], :]
            )

            dist = torch.norm(source_edges - target_edges, dim=-1).mean(dim=1)
            errors[batch_start:batch_end] = dist.detach().cpu().numpy()

        selected_frame_indices = np.argsort(errors)[:num_selected_frames]

        return selected_frame_indices

    @lru_cache
    def _load_subsampled_vertex_mask(self) -> torch.Tensor:
        """
        Load subsampled vertex indices for identity estimation.

        Args:
            num_frames: Number of frames in the sequence

        Returns:
            Tensor of subsampled vertex mask
        """
        subsampling_mask = load_subsampled_vertex_mask()
        return self._to_tensor(subsampling_mask)

    @cached_property
    def _smpl_edges(self) -> torch.Tensor:
        """
        Get edges tensor for SMPL model.

        Returns:
            Edges tensor for SMPL model
        """
        smpl_template_mesh = trimesh.Trimesh(
            self._smpl_model.v_template.cpu().numpy(),
            self._smpl_model.faces,
            process=False,
        )
        edges = self._to_tensor(smpl_template_mesh.edges_unique.copy()).long()
        return edges

    @cached_property
    def _mhr_edges(self) -> torch.Tensor:
        """
        Get edges tensor for MHR model.

        Returns:
            Edges tensor for MHR model
        """
        mhr_template_mesh = trimesh.Trimesh(
            self._mhr_model.character.mesh.vertices,
            self._mhr_model.character.mesh.faces,
            process=False,
        )
        edges = self._to_tensor(mhr_template_mesh.edges_unique.copy()).long()
        return edges
