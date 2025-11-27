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

"""Utility functions and helper classes for SMPL-MHR conversion."""

import dataclasses
import enum
import logging
from functools import lru_cache
from typing import Callable

import numpy as np
import torch
import trimesh
from mhr.mhr import MHR

from file_assets import (
    SMPL2MHR_MAPPING_FILE,
    SMPLX2MHR_MAPPING_FILE,
    SUBSAMPLED_VERTEX_INDICES_FILE,
    MHR2SMPL_MAPPING_FILE,
    MHR2SMPLX_MAPPING_FILE,
    MHR_FACE_MASK_FILE,
)

logger = logging.getLogger(__name__)

# pyre-ignore[21]: Could not find module `smplx`
import smplx


class FittingMethod(enum.Enum):
    """Enumeration of available fitting methods for MHR model conversion."""

    PYMOMENTUM = "pymomentum"
    PYTORCH = "pytorch"

    @classmethod
    def from_string(cls, method_str: str) -> "FittingMethod":
        """Create FittingMethod from string (case-insensitive)."""
        method_lower = method_str.lower()
        if method_lower == "pymomentum":
            return cls.PYMOMENTUM
        elif method_lower == "pytorch":
            return cls.PYTORCH
        else:
            raise ValueError(
                f"Invalid method: '{method_str}'. Only 'pymomentum' and 'pytorch' are accepted."
            )


@dataclasses.dataclass
class ConversionResult:
    """Data structure containing the results of SMPL to MHR conversion."""

    result_meshes: list[trimesh.Trimesh] | None = None
    result_vertices: np.ndarray | None = None
    result_parameters: dict[str, torch.Tensor] | None = None
    result_errors: np.ndarray | None = None


class ChunkedSequence:
    """Manages chunked processing of a frame sequence with overlapping boundaries."""

    def __init__(
        self, num_frames: int, num_chunks: int, num_overlapping_frames: int = 0
    ) -> None:
        """Initialize a chunked sequence."""
        self.num_frames = num_frames
        self.num_chunks = num_chunks
        self.num_overlapping_frames = num_overlapping_frames
        self._chunk_boundaries = self._calculate_chunk_boundaries()

    def _calculate_chunk_boundaries(self) -> list[tuple[int, int]]:
        """Calculate start and end indices for each chunk with overlaps."""
        if self.num_chunks <= 0:
            return []

        if self.num_chunks == 1:
            return [(0, self.num_frames)]

        total_overlap = (self.num_chunks - 1) * self.num_overlapping_frames
        base_chunk_size = (self.num_frames + total_overlap) // self.num_chunks
        extra_frames = (self.num_frames + total_overlap) % self.num_chunks

        boundaries = []
        current_pos = 0

        for i in range(self.num_chunks):
            chunk_size = base_chunk_size + (1 if i < extra_frames else 0)

            start_idx = current_pos
            end_idx = start_idx + chunk_size

            start_idx = max(0, min(start_idx, self.num_frames))
            end_idx = max(start_idx, min(end_idx, self.num_frames))

            boundaries.append((start_idx, end_idx))
            current_pos = end_idx - self.num_overlapping_frames

        if boundaries and boundaries[-1][1] != self.num_frames:
            last_start = boundaries[-1][0]
            boundaries[-1] = (last_start, self.num_frames)

        return boundaries

    def get_frame_indices(
        self, current_iteration: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get current and previous frame indices for one frame from each chunk."""
        current_indices = []
        previous_indices = []

        for start_idx, end_idx in self._chunk_boundaries:
            current_frame_idx = start_idx + current_iteration

            if current_frame_idx >= end_idx or current_frame_idx >= self.num_frames:
                continue

            current_indices.append(current_frame_idx)

            if current_iteration > 0:
                prev_idx = start_idx + current_iteration - 1
                if prev_idx >= start_idx and prev_idx < end_idx:
                    previous_indices.append(prev_idx)
                else:
                    previous_indices.append(prev_idx)

        return np.array(current_indices, dtype=np.int64), np.array(
            previous_indices, dtype=np.int64
        )

    def get_chunk_boundaries(self) -> list[tuple[int, int]]:
        """Get the start and end indices for all chunks."""
        return self._chunk_boundaries.copy()

    def get_chunk_size(self, chunk_idx: int) -> int:
        """Get the number of frames in a specific chunk."""
        if chunk_idx < 0 or chunk_idx >= len(self._chunk_boundaries):
            return 0
        start_idx, end_idx = self._chunk_boundaries[chunk_idx]
        return end_idx - start_idx

    def get_num_iterations(self) -> int:
        """Get the minimum number of iterations needed to iterate over all frame indices."""
        if not self._chunk_boundaries:
            return 0

        max_chunk_size = max(
            end_idx - start_idx for start_idx, end_idx in self._chunk_boundaries
        )
        return max_chunk_size


def load_surface_mapping(
    direction: str, smpl_model_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed surface mapping data for mesh topology conversion."""
    if direction == "smpl2mhr":
        mapping_file_path = (
            SMPL2MHR_MAPPING_FILE
            if smpl_model_type == "smpl"
            else SMPLX2MHR_MAPPING_FILE
        )
    elif direction == "mhr2smpl":
        mapping_file_path = (
            MHR2SMPL_MAPPING_FILE
            if smpl_model_type == "smpl"
            else MHR2SMPLX_MAPPING_FILE
        )
    else:
        raise ValueError(f"Invalid direction: {direction}")

    mapping = np.load(mapping_file_path)
    return mapping["triangle_ids"], mapping["baryc_coords"]


@lru_cache
def load_subsampled_vertex_mask() -> np.ndarray:
    """Load subsampled vertex indices for identity estimation."""
    subsampling_mask = np.load(SUBSAMPLED_VERTEX_INDICES_FILE)
    return subsampling_mask


def load_head_vertex_weights() -> np.ndarray:
    """Load head vertex weights from the MHR model."""
    tmp_mesh = trimesh.load(MHR_FACE_MASK_FILE, process=False)
    face_vertex_weight = tmp_mesh.visual.vertex_colors[:, 0] / 255.0
    return face_vertex_weight


def evaluate_model_fitting_error(
    model: MHR | smplx.SMPLX,
    parameters: dict[str, torch.Tensor],
    target_vertices: torch.Tensor,
    batch_size: int,
    device: str,
    model_type: str = "mhr",
    parameter_preparer: Callable[
        [dict[str, torch.Tensor], int, int, str], dict[str, torch.Tensor]
    ]
    | None = None,
) -> np.ndarray:
    """Compute conversion errors in average vertex distance using batch processing.

    This is a generic error evaluation function that works with both MHR and SMPL(X)
    models. It processes frames in batches for memory efficiency and computes the
    mean vertex distance error between generated and target vertices.

    Args:
        model: The body model (MHR or SMPL(X)) to evaluate
        parameters: Dictionary containing model parameters. For MHR models, expects
            'lbs_model_params', 'identity_coeffs', and 'face_expr_coeffs'. For SMPL
            models, expects 'betas', 'body_pose', 'global_orient', 'transl', and
            optionally SMPLX-specific parameters.
        target_vertices: Target vertices tensor in model mesh topology [B, V, 3]
            where B is batch size, V is number of vertices, 3 is spatial dimensions
        batch_size: Number of frames to process in each batch for memory efficiency
        device: Device to use for computation ('cuda' or 'cpu')
        model_type: Type of model - "mhr" or "smpl" (default: "mhr")
        parameter_preparer: Optional callback function to prepare batch parameters.
            Takes (parameters, batch_start, batch_end, device) and returns a dict
            of batched parameters. If None, uses default preparation based on model_type.

    Returns:
        Array of average vertex distance errors for each frame [B]

    Example:
        >>> # For MHR model
        >>> errors = evaluate_model_fitting_error(
        ...     model=mhr_model,
        ...     parameters=fitting_results,
        ...     target_vertices=target_verts,
        ...     batch_size=256,
        ...     device="cuda",
        ...     model_type="mhr"
        ... )
        >>> # For SMPL model
        >>> errors = evaluate_model_fitting_error(
        ...     model=smpl_model,
        ...     parameters=fitting_results,
        ...     target_vertices=target_verts,
        ...     batch_size=256,
        ...     device="cuda",
        ...     model_type="smpl"
        ... )
    """
    num_frames = _get_num_frames(parameters, model_type)

    # Pre-allocate errors on GPU for efficiency
    errors = torch.zeros(num_frames, device=device, dtype=torch.float32)

    with torch.no_grad():
        # Prepare parameters if needed (e.g., concatenate LBS params)
        if model_type == "mhr" and all(
            k in parameters for k in ["rots", "transls", "pose_params"]
        ):
            _concat_mhr_lbs_model_parameters(parameters)

        # Process frames in batches for better memory efficiency
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)

            # Prepare batch parameters
            if parameter_preparer is not None:
                batch_params = parameter_preparer(
                    parameters, batch_start, batch_end, device
                )
            else:
                batch_params = get_batched_parameters(
                    parameters, batch_start, batch_end, device, model_type
                )

            # Generate vertices for the batch using the appropriate model
            if model_type == "mhr":
                batch_vertices, _ = model(
                    identity_coeffs=batch_params["identity_coeffs"],
                    model_parameters=batch_params["lbs_model_params"],
                    face_expr_coeffs=batch_params["face_expr_coeffs"],
                    apply_correctives=True,
                )
            elif model_type == "smpl":
                smpl_output = model(**batch_params)
                batch_vertices = smpl_output.vertices
            else:
                raise ValueError(
                    f"Unsupported model_type: {model_type}. Must be 'mhr' or 'smpl'"
                )

            # Compute batch errors
            batch_target_vertices = target_vertices[batch_start:batch_end]
            batch_errors = torch.sqrt(
                ((batch_vertices - batch_target_vertices) ** 2).sum(-1)
            ).mean(1)
            errors[batch_start:batch_end] = batch_errors

    # Single CPU transfer at the end
    return errors.cpu().numpy()


def _get_num_frames(parameters: dict[str, torch.Tensor], model_type: str) -> int:
    """Get the number of frames from parameter dictionary.

    Args:
        parameters: Dictionary of model parameters
        model_type: Type of model - "mhr" or "smpl"

    Returns:
        Number of frames in the parameter dictionary
    """
    if model_type == "mhr":
        if "lbs_model_params" in parameters:
            return len(parameters["lbs_model_params"])
        elif "rots" in parameters:
            return len(parameters["rots"])
        else:
            raise ValueError("Cannot determine number of frames from MHR parameters")
    elif model_type == "smpl":
        return len(parameters["betas"])
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def _concat_mhr_lbs_model_parameters(
    separate_parameters: dict[str, torch.Tensor],
    include_identity: bool = False,
) -> None:
    """Concatenate separate MHR LBS model parameters into a single tensor.

    Combines translation, rotation, pose, and scale parameters into a single
    'lbs_model_params' tensor for efficient MHR model evaluation. Optionally
    concatenates identity blendshape coefficients. Modifies the input dictionary
    in-place.

    Args:
        separate_parameters: Dictionary containing separate parameter tensors
            with keys 'transls', 'rots', 'pose_params', 'scale_params'.
            If include_identity=True, also requires 'body_identity_coeffs',
            'head_identity_coeffs', and 'hand_identity_coeffs'.
        include_identity: If True, also concatenates identity blendshape
            coefficients into 'identity_coeffs' key (default: False)

    Side Effects:
        Updates separate_parameters dict with 'lbs_model_params' key containing
        concatenated LBS parameters. If include_identity=True, also adds
        'identity_coeffs' key containing concatenated identity blendshapes.
    """
    if (
        separate_parameters["scale_params"].shape[0]
        == separate_parameters["pose_params"].shape[0]
    ):
        expanded_scale_params = separate_parameters["scale_params"]
    else:
        expanded_scale_params = separate_parameters["scale_params"].expand(
            separate_parameters["pose_params"].shape[0], -1
        )

    separate_parameters["lbs_model_params"] = torch.cat(
        [
            separate_parameters["transls"],
            separate_parameters["rots"],
            separate_parameters["pose_params"],
            expanded_scale_params,
        ],
        dim=-1,
    )

    # Optionally concatenate identity blendshape coefficients
    if include_identity and all(
        k in separate_parameters
        for k in [
            "body_identity_coeffs",
            "head_identity_coeffs",
            "hand_identity_coeffs",
        ]
    ):
        separate_parameters["identity_coeffs"] = torch.cat(
            [
                separate_parameters["body_identity_coeffs"],
                separate_parameters["head_identity_coeffs"],
                separate_parameters["hand_identity_coeffs"],
            ],
            dim=-1,
        )


def get_batched_parameters(
    parameter_dict: dict[str, torch.Tensor],
    batch_start: int,
    batch_end: int,
    device: str,
    model_type: str = "mhr",
) -> dict[str, torch.Tensor]:
    """Get batched parameters for a specific range of frames.

    This is a generic parameter batching function that works with both MHR and SMPL(X) models.
    It extracts a batch slice from a parameter dictionary and handles parameter expansion
    for single-identity cases.

    Args:
        parameter_dict: Dictionary containing model parameters
        batch_start: Start index of the batch
        batch_end: End index of the batch
        device: Device to place the tensors on
        model_type: Type of model - "mhr" or "smpl" (default: "mhr")

    Returns:
        Dictionary containing batched parameters for the specified frame range
    """
    batched_parameter_dict = {}

    for k, v in parameter_dict.items():
        if v.shape[0] < batch_end:
            batched_parameter_dict[k] = v.expand(batch_end - batch_start, -1)
        else:
            batched_parameter_dict[k] = v[batch_start:batch_end].to(device)

    # For SMPLX model, ensure all required parameters exist with correct batch dimension
    if model_type == "smpl":
        batched_parameter_dict = complete_smplx_parameters(
            batched_parameter_dict, batch_end - batch_start, device
        )

    return batched_parameter_dict


def complete_smplx_parameters(
    smplx_parameters: dict[str, torch.Tensor],
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Complete SMPLX parameters with default values for missing keys.

    SMPLX models require additional parameters (jaw_pose, eye poses, hand poses,
    expression) that may not be present in the parameter dictionary. This function
    fills in missing parameters with zero tensors.

    Args:
        smplx_parameters: Dictionary of SMPLX parameters (may be incomplete)
        batch_size: Number of frames in the batch
        device: Device to place the tensors on

    Returns:
        Complete SMPLX parameter dictionary with all required keys
    """
    if "jaw_pose" not in smplx_parameters:
        smplx_parameters["jaw_pose"] = torch.zeros([batch_size, 1, 3], device=device)
    if "leye_pose" not in smplx_parameters:
        smplx_parameters["leye_pose"] = torch.zeros([batch_size, 1, 3], device=device)
    if "reye_pose" not in smplx_parameters:
        smplx_parameters["reye_pose"] = torch.zeros([batch_size, 1, 3], device=device)

    # Hand pose dimensions depend on PCA usage
    if "left_hand_pose" not in smplx_parameters:
        # Default to 6 dimensions (PCA mode), will be overridden if needed
        smplx_parameters["left_hand_pose"] = torch.zeros([batch_size, 6], device=device)
    if "right_hand_pose" not in smplx_parameters:
        smplx_parameters["right_hand_pose"] = torch.zeros(
            [batch_size, 6], device=device
        )

    # Expression parameters
    if "expression" not in smplx_parameters:
        # Default to 10 expression coefficients, will be overridden if needed
        smplx_parameters["expression"] = torch.zeros([batch_size, 10], device=device)

    return smplx_parameters


def compute_edge_vectors(vertices: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Compute edge vectors efficiently using advanced indexing.

    Args:
        vertices: Vertices tensor [B, V, 3] where B is batch size, V is number of vertices
        edges: Edge connectivity tensor [E, 2] where E is number of edges

    Returns:
        Edge vectors tensor [B, E, 3]
    """
    return vertices[:, edges[:, 1], :] - vertices[:, edges[:, 0], :]


def compute_vertex_loss(
    predicted_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
    vertex_weights: torch.Tensor | None = None,
    vertex_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute weighted L2 vertex loss.

    Args:
        predicted_vertices: Predicted vertices [B, V, 3]
        target_vertices: Target vertices [B, V, 3]
        vertex_weights: Optional vertex weights [B, V, 3] or [V, 3]
        vertex_mask: Optional boolean mask for which vertices to include [V]

    Returns:
        Scalar loss value
    """
    if vertex_weights is None:
        vertex_weights = torch.ones_like(predicted_vertices)

    if vertex_mask is not None:
        vertex_mask = vertex_mask.bool()
        vertex_weights = vertex_weights[:, vertex_mask]
        predicted_vertices = predicted_vertices[:, vertex_mask]
        target_vertices = target_vertices[:, vertex_mask]

    return torch.square(
        vertex_weights * predicted_vertices - vertex_weights * target_vertices
    ).mean()


def compute_edge_loss(
    predicted_edge_vecs: torch.Tensor,
    target_edge_vecs: torch.Tensor,
) -> torch.Tensor:
    """Compute L1 edge loss (more robust to outliers than L2).

    Args:
        predicted_edge_vecs: Predicted edge vectors [B, E, 3]
        target_edge_vecs: Target edge vectors [B, E, 3]

    Returns:
        Scalar loss value
    """
    return torch.abs(predicted_edge_vecs - target_edge_vecs).mean()
