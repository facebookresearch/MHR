# pyre-strict

"""SMPL(X) to MHR Model Conversion

This module provides comprehensive tools for converting between SMPL/SMPLX and MHR body models.

## Workflow Overview

1. **Input Processing**: Accept SMPL(X) / MHR vertices or parameters as target.
2. **Topology Mapping**: Use precomputed barycentric mappings between mesh topologies to get the
    target MHR / SMPL(X) vertices.
3. **Parameter Fitting**: Optimize MHR / SMPL(X) parameters to match target vertex positions.
4. **Output Generation**: Return MHR / SMPL(X) parameters, meshes, and/or vertices as requested.

## Basic Usage
    ```
    # Create converter instance
    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smpl_model,
        method="pymomentum"  # or "pytorch" for GPU acceleration
    )

    # Convert SMPL vertices to MHR parameters
    result = converter.convert_smpl2mhr(
        smpl_vertices=smpl_vertices,
        return_mhr_parameters=True,
        return_mhr_meshes=True,
        single_identity=True  # Use consistent identity across frames
    )

    # Access results
    mhr_params = result.result_parameters
    mhr_meshes = result.result_meshes
    ```
"""

import dataclasses
import enum
import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import smplx
import torch
import trimesh
from mhr.mhr import MHR

from fitting_mhr_with_pymomentum import PyMomentumModelFitting
from tqdm import tqdm

logger = logging.getLogger(__name__)


_SMPL2MHR_MAPPING_FILE = "assets/smpl2mhr_mapping.npz"
_SMPLX2MHR_MAPPING_FILE = "assets/smplx2mhr_mapping.npz"

_MHR2SMPL_MAPPING_FILE = "assets/mhr2smpl_mapping.npz"
_MHR2SMPLX_MAPPING_FILE = "assets/mhr2smplx_mapping.npz"

_NUM_VERTICES_SMPL = 6890
_NUM_VERTICES_SMPLX = 10475


class FittingMethod(enum.Enum):
    """Enumeration of available fitting methods for MHR model conversion.

    Attributes:
        PYMOMENTUM: Use PyMomentum optimization solver (CPU-only)
        PYTORCH: Use PyTorch optimization solver (supports GPU)
    """

    PYMOMENTUM = "pymomentum"
    PYTORCH = "pytorch"

    @classmethod
    def from_string(cls, method_str: str) -> "FittingMethod":
        """Create FittingMethod from string (case-insensitive).

        Args:
            method_str: String representation of the fitting method.
                       Accepts "pymomentum" or "pytorch" (case-insensitive).

        Returns:
            FittingMethod enum value corresponding to the input string.

        Raises:
            ValueError: If method_str is not a valid fitting method.
        """
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
    """Data structure containing the results of SMPL to MHR conversion.

    Attributes:
        result_meshes: List of MHR mesh objects (trimesh.Trimesh) if requested.
                      None if return_mhr_meshes was False.
        result_vertices: Numpy array of MHR vertices [B, V, 3] if requested.
                        None if return_mhr_vertices was False.
        result_parameters: Dictionary containing MHR model parameters if requested.
                          Contains 'lbs_model_params', 'shape_space_params', and 'face_expr_coeffs' keys.
                          None if return_mhr_parameters was False.
    """

    result_meshes: list[trimesh.Trimesh] | None = None
    result_vertices: np.ndarray | None = None
    result_parameters: dict[str, torch.Tensor] | None = None
    result_errors: np.ndarray | None = None


class Conversion:
    """Main class for converting between SMPL(X) and MHR model representations.

    This class provides functionality to convert SMPL or SMPLX model parameters
    and vertices to MHR model parameters using various optimization methods.
    Supports both PyMomentum (CPU-only) and PyTorch (GPU-enabled) backends.

    The conversion process uses barycentric interpolation between the model
    topologies and optimization-based fitting to match target vertex positions.
    """

    def __init__(
        self,
        mhr_model: MHR,
        smpl_model: smplx.SMPLX,
        method: str = "pytorch",
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
        self._DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
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

    def convert_smpl2mhr(
        self,
        smpl_vertices: torch.Tensor | np.ndarray | None = None,
        smpl_parameters: dict[str, torch.Tensor] | None = None,
        single_identity: bool = True,
        is_tracking: bool = False,
        return_mhr_meshes: bool = False,
        return_mhr_parameters: bool = False,
        return_mhr_vertices: bool = False,
        return_fitting_errors: bool = True,
        batch_size: int = 8,
    ) -> ConversionResult:
        """
        Convert SMPL(X) meshes or model parameters to MHR model parameters.

        Args:
            smpl_vertices: The vertex positions of the SMPL(X) model. Can be a torch tensor, numpy array or None.
            smpl_parameters: The parameters of the SMPL(X) model. Can be a dictionary of torch tensors or None
            single_identity: Whether to use a single identity for all results. If True, a common identity parameters will be optimized
                for all the input SMPL data. If False, each input SMPL data will have its own identity parameters.
            is_tracking: Whether the input SMPL data is a temporal sequence. If True, the function will use the previous frame's parameters as the initial parameters for the next frame.
            return_mhr_meshes: Whether to return the MHR meshes. If True, the function will return a list of MHR meshes.
            return_mhr_parameters: Whether to return the MHR parameters. If True, the function will return a dictionary of MHR parameters.
            return_mhr_vertices: Whether to return the MHR vertices. If True, the function will return a numpy array of MHR vertices.

        Returns:
            ConversionResult containing:
                - result_meshes: List of MHR meshes (if return_mhr_meshes=True)
                - result_vertices: Numpy array of MHR vertices (if return_mhr_vertices=True)
                - result_parameters: Dictionary with 'lbs_model_params', 'shape_space_params', and 'face_expr_coeffs'
                (if return_mhr_parameters=True)

        Raises:
            ValueError: If neither smpl_vertices nor smpl_parameters are provided,
                       or if an unsupported fitting method is specified.
        """
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
            fitting_parameter_results = self._s2p_fit_mhr_using_pymomentum(
                target_vertices, single_identity, is_tracking
            )
            # Reprocess failure cases by turning the fitting problem into a tracking problem.
            fitting_parameter_results, errors = self._s2p_reprocess_failure_cases(
                fitting_parameter_results,
                smpl_vertices,
                target_vertices,
                error_threshold=1.0,
                batch_size=batch_size,
            )
        elif self._method == FittingMethod.PYTORCH:
            fitting_parameter_results = self._s2p_fit_mhr_using_pytorch(
                target_vertices, single_identity
            )
            errors = self._s2p_evaluate_conversion_error(
                fitting_parameter_results, target_vertices, batch_size=8
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
            result_parameters=None
            if not return_mhr_parameters
            else fitting_parameter_results,
            result_errors=errors if return_fitting_errors else None,
        )

    def convert_mhr2smpl(
        self,
        mhr_vertices: torch.Tensor | np.ndarray | None = None,
        mhr_parameters: dict[str, torch.Tensor] | None = None,
        single_identity: bool = True,
        is_tracking: bool = False,
        return_smpl_meshes: bool = False,
        return_smpl_parameters: bool = False,
        return_smpl_vertices: bool = False,
        return_fitting_errors: bool = True,
        batch_size: int = 8,
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

        Returns:
            A dictionary containing the following keys:
                - smpl_meshes: A list of SMPL(X) meshes.
                - smpl_parameters: A dictionary of SMPL(X) parameters.
                - smpl_vertices: A numpy array of SMPL(X) vertices.
        """
        if mhr_vertices is not None:
            mhr_vertices = self._to_tensor(mhr_vertices)
        else:
            if mhr_parameters is None:
                raise ValueError(
                    "mhr_parameters must be provided if mhr_vertices is None."
                )
            _, mhr_vertices = self._mhr_para2mesh(
                mhr_parameters, return_mesh=False
            )
            mhr_vertices = self._to_tensor(mhr_vertices)

        target_vertices = self._compute_target_vertices(mhr_vertices, "mhr2smpl")

        # For MHR to SMPL, we only support pytorch solution
        if self._method == FittingMethod.PYTORCH:
            fitting_parameter_results = self._p2s_fit_smpl_using_pytorch(
                target_vertices, single_identity, is_tracking
            )
            errors = self._p2s_evaluate_conversion_error(
                fitting_parameter_results, target_vertices, batch_size=batch_size
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
            result_parameters=fitting_parameter_results
            if return_smpl_parameters
            else None,
            result_errors=errors if return_fitting_errors else None,
        )

    def _to_tensor(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Convert input data to tensor on the appropriate device."""
        if isinstance(data, torch.Tensor):
            return data.to(self._DEVICE)
        return torch.from_numpy(data.copy()).to(self._DEVICE)

    def _compute_target_vertices(
        self, source_vertices: torch.Tensor, direction: str = "smpl2mhr"
    ) -> torch.Tensor:
        """
        Compute target vertices using barycentric interpolation from source vertices.

        This method performs coordinate space conversion and barycentric interpolation
        to map vertices from one mesh topology to another. It handles the coordinate
        system differences between SMPL (meters) and MHR (centimeters).

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
            source_faces = self._mhr_model.character_gpu.mesh.faces.cpu().numpy()
        else:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'smpl2mhr' or 'mhr2smpl'"
            )

        target_vertices_list = []

        for source_verts in source_vertices:
            source_verts_np = source_verts.cpu().numpy()

            # Handle coordinate system conversion
            if direction == "mhr2smpl":
                # MHR uses centimeters but SMPL uses meters
                source_verts_np = source_verts_np / 100.0  # centimeters to meters

            # Perform barycentric interpolation
            target_vertices = trimesh.triangles.barycentric_to_points(
                source_verts_np[source_faces[mapped_face_id]],
                baryc_coords,
            )

            # Handle coordinate system conversion for output
            if direction == "smpl2mhr":
                # MHR uses centimeters but SMPL uses meters
                target_vertices = target_vertices * 100.0  # meters to centimeters

            target_vertices_list.append(target_vertices)

        return torch.from_numpy(np.array(target_vertices_list)).float().to(self._DEVICE)

    def _s2p_define_trainable_variables(
        self, num_frames: int, single_identity: bool
    ) -> dict[str, torch.Tensor]:
        """Define trainable variables for SMPL to MHR optimization.

        Args:
            num_frames: Number of frames, which is also the number of poses
            single_identity: Whether to use single identity across frames

        Returns:
            Dictionary containing tensors for:
                - rots: Rotation parameters [B, 3]
                - transls: Translation parameters [B, 3]
                - pose_parms: Pose parameters [B, P]
                - identity_coeffs: Identity coefficients [I, N_id]
                - face_expr_coeffs: Facial expression coefficients [I, N_expr]
            where B is batch size, I is 1 if single_identity else B
        """
        rots = torch.zeros(num_frames, 3, device=self._DEVICE).requires_grad_()
        transls = torch.zeros(
            num_frames, 3, device=self._DEVICE
        )  # We will initialize it later.

        # Get pose parameter size from MHR character
        num_pose_param = int(
            self._mhr_model.character.parameter_transform.pose_parameters.sum() - 6
        )  # Remove global translation and rotation
        pose_parms = torch.zeros(
            num_frames,
            num_pose_param,
            device=self._DEVICE,
        )  # Joint angles
        pose_parms.requires_grad_()

        # Identity related parameters: scaling and shape blendshapes
        num_identities = 1 if single_identity else num_frames
        # Scaling parameters
        num_scale_params = int(
            self._mhr_model.character.parameter_transform.scaling_parameters.sum()
        )
        scale_parms = torch.zeros(
            num_identities, num_scale_params, device=self._DEVICE
        )  # 10 scaling parameters
        scale_parms.requires_grad_()

        # Identity blendshapes
        num_pca_comp = self._mhr_model.identity_model.blend_shapes.shape[0]
        identity_coeffs = torch.zeros(
            num_identities, num_pca_comp, device=self._DEVICE
        ).requires_grad_()

        # Facial expression parameters
        num_face_expr = (
            self._mhr_model.face_expressions_model.blend_shapes.shape[0]
            if self._mhr_model.face_expressions_model is not None
            else 0
        )
        face_expr_coeffs = torch.zeros(
            num_frames, num_face_expr, device=self._DEVICE
        ).requires_grad_()

        return {
            "rots": rots,
            "transls": transls,
            "pose_parms": pose_parms,
            "scale_parms": scale_parms,
            "identity_coeffs": identity_coeffs,
            "face_expr_coeffs": face_expr_coeffs,
        }

    def _p2s_define_trainable_variables(
        self, num_frames: int, single_identity: bool
    ) -> dict[str, torch.Tensor]:
        """Define trainable variables for MHR to SMPL optimization.

        Args:
            num_frames: Number of frames, which is also the number of poses
            single_identity: Whether to use single identity across frames

        Returns:
            Dictionary containing tensors for:
                - global_orient: Global orientation parameters [B, 3]
                - transl: Translation parameters [B, 3]
                - body_pose: Body pose parameters [B, P]
                - left_hand_pose: Left hand pose parameters [B, H]
                - right_hand_pose: Right hand pose parameters [B, H]
                - expression: Expression parameters [B, E]
                - betas: Shape parameters [I, N_betas]
            where B is batch size, I is 1 if single_identity else B
        """
        global_orient = torch.zeros(
            num_frames, 3, device=self._DEVICE, requires_grad=True
        )
        transl = torch.zeros(
            num_frames, 3, device=self._DEVICE
        )  # Will initialize later
        body_pose_dim = 69 if self.smpl_model_type == "smpl" else 63
        body_pose = torch.zeros(
            num_frames, body_pose_dim, device=self._DEVICE, requires_grad=True
        )

        left_hand_pose = torch.zeros(
            num_frames, self._hand_pose_dim, device=self._DEVICE, requires_grad=True
        )
        right_hand_pose = torch.zeros(
            num_frames, self._hand_pose_dim, device=self._DEVICE, requires_grad=True
        )
        expression = torch.zeros(
            num_frames,
            self._smpl_model.num_expression_coeffs,
            device=self._DEVICE,
            requires_grad=True,
        )

        num_identities = 1 if single_identity else num_frames
        num_betas = self._smpl_model.num_betas
        betas = torch.zeros(
            num_identities, num_betas, device=self._DEVICE, requires_grad=True
        )

        return {
            "global_orient": global_orient,
            "transl": transl,
            "body_pose": body_pose,
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "expression": expression,
            "betas": betas,
        }

    def _s2p_optimize_one_batch(
        self,
        batch_start: int,
        batch_end: int,
        variables: dict[str, torch.Tensor],
        edges: torch.Tensor,
        target_edges: torch.Tensor,
        target_verts_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        edge_weight: float | None = None,
    ) -> None:
        """
        Optimize one batch of MHR model parameters against target data.

        Args:
            batch_start: Starting index of the batch
            batch_end: End index of the batch
            transls: Translation parameters tensor
            rots: Rotation parameters tensor
            pose_parms: Pose parameters tensor
            identity_coeff: Identity coefficients for the batch
            face_expr_coeff: Facial expression coefficients for the batch
            edges: Mesh edges tensor
            target_edges: Target edge vectors
            target_verts_batch: Target vertices for the batch
            optimizer: PyTorch optimizer
            scheduler: Optional learning rate scheduler
            include_vertex_loss: Whether to include vertex loss in addition to edge loss
            edge_weight: Weight for edge loss if vertex loss is included
        """
        self._concat_mhr_lbs_model_parameters(variables)
        batched_mhr_parameters = self._get_batched_body_model_parameters(
            variables, batch_start, batch_end, self._DEVICE, is_mhr_parameters=True
        )
        # if is_single_identity:
        #     batched_identity_coeff = variables["identity_coeffs"][:1].expand(
        #         batch_end - batch_start, -1
        #     )
        # else:
        #     batched_identity_coeff = variables["identity_coeffs"][batch_start:batch_end]

        mhr_verts = self._mhr_model(
            identity_coeffs=batched_mhr_parameters["identity_coeffs"],
            model_parameters=batched_mhr_parameters["lbs_model_params"],
            face_expr_coeffs=batched_mhr_parameters["face_expr_coeffs"],
            apply_correctives=True,
        )
        mhr_edges = mhr_verts[:, edges[:, 1], :] - mhr_verts[:, edges[:, 0], :]

        # Compute edge loss using absolute difference
        edge_loss = torch.abs(mhr_edges - target_edges).mean()

        # Compute total loss
        if edge_weight is not None:
            vertex_loss = torch.square(mhr_verts - target_verts_batch).mean()
            loss = edge_weight * edge_loss + vertex_loss
        else:
            loss = edge_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    def _s2p_fit_mhr_using_pymomentum(
        self,
        target_vertices: torch.Tensor,
        single_identity: bool,
        is_tracking: bool,
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

        Returns:
            Dictionary containing fitted MHR model parameters:
                - 'lbs_model_params': Linear blend skinning parameters tensor [B, P]
                  where P is number of LBS parameters
                - 'shape_space_params': Identity shape parameters tensor [B, S]
                  where S is number of identity blendshapes
                - 'face_expr_coeffs': Facial expression coefficients tensor [B, F]
                  where F is number of facial expression blendshapes
        """
        if self.pymomentum_solver is None:
            self.pymomentum_solver = PyMomentumModelFitting(self._mhr_model)
        fitting_parameter_results = {
            "lbs_model_params": [],
            "shape_space_params": [],
            "face_expr_coeffs": [],
        }
        # First, per-frame fit of MHR model to target vertices.
        logger.info("Fitting per-frame MHR model to SMPL...")

        for i, target_verts in enumerate(tqdm(target_vertices)):
            if not is_tracking:
                self.pymomentum_solver.reset()
            skip_global_stages = False
            if is_tracking and i > 0:
                skip_global_stages = True
            self.pymomentum_solver.fit(
                target_verts, skip_global_stages=skip_global_stages
            )
            fitting_result = self.pymomentum_solver.get_fitting_results()
            fitting_parameter_results["lbs_model_params"].append(
                fitting_result["lbs_model_params"]
            )
            fitting_parameter_results["shape_space_params"].append(
                fitting_result["shape_space_params"]
            )
            fitting_parameter_results["face_expr_coeffs"].append(
                fitting_result["face_expr_coeffs"]
            )

        for k, v in fitting_parameter_results.items():
            fitting_parameter_results[k] = torch.stack(v, dim=0)

        # Average the per-frame identity parameters and use the averaged identity parameters for all frames.
        if single_identity:
            # Average the per-frame identity parameters.
            concatenated_parameters = torch.cat(
                [
                    fitting_parameter_results["lbs_model_params"],
                    fitting_parameter_results["shape_space_params"],
                    fitting_parameter_results["face_expr_coeffs"],
                ],
                dim=-1,
            )
            # TODO(T239857979): Consider a weighted average of the per-frame identity parameters.
            average_fitting_parameter = concatenated_parameters.mean(dim=0)
            # Get the identity related parameters mask, so that these can be set to constant.
            num_identity_bs = self._mhr_model.identity_model.blend_shapes.shape[0]
            num_expr_bs = self._mhr_model.face_expressions_model.blend_shapes.shape[0]
            scale_mask = (
                self._mhr_model.character.parameter_transform.scaling_parameters
            )

            identity_parameter_mask = torch.cat(
                [
                    scale_mask,
                    torch.ones_like(scale_mask)[:num_identity_bs],
                    torch.zeros_like(scale_mask)[:num_expr_bs],
                ]
            ).to(self._DEVICE)

            logger.info("Fitting single-identity MHR model to SMPL data...")
            for i, target_verts in enumerate(tqdm(target_vertices)):
                # Initialize the parameters with the per-frame fitting results
                concatenated_parameter = torch.cat(
                    [
                        fitting_parameter_results["lbs_model_params"][i],
                        fitting_parameter_results["shape_space_params"][i],
                        fitting_parameter_results["face_expr_coeffs"][i],
                    ],
                    dim=-1,
                )
                self.pymomentum_solver.set_initial_parameters(concatenated_parameter)
                # Set the identity related parameters to be constant
                self.pymomentum_solver.set_constant_parameters(
                    identity_parameter_mask,
                    average_fitting_parameter[identity_parameter_mask],
                )
                self.pymomentum_solver.fit(target_verts, skip_global_stages=True)
                fitting_result = self.pymomentum_solver.get_fitting_results()
                for k, v in fitting_result.items():
                    fitting_parameter_results[k][i] = v
        return fitting_parameter_results

    def _s2p_fit_mhr_using_pytorch(
        self,
        target_vertices: torch.Tensor,
        single_identity: bool,
        batch_size: int = 8,
    ) -> dict[str, torch.Tensor]:
        """
        Fit MHR model to target vertices using PyTorch optimizer.

        Args:
            target_vertices: Target vertices in MHR mesh topology for each frame
            single_identity: Whether to use a single identity for all frames
            batch_size: Number of target vertices to process in each batch (default: 8)

        Returns:
            Fitted MHR model parameters [B, num_params]
        """
        # Get mhr mesh edges.
        mhr_template_mesh = trimesh.Trimesh(
            self._mhr_model.character_gpu.mesh.rest_vertices.cpu().numpy(),
            self._mhr_model.character_gpu.mesh.faces.cpu().numpy(),
            process=False,
        )
        edges = self._to_tensor(mhr_template_mesh.edges_unique).int()

        num_frames = target_vertices.shape[0]

        # Define trainable variables for optimization.
        variables = self._s2p_define_trainable_variables(
            num_frames=num_frames, single_identity=single_identity
        )

        logger.info("Initial pose optimization...")
        for batch_start in tqdm(
            range(0, num_frames, batch_size), desc="Initial pose optimization batches"
        ):
            batch_end = min(batch_start + batch_size, num_frames)
            target_verts_batch = target_vertices[batch_start:batch_end]
            target_edges = (
                target_verts_batch[:, edges[:, 1], :]
                - target_verts_batch[:, edges[:, 0], :]
            )

            optimizable_parameters = (
                ["rots"],  # Global rotation
                [
                    "rots",
                    "identity_coeffs",
                    "pose_parms",
                    "scale_parms",
                ],  # Pose and scaling
            )

            iterations = (21, 51)
            for op, it in zip(optimizable_parameters, iterations):
                optimizer = torch.optim.Adam([variables[p] for p in op], lr=0.1)
                for _ in range(it):
                    self._s2p_optimize_one_batch(
                        batch_start,
                        batch_end,
                        variables,
                        edges,
                        target_edges,
                        target_verts_batch,
                        optimizer,
                    )

            # Compute the initial global translation for each frame.
            with torch.no_grad():
                self._concat_mhr_lbs_model_parameters(variables)
                batched_variables = self._get_batched_body_model_parameters(
                    variables, batch_start, batch_end, self._DEVICE
                )
                mhr_verts = self._mhr_model(
                    identity_coeffs=batched_variables["identity_coeffs"],
                    model_parameters=batched_variables["lbs_model_params"],
                    face_expr_coeffs=batched_variables["face_expr_coeffs"],
                    apply_correctives=True,
                )
                variables["transls"][batch_start:batch_end] = (
                    target_verts_batch - mhr_verts
                ).mean(dim=1).detach() / 10.0
            variables["transls"].requires_grad_()

        # Optimize all parameters (added identity related parameters).
        logger.info("Optimize all parameters...")
        optimizer = torch.optim.Adam(
            [v for k, v in variables.items() if k != "lbs_model_params"], lr=0.01
        )  # "lbs_model_params" is not trainable. It is a concatenation of trainable variables.
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50], gamma=0.1
        )
        for epoch_id in tqdm(range(100), desc="Optimize all parameters in batches"):
            for batch_start in range(0, num_frames, batch_size):
                batch_end = min(batch_start + batch_size, num_frames)
                target_verts_batch = target_vertices[batch_start:batch_end]
                target_edges = (
                    target_verts_batch[:, edges[:, 1], :]
                    - target_verts_batch[:, edges[:, 0], :]
                )

                edge_weight = 1.0 if epoch_id < 50 else 0
                self._s2p_optimize_one_batch(
                    batch_start,
                    batch_end,
                    variables,
                    edges,
                    target_edges,
                    target_verts_batch,
                    optimizer,
                    scheduler,
                    edge_weight=edge_weight,
                )

        # Organize the parameters for output.
        self._concat_mhr_lbs_model_parameters(variables)
        if single_identity:
            variables["identity_coeffs"] = variables["identity_coeffs"][:1].expand(
                num_frames, -1
            )
        fitting_parameter_results = {
            "lbs_model_params": variables["lbs_model_params"].detach(),
            "shape_space_params": variables["identity_coeffs"].detach(),
            "face_expr_coeffs": variables["face_expr_coeffs"].detach(),
        }
        return fitting_parameter_results

    def _p2s_optimize_one_batch(
        self,
        batch_start: int,
        batch_end: int,
        variables: dict[str, torch.Tensor],
        edges: torch.Tensor,
        target_edges: torch.Tensor,
        target_verts_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        edge_weight: float | None = None,
    ) -> None:
        """
        Optimize one batch of SMPL model parameters against target data.

        Args:
            batch_start: Starting index of the batch
            batch_end: End index of the batch
            transls: Translation parameters tensor
            rots: Rotation parameters tensor
            pose_parms: Pose parameters tensor
            identity_coeff: Identity coefficients for the batch
            face_expr_coeff: Facial expression coefficients for the batch
            edges: Mesh edges tensor
            target_edges: Target edge vectors
            target_verts_batch: Target vertices for the batch
            optimizer: PyTorch optimizer
            scheduler: Optional learning rate scheduler
            edge_weight: Weight for edge loss if vertex loss is included
        """
        batched_smpl_parameters = self._get_batched_body_model_parameters(
            variables, batch_start, batch_end, self._DEVICE
        )

        # Compute SMPL vertices and edges.
        smpl_verts = self._smpl_model(**batched_smpl_parameters).vertices  # [B, V, 3]
        smpl_edges = smpl_verts[:, edges[:, 1], :] - smpl_verts[:, edges[:, 0], :]

        # Compute edge loss using absolute difference
        edge_loss = torch.abs(smpl_edges - target_edges).mean()

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

    def _p2s_fit_smpl_using_pytorch(
        self,
        target_vertices: torch.Tensor,
        single_identity: bool,
        is_tracking: bool,
        batch_size: int = 8,
    ) -> dict[str, torch.Tensor]:
        """
        Fit SMPL model to target vertices using PyTorch optimizer.

        Args:
            target_vertices: Target vertices in SMPL mesh topology for each frame
            single_identity: Whether to use a single identity (betas) for all frames
            is_tracking: Whether to use tracking information to initialize fitting parameters
            batch_size: Number of target vertices to process in each batch (default: 8)

        Returns:
            Dictionary containing fitted SMPL parameters
        """
        # Get SMPL mesh edges.
        smpl_template_mesh = trimesh.Trimesh(
            self._smpl_model.v_template.cpu().numpy(),
            self._smpl_model.faces,
            process=False,
        )
        edges = (
            torch.from_numpy(smpl_template_mesh.edges_unique.copy()).long().to(self._DEVICE)
        )

        num_frames = target_vertices.shape[0]

        # Define variables for optimization.
        variables = self._p2s_define_trainable_variables(num_frames, single_identity)

        logger.info("Initial pose optimization...")
        for batch_start in tqdm(
            range(0, num_frames, batch_size), desc="Initial pose optimization batches"
        ):
            batch_end = min(batch_start + batch_size, num_frames)

            target_verts_batch = target_vertices[batch_start:batch_end]
            target_edges = (
                target_verts_batch[:, edges[:, 1], :]
                - target_verts_batch[:, edges[:, 0], :]
            )

            optimizable_parameters = [
                ["global_orient"],
                [
                    "global_orient",
                    "body_pose",
                    "betas",
                    "left_hand_pose",
                    "right_hand_pose",
                    "expression",
                ],
            ]
            iterations = (21, 51)
            for op, it in zip(optimizable_parameters, iterations):
                optimizer = torch.optim.Adam([variables[p] for p in op], lr=0.1)
                for _ in range(it):
                    self._p2s_optimize_one_batch(
                        batch_start,
                        batch_end,
                        variables,
                        edges,
                        target_edges,
                        target_verts_batch,
                        optimizer,
                    )

            # Compute the initial global translation for each frame.
            batched_smpl_parameters = self._get_batched_body_model_parameters(
                variables, batch_start, batch_end, self._DEVICE
            )
            smpl_verts = self._smpl_model(**batched_smpl_parameters).vertices
            variables["transl"][batch_start:batch_end] = (
                (target_verts_batch - smpl_verts).mean(dim=1).detach()
            )
        variables["transl"].requires_grad_()

        # Optimize all parameters (including shape parameters betas)
        logger.info("Optimize all parameters...")
        optimizer = torch.optim.Adam(
            [v for k, v in variables.items() if k != "lbs_model_params"], lr=0.01
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50], gamma=0.1
        )
        for epoch_id in tqdm(range(100), desc="Optimize all parameters in batches"):
            for batch_start in range(0, num_frames, batch_size):
                batch_end = min(batch_start + batch_size, num_frames)
                target_verts_batch = target_vertices[batch_start:batch_end]
                target_edges = (
                    target_verts_batch[:, edges[:, 1], :]
                    - target_verts_batch[:, edges[:, 0], :]
                )

                edge_weight = 1.0 if epoch_id < 50 else 0
                self._p2s_optimize_one_batch(
                    batch_start,
                    batch_end,
                    variables,
                    edges,
                    target_edges,
                    target_verts_batch,
                    optimizer,
                    scheduler,
                    edge_weight=edge_weight,
                )

        # Return parameters as dictionary
        if single_identity:
            variables["betas"] = variables["betas"][:1].expand(num_frames, -1)
        if self.smpl_model_type == "smplx":
            return variables
        else:
            return {
                "betas": variables["betas"],
                "body_pose": variables["body_pose"],
                "global_orient": variables["global_orient"],
                "transl": variables["transl"],
            }

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
        # Select appropriate mapping file
        if direction == "smpl2mhr":
            mapping_file_path = (
                _SMPL2MHR_MAPPING_FILE
                if self.smpl_model_type == "smpl"
                else _SMPLX2MHR_MAPPING_FILE
            )
        elif direction == "mhr2smpl":
            mapping_file_path = (
                _MHR2SMPL_MAPPING_FILE
                if self.smpl_model_type == "smpl"
                else _MHR2SMPLX_MAPPING_FILE
            )
        else:
            raise ValueError(f"Invalid direction: {direction}")

        # Load mapping data
        try:
            with open(mapping_file_path, "rb") as f:
                mapping = np.load(f)
                return mapping["triangle_ids"], mapping["baryc_coords"]
        except Exception as e:
            raise ValueError(
                f"Failed to load mapping file '{mapping_file_path}': {e}. "
                f"Please ensure the mapping file exists in the resources folder."
            )

    @lru_cache
    def _load_surface_mapping_SMPL2MHR(self) -> tuple[np.ndarray, np.ndarray]:
        """Load SMPL/SMPLX to MHR surface mapping (cached)."""
        return self._load_surface_mapping("smpl2mhr")

    @lru_cache
    def _load_surface_mapping_MHR2SMPL(self) -> tuple[np.ndarray, np.ndarray]:
        """Load MHR to SMPL/SMPLX surface mapping (cached)."""
        return self._load_surface_mapping("mhr2smpl")

    def _concat_mhr_lbs_model_parameters(
        self, separate_lbs_model_parameters: dict[str, torch.Tensor]
    ) -> None:
        if (
            separate_lbs_model_parameters["scale_parms"].shape[0]
            == separate_lbs_model_parameters["pose_parms"].shape[0]
        ):
            expanded_scale_parms = separate_lbs_model_parameters["scale_parms"]
        else:
            expanded_scale_parms = separate_lbs_model_parameters["scale_parms"].expand(
                separate_lbs_model_parameters["pose_parms"].shape[0], -1
            )

        separate_lbs_model_parameters["lbs_model_params"] = torch.cat(
            [
                separate_lbs_model_parameters["transls"],
                separate_lbs_model_parameters["rots"],
                separate_lbs_model_parameters["pose_parms"],
                expanded_scale_parms,
            ],
            dim=-1,
        )

    def _mhr_para2mesh(
        self,
        mhr_parameters: dict[str, torch.Tensor],
        return_mesh: bool = True,
        batch_size: int = 8,
        verbose: bool = False,
    ) -> tuple[list[trimesh.Trimesh], np.ndarray]:
        """
        Convert MHR parameters to meshes and vertices.

        Args:
            mhr_parameters: Dictionary containing MHR parameters
            return_mesh: Whether to return meshes (default: True)
            batch_size: Number of frames to process in each batch (default: 8)

        Returns:
            Tuple of (list of meshes, numpy array of vertices)
        """
        faces = self._mhr_model.character_gpu.mesh.faces.cpu().numpy()
        meshes = []
        mhr_vertices = []
        num_samples = mhr_parameters["lbs_model_params"].shape[0]

        if verbose:
            logger.info("Converting MHR parameters to meshes...")
        for batch_start in tqdm(range(0, num_samples, batch_size)):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_parameters = self._get_batched_body_model_parameters(
                mhr_parameters, batch_start, batch_end, self._DEVICE
            )
            with torch.no_grad():
                mhr_verts = self._mhr_model(
                    identity_coeffs=batch_parameters["shape_space_params"],
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
        batch_size: int = 8,
        verbose: bool = False,
    ) -> tuple[list[trimesh.Trimesh], np.ndarray]:
        """
        Convert SMPL parameters to meshes and vertices.

        Args:
            smpl_parameters: Dictionary containing SMPL parameters
            return_mesh: Whether to return meshes (default: True)
            batch_size: Number of frames to process in each batch (default: 8)

        Returns:
            Tuple of (list of meshes, numpy array of vertices)
        """
        faces = self._smpl_model.faces
        meshes = []
        smpl_vertices = []
        num_samples = smpl_parameters["betas"].shape[0]

        if verbose:
            logger.info("Converting SMPL parameters to meshes...")
        for batch_start in tqdm(range(0, num_samples, batch_size)):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_parameters = self._get_batched_body_model_parameters(
                smpl_parameters, batch_start, batch_end, self._DEVICE
            )

            with torch.no_grad():
                smpl_output = self._smpl_model(**batch_parameters)
                smpl_vertices.append(smpl_output.vertices.detach().cpu().numpy())
        smpl_vertices = np.concatenate(smpl_vertices, axis=0)
        if return_mesh:
            for vertices in smpl_vertices:
                meshes.append(trimesh.Trimesh(vertices, faces, process=False))

        return meshes, smpl_vertices

    def _s2p_reprocess_failure_cases(
        self,
        fitting_parameter_results: dict[str, torch.Tensor],
        smpl_vertices: torch.Tensor,
        target_vertices: torch.Tensor,
        error_threshold: float = 1.0,
        batch_size: int = 8,
        smpl_parameters: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        """
        Reprocess failure cases of SMPL->MHR conversion using PyMomentum approach.

        This method identifies conversion failure cases based on average vertex distance errors
        and reprocesses them using an interpolation-based approach for better results.

        Args:
            fitting_parameter_results: Initial MHR fitting results with 'lbs_model_params',
                                     'shape_space_params', and 'face_expr_coeffs' tensors
            smpl_vertices: SMPL vertices tensor in SMPL mesh topology [B, V_s, 3]
            target_vertices: Target vertices tensor in MHR mesh topology [B, V_t, 3]
            error_threshold: Threshold for average vertex distance to identify failure cases
                           (default: 0.05)
            batch_size: Number of frames to process in each batch for memory efficiency
                       (default: 8)
            smpl_parameters: Optional dictionary containing SMPL parameters for reprocessing

        Returns:
            Updated fitting parameter results with improved parameters for failure cases
        """
        logger.info("Reprocessing failure cases with PyMomentum approach...")

        # Step 1: Compute conversion errors in average vertex distance using batch processing
        errors = self._s2p_evaluate_conversion_error(
            fitting_parameter_results, target_vertices, batch_size
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

        # Step 3: For each failure case, reprocess using interpolation approach
        for frame_idx in failure_indices:
            frame_idx = frame_idx.item()
            logger.info(f"Reprocessing failure case for frame {frame_idx}")

            # Step 3.1: If only SMPL vertices are provided, fit SMPL model to get SMPL parameters
            target_frame_vertices = target_vertices[frame_idx]  # [V, 3]
            if smpl_parameters is None:
                smpl_params = self._fit_smpl_to_vertices(smpl_vertices[frame_idx])
            else:
                smpl_params = {
                    key: smpl_parameters[key][frame_idx].unsqueeze(0)
                    for key in smpl_parameters
                }

            # Step 3.2: Interpolate SMPL parameters from all zero to target with 4 interpolations
            interpolation_steps = 4
            interpolated_vertices_sequence = []

            for step in range(interpolation_steps):
                alpha = step / interpolation_steps

                # Interpolate each parameter
                interpolated_params = {}
                for key, value in smpl_params.items():
                    zero_param = torch.zeros_like(value)
                    interpolated_params[key] = (zero_param + alpha * value).to(
                        self._DEVICE
                    )

                # Generate SMPL vertices for this interpolation step
                with torch.no_grad():
                    smpl_output = self._smpl_model(**interpolated_params)

                # Convert SMPL vertices to MHR space
                mhr_interp_vertices = self._compute_target_vertices(
                    smpl_output.vertices.detach(), "smpl2mhr"
                )[0]  # [V_mhr, 3]

                interpolated_vertices_sequence.append(mhr_interp_vertices)
            interpolated_vertices_sequence.append(target_frame_vertices)

            sequence_vertices = torch.stack(
                interpolated_vertices_sequence, dim=0
            )  # [steps+1, V, 3]

            # Step 3.3: Run _s2p_fit_mhr_using_pymomentum with is_tracking=True
            improved_results = self._s2p_fit_mhr_using_pymomentum(
                sequence_vertices, single_identity=False, is_tracking=True
            )

            # Step 3.4: Get the fitting result for the final frame and update the mhr result parameters
            final_frame_idx = -1  # Last frame in the sequence

            # Get the error after reprocessing.
            improved_verts = self._mhr_model(
                identity_coeffs=improved_results["shape_space_params"][final_frame_idx][
                    None, ...
                ].to(self._DEVICE),
                model_parameters=improved_results["lbs_model_params"][final_frame_idx][
                    None, ...
                ].to(self._DEVICE),
                face_expr_coeffs=improved_results["face_expr_coeffs"][final_frame_idx][
                    None, ...
                ].to(self._DEVICE),
                apply_correctives=True,
            )[0]

            error = torch.sqrt(
                ((improved_verts - target_frame_vertices) ** 2).sum(-1)
            ).mean()

            # Update the mhr result parameters if the error is lower than the original error.
            if error < errors[frame_idx]:
                fitting_parameter_results["lbs_model_params"][frame_idx] = (
                    improved_results["lbs_model_params"][final_frame_idx]
                )
                fitting_parameter_results["shape_space_params"][frame_idx] = (
                    improved_results["shape_space_params"][final_frame_idx]
                )
                fitting_parameter_results["face_expr_coeffs"][frame_idx] = (
                    improved_results["face_expr_coeffs"][final_frame_idx]
                )
                logger.info(
                    f"Frame {frame_idx}: Error improved from {errors[frame_idx]:.6f} to {error:.6f}"
                )
                errors[frame_idx] = error
            else:
                logger.info(
                    f"Frame {frame_idx}: Error not improved. Original error: {errors[frame_idx]:.6f}, after reprocessing: {error:.6f}"
                )

        # Step 4: Return updated conversion results
        logger.info("Failure case reprocessing completed.")
        return fitting_parameter_results, errors

    def _fit_smpl_to_vertices(
        self, target_vertices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Fit SMPL model parameters to target vertices using PyTorch optimization.

        Args:
            target_vertices: Target vertices in SMPL topology [V, 3]

        Returns:
            Dictionary containing fitted SMPL parameters
        """
        target_vertices = target_vertices.unsqueeze(0)  # [1, V, 3]

        # Initialize SMPL parameters
        betas = torch.zeros(
            1, self._smpl_model.num_betas, device=self._DEVICE, requires_grad=True
        )
        body_pose_dim = 69 if self.smpl_model_type == "smpl" else 63
        body_pose = torch.zeros(
            1, body_pose_dim, device=self._DEVICE, requires_grad=True
        )
        global_orient = torch.zeros(1, 3, device=self._DEVICE, requires_grad=True)
        transl = torch.zeros(1, 3, device=self._DEVICE, requires_grad=True)

        if self.smpl_model_type == "smplx":
            left_hand_pose = torch.zeros(
                1, self._hand_pose_dim, device=self._DEVICE, requires_grad=True
            )
            right_hand_pose = torch.zeros(
                1, self._hand_pose_dim, device=self._DEVICE, requires_grad=True
            )
            expression = torch.zeros(
                1,
                self._smpl_model.num_expression_coeffs,
                device=self._DEVICE,
                requires_grad=True,
            )
        else:
            left_hand_pose = torch.zeros(1, 6, device=self._DEVICE)
            right_hand_pose = torch.zeros(1, 6, device=self._DEVICE)
            expression = torch.zeros(1, 10, device=self._DEVICE)

        # Optimize parameters
        if self.smpl_model_type == "smplx":
            optimizer = torch.optim.Adam(
                [
                    betas,
                    body_pose,
                    global_orient,
                    transl,
                    left_hand_pose,
                    right_hand_pose,
                    expression,
                ],
                lr=0.01,
            )
        else:
            optimizer = torch.optim.Adam(
                [betas, body_pose, global_orient, transl], lr=0.01
            )

        # Get SMPL mesh edges for edge loss computation
        smpl_template_mesh = trimesh.Trimesh(
            self._smpl_model.v_template.cpu().numpy(),
            self._smpl_model.faces,
            process=False,
        )
        edges = (
            torch.from_numpy(smpl_template_mesh.edges_unique.copy()).long().to(self._DEVICE)
        )

        # Compute target edges from target vertices
        target_edges = (
            target_vertices[:, edges[:, 1], :] - target_vertices[:, edges[:, 0], :]
        )

        for _ in range(100):  # Quick fitting
            # Generate SMPL vertices
            if self.smpl_model_type == "smplx":
                smpl_output = self._smpl_model(
                    betas=betas,
                    body_pose=body_pose,
                    global_orient=global_orient,
                    transl=transl,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    jaw_pose=torch.zeros(1, 1, 3, device=self._DEVICE),
                    leye_pose=torch.zeros(1, 1, 3, device=self._DEVICE),
                    reye_pose=torch.zeros(1, 1, 3, device=self._DEVICE),
                    expression=expression,
                )
            else:
                smpl_output = self._smpl_model(
                    betas=betas,
                    body_pose=body_pose,
                    global_orient=global_orient,
                    transl=transl,
                )

            # Compute edge loss
            predicted_vertices = smpl_output.vertices
            predicted_edges = (
                predicted_vertices[:, edges[:, 1], :]
                - predicted_vertices[:, edges[:, 0], :]
            )
            edge_loss = torch.square(predicted_edges - target_edges).mean()

            # Compute vertex loss
            vertex_loss = torch.square(predicted_vertices - target_vertices).mean()

            # Combine edge and vertex loss (equal weighting)
            loss = 100.0 * edge_loss + vertex_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Return parameters
        result = {
            "betas": betas.detach(),
            "body_pose": body_pose.detach(),
            "global_orient": global_orient.detach(),
            "transl": transl.detach(),
        }

        if self.smpl_model_type == "smplx":
            result["left_hand_pose"] = left_hand_pose.detach()
            result["right_hand_pose"] = right_hand_pose.detach()
            result["expression"] = expression.detach()

        return result

    def _s2p_evaluate_conversion_error(
        self,
        fitting_parameter_results: dict[str, torch.Tensor],
        target_vertices: torch.Tensor,
        batch_size: int,
    ) -> np.ndarray:
        """
        Compute conversion errors in average vertex distance using batch processing.

        Args:
            fitting_parameter_results: Dictionary containing MHR fitting results with
                                     'lbs_model_params', 'shape_space_params', and 'face_expr_coeffs'
            target_vertices: Target vertices tensor in MHR mesh topology [B, V, 3]
            batch_size: Number of frames to process in each batch for memory efficiency

        Returns:
            Array of average vertex distance errors for each frame
        """
        num_frames = len(fitting_parameter_results["lbs_model_params"])
        error_lists = []

        # Process frames in batches for better memory efficiency
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)
            batch_indices = range(batch_start, batch_end)

            # Prepare batch parameters
            lbs_params_batch = torch.stack(
                [
                    fitting_parameter_results["lbs_model_params"][i]
                    for i in batch_indices
                ],
                dim=0,
            )
            shape_params_batch = torch.stack(
                [
                    fitting_parameter_results["shape_space_params"][i]
                    for i in batch_indices
                ],
                dim=0,
            )
            face_expr_batch = torch.stack(
                [
                    fitting_parameter_results["face_expr_coeffs"][i]
                    for i in batch_indices
                ],
                dim=0,
            )

            # Generate vertices for the entire batch
            batch_vertices = self._mhr_model(
                identity_coeffs=shape_params_batch.to(self._DEVICE),
                model_parameters=lbs_params_batch.to(self._DEVICE),
                face_expr_coeffs=face_expr_batch.to(self._DEVICE),
                apply_correctives=True,
            )

            batch_target_vertices = target_vertices[batch_start:batch_end]
            batch_errors = torch.sqrt(
                ((batch_vertices - batch_target_vertices) ** 2).sum(-1)
            ).mean(1)
            batch_errors = batch_errors.detach().cpu().numpy().tolist()

            error_lists.extend(batch_errors)

        return np.array(error_lists)

    def _p2s_evaluate_conversion_error(
        self,
        fitting_parameter_results: dict[str, torch.Tensor],
        target_vertices: torch.Tensor,
        batch_size: int,
    ) -> np.ndarray:
        """
        Compute conversion errors in average vertex distance using batch processing.

        Args:
            fitting_parameter_results: Dictionary containing SMPL fitting results with
                                     'betas', 'body_pose', 'global_orient', 'transl', etc.
            target_vertices: Target vertices tensor in SMPL mesh topology [B, V, 3]
            batch_size: Number of frames to process in each batch for memory efficiency

        Returns:
            Array of average vertex distance errors for each frame
        """
        num_frames = len(fitting_parameter_results["betas"])
        error_lists = []

        # Process frames in batches for better memory efficiency
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)
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
                # Default for SMPL model
                left_hand_pose_batch = torch.zeros(
                    len(batch_indices), 6, device=self._DEVICE
                )
                right_hand_pose_batch = torch.zeros(
                    len(batch_indices), 6, device=self._DEVICE
                )

            # Generate additional SMPLX parameters if needed
            current_batch_size = batch_end - batch_start
            jaw_pose_batch = torch.zeros(
                [current_batch_size, 1, 3], device=self._DEVICE
            )
            leye_pose_batch = torch.zeros(
                [current_batch_size, 1, 3], device=self._DEVICE
            )
            reye_pose_batch = torch.zeros(
                [current_batch_size, 1, 3], device=self._DEVICE
            )

            # Handle expression parameters for SMPLX
            if "expression" in fitting_parameter_results:
                expression_batch = torch.stack(
                    [fitting_parameter_results["expression"][i] for i in batch_indices],
                    dim=0,
                )
            else:
                # Default expression parameters
                expression_batch = torch.zeros(
                    current_batch_size,
                    self._smpl_model.num_expression_coeffs,
                    device=self._DEVICE,
                )

            # Generate vertices for the entire batch using SMPL model
            smpl_output = self._smpl_model(
                betas=betas_batch.to(self._DEVICE),
                body_pose=body_pose_batch.to(self._DEVICE),
                global_orient=global_orient_batch.to(self._DEVICE),
                transl=transl_batch.to(self._DEVICE),
                left_hand_pose=left_hand_pose_batch.to(self._DEVICE),
                right_hand_pose=right_hand_pose_batch.to(self._DEVICE),
                jaw_pose=jaw_pose_batch,
                leye_pose=leye_pose_batch,
                reye_pose=reye_pose_batch,
                expression=expression_batch.to(self._DEVICE),
            )
            batch_vertices = smpl_output.vertices

            batch_target_vertices = target_vertices[batch_start:batch_end]
            batch_errors = torch.sqrt(
                ((batch_vertices - batch_target_vertices) ** 2).sum(-1)
            ).mean(1)
            batch_errors = batch_errors.detach().cpu().numpy().tolist()

            error_lists.extend(batch_errors)

        return np.array(error_lists)

    def _get_batched_body_model_parameters(
        self,
        parameter_dict: dict[str, torch.Tensor],
        batch_start: int,
        batch_end: int,
        device: torch.device,
        is_mhr_parameters: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Get SMPL parameters for a batch of frames.

        Args:
            parameter_dict: Dictionary containing MHR fitting results with
                                    'lbs_model_params', 'shape_space_params', and 'face_expr_coeffs'
            batch_start: Start index of the batch
            batch_end: End index of the batch
            device: Device to place the tensors on

        Returns:
            Dictionary containing SMPL parameters for the batch
        """
        batched_parameter_dict = {}

        for k, v in parameter_dict.items():
            if v.shape[0] < batch_end:
                batched_parameter_dict[k] = v.expand(batch_end - batch_start, -1)
            else:
                batched_parameter_dict[k] = v[batch_start:batch_end].to(device)

        # Not for SMPLX model, we need to make sure jaw_pose, leye_pose,
        # reye_pose, left_hand_pose, right_hand_pose, and expression exist
        # and have the right batch dimension.
        if (not is_mhr_parameters) and self.smpl_model_type == "smplx":
            batch_size = batch_end - batch_start
            if "jaw_pose" not in batched_parameter_dict:
                batched_parameter_dict["jaw_pose"] = torch.zeros(
                    [batch_size, 1, 3], device=device
                )
            if "leye_pose" not in batched_parameter_dict:
                batched_parameter_dict["leye_pose"] = torch.zeros(
                    [batch_size, 1, 3], device=device
                )
            if "reye_pose" not in batched_parameter_dict:
                batched_parameter_dict["reye_pose"] = torch.zeros(
                    [batch_size, 1, 3], device=device
                )
            hand_pose_dim = 6 if self._smpl_model.use_pca else 45
            if "left_hand_pose" not in batched_parameter_dict:
                batched_parameter_dict["left_hand_pose"] = torch.zeros(
                    [batch_size, hand_pose_dim], device=device
                )
            if "right_hand_pose" not in batched_parameter_dict:
                batched_parameter_dict["right_hand_pose"] = torch.zeros(
                    [batch_size, hand_pose_dim], device=device
                )
            expression_dim = self._smpl_model.num_expression_coeffs
            if "expression" not in batched_parameter_dict:
                batched_parameter_dict["expression"] = torch.zeros(
                    [batch_size, expression_dim], device=device
                )

        return batched_parameter_dict
