"""PyMomentum-based fitting classes for MHR model conversion

This module provides PyMomentum-based optimization tools for fitting MHR models
to target vertex positions (assuming we have target positions for each MHR
vertices). It implements a hierarchical optimization approach that progressively
refines model parameters from rigid transformations to full body shape.

Example:
    Basic usage for fitting a MHR model to target vertices:

    >>> fitter = PyMomentumModelFitting(
    ...     mhr_model=mhr_model,
    ...     num_subsampled_mhr_vertices=4000
    ... )
    >>> # Fit model using hierarchical optimization
    >>> fitter.fit(
    ...     target_vertices=target_vertices,
    ...     skip_global_stages=False,
    ...     exclude_expression=False,  # True if you don't want to optimize for face expression.
    ...     verbose=True
    ... )
    >>>
    >>> # Get fitting results
    >>> results = fitter.get_fitting_results()
    >>> lbs_params = results["lbs_model_params"]
    >>> shape_params = results["shape_space_params"]
"""

import dataclasses
import logging
from functools import lru_cache

import numpy as np
import pymomentum.geometry as pym_geo
import pymomentum.solver as pym_solve
import torch
from mhr.mhr import MHR
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PyMomentumOptimizationStage:
    """Configuration for a single optimization stage in hierarchical fitting.

    Attributes:
        active_parameter_mask: Boolean mask indicating which parameters are active in this stage
        vertex_cons_weights: Weights for vertex constraints during optimization
        information: Descriptive string about what this stage optimizes
    """

    active_parameter_mask: torch.Tensor
    vertex_cons_weights: torch.Tensor
    information: str


class PyMomentumModelFitting:
    """Main class for fitting MHR models using PyMomentum optimization.

    This class provides a hierarchical optimization approach for fitting MHR body models
    to target vertex positions. The optimization proceeds through multiple stages, starting
    with rigid transformations and progressively adding more degrees of freedom.

    Attributes:
        _mhr_model: The MHR body model to be fitted
        _num_blendshapes: Number of identity blendshapes in the model
        _bs_character: PyMomentum character with blendshapes
        _solved_parameters: Current optimized parameters
        _constant_parameter_mask: Mask for parameters that remain constant
        _subsampled_vertex_mask: Mask for vertex subsampling to improve performance
        _hierarchical_parameter_masks: Cached parameter masks for optimization stages
        _hierarchical_vertex_masks: Cached vertex masks for optimization stages
    """

    def __init__(
        self,
        mhr_model: MHR,
        num_subsampled_mhr_vertices: int = 4000,
    ) -> None:
        """Initialize the PyMomentum model fitting instance.

        Args:
            mhr_model: The MHR body model to fit
            num_subsampled_mhr_vertices: Number of vertices to use for efficient fitting.
                                           Reduces computation by clustering similar vertices.
        """
        self._mhr_model = mhr_model.to(
            "cpu"
        )  # PyMomentum IK solver only works on CPU
        self._num_blendshapes = self._mhr_model.identity_model.blend_shapes.shape[0]
        self._num_expression_blendshapes = (
            self._mhr_model.face_expressions_model.blend_shapes.shape[0]
        )

        # Create character with body blendshapes
        self._bs_character = self._create_body_shape_expression_character()

        self._num_parameters = len(self._bs_character.parameter_transform.names)

        # Initialize model parameters based on bs_character's parameter transform size
        self._solved_parameters = torch.zeros(self._num_parameters, device="cpu")

        # Initialize constant parameter mask
        self._constant_parameter_mask = torch.zeros(
            self._num_parameters,
            device="cpu",
        ).bool()

        self._hierarchical_parameter_masks: list[torch.Tensor] = [torch.Tensor()]
        self._hierarchical_vertex_masks: list[torch.Tensor] = [torch.Tensor()]

        self._subsampled_vertex_mask: torch.Tensor = self._subsample_mhr_vertices(
            num_subsampled_mhr_vertices
        )

    def _to_tensor(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Convert input data to CPU torch tensor with double precision.

        Args:
            data: Input tensor or numpy array

        Returns:
            CPU tensor with double precision
        """
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).double().cpu()
        return data.double().cpu()

    def set_initial_parameters(
        self,
        initial_parameters: torch.Tensor | np.ndarray,
    ) -> None:
        """
        Set the initial parameters for the model.

        Args:
            initial_parameters: A tensor of initial parameters for the model.
        """
        self._solved_parameters = self._to_tensor(initial_parameters)

    def set_constant_parameters(
        self,
        constant_parameter_mask: torch.Tensor | np.ndarray,
        constant_parameter: torch.Tensor | np.ndarray,
    ):
        """Set specific parameters to remain constant during optimization.

        Args:
            constant_parameter_mask: Boolean tensor indicating which parameters to keep constant
            constant_parameter: Values to set for the constant parameters
        """
        mask = self._to_tensor(constant_parameter_mask).bool()
        values = self._to_tensor(constant_parameter)

        self._constant_parameter_mask = mask
        self._solved_parameters[mask] = values

    def fit(
        self,
        target_vertices: torch.Tensor | np.ndarray,
        skip_global_stages: bool = False,
        exclude_expression: bool = False,
        verbose: bool = False,
    ):
        """Fit the MHR model to target vertex positions using hierarchical optimization.

        The fitting process uses a multi-stage approach:
        1. Rigid transformations (global pose)
        2. Torso and limb roots
        3. Full body without hands
        4. Complete body including hands and shape parameters

        Args:
            target_vertices: Target vertex positions (N x 3) to fit the model to
            skip_global_stages: Skip the first two global stages (recommended when good initialization can be obtained).
            verbose: Whether to print optimization progress information
        """
        target_vertices = self._to_tensor(target_vertices)
        parameter_masks, vertex_weights, infos = self._create_hierarchical_masks(exclude_expression=exclude_expression)

        # Run optimization stages
        for i, (mask, weight, info) in enumerate(
            zip(parameter_masks, vertex_weights, infos)
        ):
            if skip_global_stages and i < 2:
                continue
            if verbose:
                logger.info(f"Starting {info}...")

            stage = PyMomentumOptimizationStage(
                active_parameter_mask=mask & ~self._constant_parameter_mask,
                vertex_cons_weights=weight,
                information=info,
            )
            self._optimize_one_stage(target_vertices, stage)

    def reset(self):
        """Reset all model parameters to zero."""
        self._solved_parameters *= 0.0

    def get_fitting_results(self) -> dict[str, torch.Tensor]:
        """Get the optimized model parameters after fitting.

        Returns:
            Dictionary containing:
                - 'lbs_model_params': Linear blend skinning parameters for pose/expression
                - 'shape_space_params': Identity shape parameters (blendshape coefficients)
        """
        return {
            "lbs_model_params": self._solved_parameters[
                : -(self._num_blendshapes + self._num_expression_blendshapes)
            ].float(),
            "shape_space_params": self._solved_parameters[
                -(
                    self._num_blendshapes + self._num_expression_blendshapes
                ) : -self._num_expression_blendshapes
            ].float(),
            "face_expr_coeffs": self._solved_parameters[
                -self._num_expression_blendshapes :
            ].float(),
        }

    def _create_body_shape_expression_character(self) -> pym_geo.Character:
        """
        Create a PyMomentum character with blendshapes.

        Returns:
            The created character object.
        """
        character = self._mhr_model.character

        mean_body_vertices = self._mhr_model.identity_model.mean_shape.numpy()
        shapes_basis = self._mhr_model.identity_model.blend_shapes.numpy()
        shapes_basis = shapes_basis[: self._num_blendshapes, :]

        normals = pym_geo.compute_vertex_normals(
            torch.FloatTensor(mean_body_vertices),
            torch.LongTensor(character.mesh.faces),
        )
        new_mesh = pym_geo.Mesh(
            mean_body_vertices,
            character.mesh.faces,
            normals=normals.numpy(),
            texcoords=character.mesh.texcoords,
            texcoord_faces=character.mesh.texcoord_faces,
        )
        bs_character = character.with_mesh_and_skin_weights(
            new_mesh, character.skin_weights
        )
        expression_shapes_basis = (
            self._mhr_model.face_expressions_model.blend_shapes.numpy()
        )
        expression_shapes_basis = expression_shapes_basis[
            : self._num_expression_blendshapes
        ]

        all_shapes_basis = np.concatenate(
            [shapes_basis, expression_shapes_basis], axis=0
        )

        bs_character = bs_character.with_blend_shape(
            pym_geo.BlendShape.from_tensors(mean_body_vertices, all_shapes_basis)
        )

        return bs_character

    def _get_vertex_weight_from_parameter_mask(
        self, parameter_mask, parameter2joints_mapping, joint2vertex_weight
    ):
        """Compute vertex weights based on which parameters are active in current stage.

        Args:
            parameter_mask: Boolean mask of active parameters
            parameter2joints_mapping: Mapping from parameters to affected joints
            joint2vertex_weight: Skinning weights from joints to vertices

        Returns:
            Vertex weights for the current optimization stage
        """
        # Joints that are affected by activated parameters.
        affected_joints_mask = parameter2joints_mapping[:, parameter_mask].sum(1) > 0
        # Joints that are completely not affected by any of the activated parameters.
        not_affected_joints_mask = (
            parameter2joints_mapping[:, ~parameter_mask].sum(1) > 0
        )
        # Joints that are affected only by the activated parameters.
        fully_affected_joints_mask = np.logical_and(
            affected_joints_mask, ~not_affected_joints_mask
        )

        # Get the sum of the corresponding skinning weights for the fully affected joints.
        vertex_weight = joint2vertex_weight[:, fully_affected_joints_mask].sum(1)
        return torch.from_numpy(vertex_weight).cpu().double()

    # Class constants for body part detection
    _BODY_PARTS = {"spine", "neck", "head", "shoulder", "clavicle", "upleg", "uparm"}
    _HAND_PARTS = {"index", "middle", "ring", "pinky", "thumb", "wrist"}

    def _contains_parts(self, name: str, parts: set[str]) -> bool:
        """Check if parameter name contains any of the specified parts.

        Args:
            name: Parameter name to check
            parts: Set of part names to look for

        Returns:
            True if name contains any part from the set
        """
        return any(part in name for part in parts)

    def _contains_body_part(self, name: str) -> bool:
        """Check if parameter name contains core body part keywords."""
        return self._contains_parts(name, self._BODY_PARTS)

    def _contains_hand_part(self, name: str) -> bool:
        """Check if parameter name contains hand/finger keywords."""
        return self._contains_parts(name, self._HAND_PARTS)

    @lru_cache
    def _create_hierarchical_masks(
        self,
        exclude_expression: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
        """Create hierarchical parameter and vertex masks for staged optimization.

        Creates four optimization stages:
        1. Rigid transformations only
        2. Rigid + torso and limb roots
        3. Full body excluding hands
        4. Complete body including hands

        Returns:
            Tuple containing:
                - List of parameter masks for each stage
                - List of vertex weights for each stage
                - List of descriptive strings for each stage
        """
        lbs_parameter_names = self._mhr_model.character.parameter_transform.names
        parameter2joints_mapping = (
            self._mhr_model.character.parameter_transform.transform.numpy()
            .reshape(-1, 7, len(lbs_parameter_names))
            .sum(1)
        ).astype(bool)  # (J, P)
        num_joints = parameter2joints_mapping.shape[0]
        skinning_weight_matrix = np.zeros((18439, num_joints))
        v_indx, j_indx = np.where(self._mhr_model.character.skin_weights.index)
        skinning_weight_matrix[
            v_indx, self._mhr_model.character.skin_weights.index[v_indx, j_indx]
        ] = self._mhr_model.character.skin_weights.weight[v_indx, j_indx]

        full_parameter_masks = []
        vertex_weights = []
        infos = []

        def _add_stage(lbs_mask: torch.Tensor, info: str):
            """Helper to add a stage to the optimization lists.

            Args:
                lbs_mask: Boolean mask for LBS parameters in this stage
                info: Descriptive string for this optimization stage
            """
            # Create full parameter mask including blendshapes
            full_mask = torch.cat(
                [
                    lbs_mask,
                    torch.zeros(
                        self._num_blendshapes + self._num_expression_blendshapes,
                        dtype=lbs_mask.dtype,
                    ),
                ]
            )
            vertex_weight = self._get_vertex_weight_from_parameter_mask(
                lbs_mask, parameter2joints_mapping, skinning_weight_matrix
            )
            full_parameter_masks.append(full_mask)
            vertex_weights.append(vertex_weight)
            infos.append(info)

        # Level 0: Rigid transformations only
        lbs_mask = (
            self._mhr_model.character.parameter_transform.rigid_parameters.clone()
        )
        _add_stage(lbs_mask, "level 0: rigid transform")

        # Level 1: Add torso and limb roots
        for i, name in enumerate(lbs_parameter_names):
            if self._contains_body_part(name):
                lbs_mask[i] = True
        _add_stage(lbs_mask, "level 1: torso and limbs' roots")

        # Level 2: Full body excluding hands
        lbs_mask = torch.ones_like(lbs_mask)
        for i, name in enumerate(lbs_parameter_names):
            if self._contains_hand_part(name):
                lbs_mask[i] = False
        _add_stage(lbs_mask, "level 2: limbs")

        # Level 3: All parameters including shape and expression.
        full_parameter_masks.append(torch.ones_like(full_parameter_masks[-1]))
        if exclude_expression:
            full_parameter_masks[-1][-self._num_expression_blendshapes:] = False
        vertex_weights.append(torch.ones_like(vertex_weights[-1]))
        infos.append("level 3: all")

        return full_parameter_masks, vertex_weights, infos

    def _subsample_mhr_vertices(
        self, num_subsampled_mhr_vertices: int
    ) -> torch.Tensor:
        """Subsample MHR vertices using clustering for computational efficiency.

        Args:
            num_subsampled_mhr_vertices: The number of vertices to subsample

        Returns:
            Boolean mask tensor indicating which vertices are selected
        """
        # Get the vertex positions
        vertex_positions = self._mhr_model.character.mesh.vertices

        # Sample the vertices
        selected_indices = _clustering_based_sampling(
            vertex_positions, num_subsampled_mhr_vertices
        )

        # Return the subsampled vertex mask
        mask = torch.zeros(vertex_positions.shape[0], dtype=torch.bool)
        mask[selected_indices] = True
        return mask

    def _get_solver_options(
        self, stage: PyMomentumOptimizationStage
    ) -> pym_solve.SolverOptions:
        """Get optimized solver options for PyMomentum optimization.

        Args:
            stage: The current optimization stage (currently unused, but allows
                  for stage-specific optimization in the future)

        Returns:
            SolverOptions configured for efficient MHR model fitting
        """
        return pym_solve.SolverOptions(
            linear_solver=pym_solve.LinearSolverType.Cholesky,
            levmar_lambda=0.01,
            min_iter=2,
            max_iter=5,
            threshold=1.0,
            line_search=False,
        )

    def _optimize_one_stage(
        self, target_vertices: torch.Tensor, stage: PyMomentumOptimizationStage
    ):
        """Optimize model parameters for a single hierarchical stage.

        Args:
            target_vertices: Target vertex positions to fit to
            stage: Configuration for the current optimization stage
        """
        # Define vertex correspondence for vertex loss.
        num_vertices = target_vertices.shape[0]
        vertex_cons_vertices = torch.arange(num_vertices)
        vertex_mask = stage.vertex_cons_weights > 0
        vertex_mask = vertex_mask & self._subsampled_vertex_mask

        vertex_cons_vertices = vertex_cons_vertices[vertex_mask]
        target_vertices = target_vertices[vertex_mask]

        # Get stage-specific solver options
        solver_options = self._get_solver_options(stage)

        solved_parameters = pym_solve.solve_ik(
            character=self._bs_character,
            active_parameters=stage.active_parameter_mask,
            model_parameters_init=self._solved_parameters.clone(),
            options=solver_options,
            active_error_functions=[
                pym_solve.ErrorFunctionType.Limit,
                pym_solve.ErrorFunctionType.Vertex,
            ],
            error_function_weights=torch.FloatTensor([1.0, 1.0]),
            vertex_cons_type=pym_solve.VertexConstraintType.Position,
            vertex_cons_vertices=vertex_cons_vertices,
            vertex_cons_weights=stage.vertex_cons_weights[vertex_mask],
            vertex_cons_target_positions=target_vertices,
        )
        self._solved_parameters = solved_parameters
        return


def _clustering_based_sampling(points, p):
    """Subsample points using K-means clustering for computational efficiency.

    Uses K-means clustering to group similar vertices and selects representative
    points from each cluster. This maintains geometric coverage while reducing
    the number of vertices used in optimization.

    Args:
        points: Array of 3D vertex positions (N x 3)
        p: Target number of points to select

    Returns:
        List of selected vertex indices
    """
    n = points.shape[0]

    # Use more clusters than needed, then sample from each
    n_clusters = min(int(p * 1.5), n)  # Use 3x more clusters than needed

    # Cluster the points
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
    labels = kmeans.fit_predict(points)

    selected_indices = []

    # Sample one point from each cluster (round-robin style)
    for cluster_id in range(n_clusters):
        if len(selected_indices) >= p:
            break
        cluster_points = np.where(labels == cluster_id)[0]
        if len(cluster_points) > 0:
            # Pick the point closest to cluster center
            cluster_center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(points[cluster_points] - cluster_center, axis=1)
            best_idx = cluster_points[np.argmin(distances)]
            selected_indices.append(best_idx)

    return selected_indices[:p]
