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

"""PyMomentum-based fitting classes for MHR model conversion

This module provides PyMomentum-based optimization tools for fitting MHR models
to target vertex positions. It implements a hierarchical optimization approach that progressively
refines model parameters from rigid transformations to full body shape.
"""

import dataclasses
import logging
from functools import lru_cache

import numpy as np
import pymomentum.solver as pym_solve
import torch
from mhr.mhr import MHR
from sklearn.cluster import KMeans

from file_assets import HEAD_HAND_MASK_FILE

# There are 204 parameters for the MHR rig, including global rigid transform,
# joint rotations, and joint scales.
_NUM_RIG_PARAMETERS = 204
# In total there are 45 identity blendshapes for MHR.
_NUM_BODY_BLENDSHAPES = 20  # The first 20 are for body shape.
_NUM_HEAD_BLENDSHAPES = 20  # The next 20 are for head shape.
_NUM_HAND_BLENDSHAPES = 5   # The last 5 are for hand shape.

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PyMomentumOptimizationStage:
    """Configuration for a single optimization stage in hierarchical fitting.

    Attributes:
        active_parameter_mask: Boolean mask indicating which parameters are active in this stage
        vertex_cons_weights: Weights for vertex constraints during optimization
        info: Descriptive string about what this stage optimizes
    """

    active_parameter_mask: torch.Tensor
    vertex_cons_weights: torch.Tensor
    info: str
    constant_parameter_mask: torch.Tensor | None = None
    levmar_lambda: float | None = None


class PyMomentumModelFitting:
    """Main class for fitting MHR models using PyMomentum optimization.

    This class provides a hierarchical optimization approach for fitting MHR body models
    to target vertex positions. The optimization proceeds through multiple stages, starting
    with rigid transformations and progressively adding more degrees of freedom.
    """

    def __init__(
        self,
        mhr_model: MHR,
        subsampled_mhr_vertex_mask: torch.Tensor | None = None,
        num_subsampled_mhr_vertices: int = 4000,
    ) -> None:
        """Initialize the PyMomentum model fitting instance.

        Args:
            mhr_model: The MHR body model to fit
            subsampled_mhr_vertex_mask: Optional pre-computed vertex mask for subsampling
            num_subsampled_mhr_vertices: Number of vertices to use for efficient fitting.
        """
        self._mhr_model = mhr_model.to("cpu")
        self._num_blendshapes = self._mhr_model.get_num_identity_blendshapes()
        self._num_expression_blendshapes = (
            self._mhr_model.get_num_face_expression_blendshapes()
        )
        self._num_vertices = self._mhr_model.character.mesh.vertices.shape[0]

        # Create character with body blendshapes
        self._bs_character = self._mhr_model.character

        self._num_parameters = len(self._bs_character.parameter_transform.names)

        # Initialize model parameters
        self._solved_parameters = torch.zeros(
            self._num_parameters, dtype=torch.float64, device="cpu"
        )

        # Initialize constant parameter mask
        self._constant_parameter_mask = torch.zeros(
            self._num_parameters,
            device="cpu",
        ).bool()

        self._hierarchical_parameter_masks: list[torch.Tensor] = [torch.Tensor()]
        self._hierarchical_vertex_masks: list[torch.Tensor] = [torch.Tensor()]

        if subsampled_mhr_vertex_mask is not None:
            self._subsampled_vertex_mask = subsampled_mhr_vertex_mask
        else:
            self._subsampled_vertex_mask: torch.Tensor = self._subsample_mhr_vertices(
                num_subsampled_mhr_vertices
            )

        # Load head, hand, and body masks.
        self._head_mask, self._hand_mask, self._body_mask = self._load_head_hand_mask()

    def _to_tensor(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Convert input data to CPU torch tensor with double precision."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).double().cpu()
        return data.double().cpu()

    def set_initial_parameters(
        self,
        initial_parameters: torch.Tensor | np.ndarray,
    ) -> None:
        """Set the initial parameters for the model."""
        self._solved_parameters = self._to_tensor(initial_parameters)

    def set_constant_parameters(
        self,
        constant_parameter_mask: torch.Tensor | np.ndarray,
        constant_parameter: torch.Tensor | np.ndarray,
    ) -> None:
        """Set specific parameters to remain constant during optimization."""
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
    ) -> None:
        """Fit the MHR model to target vertex positions using hierarchical optimization."""
        target_vertices = self._to_tensor(target_vertices)
        stages = self._create_hierarchical_masks(
            exclude_expression=exclude_expression,
        )

        # Run optimization stages
        start_global_rigid_params = self._solved_parameters[:6].clone()
        for stage in stages:
            if verbose:
                logger.info("Starting %s...", stage.info)
            if stage.constant_parameter_mask is not None:
                self.set_constant_parameters(
                    stage.constant_parameter_mask,
                    self._solved_parameters.clone()[stage.constant_parameter_mask],
                )
            # Restore the initial global rigid transform for the whole body.
            if "Stage 2" in stage.info:
                self._solved_parameters[:6] = start_global_rigid_params

            if skip_global_stages:
                if "Stage 2" in stage.info or "Stage 3" in stage.info:
                    continue

            stage.active_parameter_mask = (
                stage.active_parameter_mask & ~self._constant_parameter_mask
            )
            self._optimize_one_stage(target_vertices, stage)

    def reset(self) -> None:
        """Reset all model parameters to zero."""
        self._solved_parameters *= 0.0
        self._constant_parameter_mask = (self._constant_parameter_mask * 0.0).bool()

    def get_fitting_results(self) -> dict[str, torch.Tensor]:
        """Get the optimized model parameters after fitting."""
        return {
            "lbs_model_params": self._solved_parameters[
                : -(self._num_blendshapes + self._num_expression_blendshapes)
            ].float(),
            "identity_coeffs": self._solved_parameters[
                -(
                    self._num_blendshapes + self._num_expression_blendshapes
                ) : -self._num_expression_blendshapes
            ].float(),
            "face_expr_coeffs": self._solved_parameters[
                -self._num_expression_blendshapes :
            ].float(),
        }

    def _get_vertex_weight_from_parameter_mask(
        self,
        parameter_mask: np.ndarray,
        parameter2joints_mapping: np.ndarray,
        joint2vertex_weight: np.ndarray,
    ) -> torch.Tensor:
        """Compute vertex weights based on which parameters are active in current stage."""
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

    _BODY_PARTS = {"spine", "neck", "head", "shoulder", "clavicle", "upleg", "uparm"}
    _HAND_PARTS = {"index", "middle", "ring", "pinky", "thumb", "wrist"}

    def _contains_parts(self, name: str, parts: set[str]) -> bool:
        """Check if parameter name contains any of the specified parts."""
        return any(part in name for part in parts)

    def _contains_body_part(self, name: str) -> bool:
        """Check if parameter name contains core body part keywords."""
        return self._contains_parts(name, self._BODY_PARTS)

    def _contains_hand_part(self, name: str) -> bool:
        """Check if parameter name contains hand/finger keywords."""
        return self._contains_parts(name, self._HAND_PARTS)

    def _add_stage(
        self,
        lbs_mask: torch.Tensor,
        blendshapes_mask: torch.Tensor,
        info: str,
        parameter2joints_mapping: np.ndarray,
        skinning_weight_matrix: np.ndarray,
        blendshapes_vertex_mask: torch.Tensor | None = None,
        vertex_weight: torch.Tensor | None = None,
        constant_parameter_mask: torch.Tensor | None = None,
        exclude_expression: bool = True,
        levmar_lambda: float | None = None,
    ) -> PyMomentumOptimizationStage:
        """Helper to add a stage to the optimization lists."""
        expression_parameter_mask = torch.ones(self._num_expression_blendshapes)
        if exclude_expression:
            expression_parameter_mask *= 0
        full_mask = torch.cat(
            [
                lbs_mask,
                blendshapes_mask,
                expression_parameter_mask.bool(),
            ]
        )
        if vertex_weight is None:
            vertex_weight_for_lbs = self._get_vertex_weight_from_parameter_mask(
                lbs_mask, parameter2joints_mapping, skinning_weight_matrix
            )
            if blendshapes_vertex_mask is not None:
                vertex_weight = vertex_weight_for_lbs + blendshapes_vertex_mask
            else:
                vertex_weight = vertex_weight_for_lbs
            if not exclude_expression:
                vertex_weight += self._head_mask

        return PyMomentumOptimizationStage(
            active_parameter_mask=full_mask,
            vertex_cons_weights=vertex_weight,
            constant_parameter_mask=constant_parameter_mask,
            levmar_lambda=levmar_lambda,
            info=info,
        )

    @lru_cache
    def _create_hierarchical_masks(
        self,
        exclude_expression: bool = False,
    ) -> list[PyMomentumOptimizationStage]:
        """Create hierarchical parameter and vertex masks for staged optimization."""
        lbs_parameter_names = self._bs_character.parameter_transform.names[
            :_NUM_RIG_PARAMETERS
        ]
        parameter2joints_mapping = (
            self._bs_character.parameter_transform.transform.numpy()
            .reshape(-1, 7, len(self._bs_character.parameter_transform.names))
            .sum(1)
        )[..., :_NUM_RIG_PARAMETERS].astype(bool)
        num_joints = parameter2joints_mapping.shape[0]
        skinning_weight_matrix = np.zeros((self._num_vertices, num_joints))
        v_indx, j_indx = np.where(self._bs_character.skin_weights.index)
        skinning_weight_matrix[
            v_indx, self._bs_character.skin_weights.index[v_indx, j_indx]
        ] = self._bs_character.skin_weights.weight[v_indx, j_indx]

        stages: list[PyMomentumOptimizationStage] = []

        # Level 0: Face rigid transformation
        lbs_mask = self._bs_character.parameter_transform.rigid_parameters.clone()[
            :_NUM_RIG_PARAMETERS
        ]
        for i, name in enumerate(lbs_parameter_names):
            if "neck" in name or "head" in name:
                lbs_mask[i] = True
        blendshapes_mask = torch.zeros(self._num_blendshapes, dtype=lbs_mask.dtype)
        vertex_weight = self._head_mask
        stages.append(
            self._add_stage(
                lbs_mask,
                blendshapes_mask,
                info="Stage 0: face rigid",
                parameter2joints_mapping=parameter2joints_mapping,
                skinning_weight_matrix=skinning_weight_matrix,
                vertex_weight=vertex_weight,
                levmar_lambda=0.001,
            )
        )

        # Level 1: Face identity and expression
        blendshapes_mask[_NUM_BODY_BLENDSHAPES:-_NUM_HAND_BLENDSHAPES] = True
        stages.append(
            self._add_stage(
                lbs_mask,
                blendshapes_mask,
                info="Stage 1.0: face identity and expression",
                parameter2joints_mapping=parameter2joints_mapping,
                skinning_weight_matrix=skinning_weight_matrix,
                vertex_weight=vertex_weight,
                exclude_expression=True,
                levmar_lambda=0.001,
            )
        )
        if not exclude_expression:
            stages.append(
                self._add_stage(
                    lbs_mask,
                    blendshapes_mask,
                    info="Stage 1.1: face expression",
                    parameter2joints_mapping=parameter2joints_mapping,
                    skinning_weight_matrix=skinning_weight_matrix,
                    vertex_weight=vertex_weight,
                    exclude_expression=exclude_expression,
                    levmar_lambda=0.001,
                )
            )

        constant_face_parameter_mask = torch.cat(
            [
                torch.zeros_like(lbs_mask),
                torch.zeros(_NUM_BODY_BLENDSHAPES, dtype=lbs_mask.dtype),
                torch.ones(_NUM_HEAD_BLENDSHAPES, dtype=lbs_mask.dtype),
                torch.zeros(_NUM_HAND_BLENDSHAPES, dtype=lbs_mask.dtype),
                torch.ones(self._num_expression_blendshapes, dtype=lbs_mask.dtype),
            ]
        )
        # Level 2: Body rigid transformations only
        lbs_mask = (
            self._mhr_model.character.parameter_transform.rigid_parameters.clone()[
                :_NUM_RIG_PARAMETERS
            ]
        )
        blendshapes_mask = torch.zeros(self._num_blendshapes, dtype=lbs_mask.dtype)
        blendshapes_vertex_mask = torch.zeros(self._num_vertices, dtype=torch.float)
        stages.append(
            self._add_stage(
                lbs_mask,
                blendshapes_mask,
                info="Stage 2: body rigid transform",
                parameter2joints_mapping=parameter2joints_mapping,
                skinning_weight_matrix=skinning_weight_matrix,
                blendshapes_vertex_mask=blendshapes_vertex_mask,
                constant_parameter_mask=constant_face_parameter_mask,
                levmar_lambda=0.001,
            )
        )

        # Level 3: Add torso and limb roots
        for i, name in enumerate(lbs_parameter_names):
            if self._contains_body_part(name):
                lbs_mask[i] = True
        stages.append(
            self._add_stage(
                lbs_mask,
                blendshapes_mask,
                info="Stage 3: torso and limbs' roots",
                parameter2joints_mapping=parameter2joints_mapping,
                skinning_weight_matrix=skinning_weight_matrix,
                blendshapes_vertex_mask=blendshapes_vertex_mask,
                levmar_lambda=0.001,
            )
        )

        # Level 4: Full body excluding hands
        lbs_mask = torch.ones_like(lbs_mask)
        for i, name in enumerate(lbs_parameter_names):
            if self._contains_hand_part(name):
                lbs_mask[i] = False
        blendshapes_mask[:_NUM_BODY_BLENDSHAPES] = True
        stages.append(
            self._add_stage(
                lbs_mask,
                blendshapes_mask,
                info="Stage 4: limbs",
                parameter2joints_mapping=parameter2joints_mapping,
                skinning_weight_matrix=skinning_weight_matrix,
                blendshapes_vertex_mask=blendshapes_vertex_mask,
                levmar_lambda=0.001,
            )
        )

        # Level 5: All parameters
        stages.append(
            self._add_stage(
                torch.ones_like(lbs_mask),
                torch.ones_like(blendshapes_mask),
                vertex_weight=torch.ones_like(vertex_weight),
                parameter2joints_mapping=parameter2joints_mapping,
                skinning_weight_matrix=skinning_weight_matrix,
                constant_parameter_mask=None,
                info="Stage 5: all",
                levmar_lambda=0.001,
            )
        )
        return stages

    def _subsample_mhr_vertices(self, num_subsampled_mhr_vertices: int) -> torch.Tensor:
        """Subsample MHR vertices using clustering for computational efficiency."""
        vertex_positions = self._mhr_model.character.mesh.vertices

        selected_indices = clustering_based_sampling(
            vertex_positions, num_subsampled_mhr_vertices
        )

        mask = torch.zeros(vertex_positions.shape[0], dtype=torch.bool)
        mask[selected_indices] = True
        return mask

    def _get_solver_options(
        self, stage: PyMomentumOptimizationStage
    ) -> pym_solve.SolverOptions:
        """Get optimized solver options for PyMomentum optimization."""
        levmar_lambda = 0.01
        if stage.levmar_lambda is not None:
            levmar_lambda = stage.levmar_lambda
        return pym_solve.SolverOptions(
            linear_solver=pym_solve.LinearSolverType.Cholesky,
            levmar_lambda=levmar_lambda,
            min_iter=2,
            max_iter=5,
            threshold=1.0,
            line_search=False,
        )

    def _optimize_one_stage(
        self, target_vertices: torch.Tensor, stage: PyMomentumOptimizationStage
    ) -> None:
        """Optimize model parameters for a single hierarchical stage."""
        num_vertices = target_vertices.shape[0]
        vertex_cons_vertices = torch.arange(num_vertices)
        vertex_mask = stage.vertex_cons_weights > 0
        vertex_mask = vertex_mask & self._subsampled_vertex_mask

        vertex_cons_vertices = vertex_cons_vertices[vertex_mask]
        target_vertices = target_vertices[vertex_mask]

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

    def _load_head_hand_mask(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load face and hand mask from the PROTO model."""
        data = dict(np.load(HEAD_HAND_MASK_FILE))
        for k, v in data.items():
            data[k] = self._to_tensor(v)
        return data["head_mask"], data["hand_mask"], data["body_mask"]


def clustering_based_sampling(points: torch.Tensor | np.ndarray, p: int) -> list[int]:
    """Subsample points using K-means clustering for computational efficiency."""
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    n = points.shape[0]
    n_clusters = min(int(p * 1.5), n)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
    labels = kmeans.fit_predict(points)

    selected_indices = []

    for cluster_id in range(n_clusters):
        if len(selected_indices) >= p:
            break
        cluster_points = np.where(labels == cluster_id)[0]
        if len(cluster_points) > 0:
            cluster_center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(points[cluster_points] - cluster_center, axis=1)
            best_idx = cluster_points[np.argmin(distances)]
            selected_indices.append(best_idx)

    return selected_indices[:p]
