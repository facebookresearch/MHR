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

from ._pymomentum_version import check_pymomentum_version

check_pymomentum_version()

import pymomentum.geometry as pym_geometry

import pymomentum.torch.character as torch_character

import torch

from . import _model
from ._model import (  # noqa: F401  (re-exported for existing imports)
    LOD,
    NUM_FACE_EXPRESSION_BLENDSHAPES,
    NUM_IDENTITY_BLENDSHAPES,
    MHRPoseCorrectivesModel,
)
from .io import (
    asset_not_found_message,
    get_asset_folder_candidates,
    get_corrective_activation_path,
    get_default_asset_folder,
    get_mhr_blendshapes_path,
    get_mhr_fbx_path,
    get_mhr_model_path,
    has_pose_corrective_blendshapes,
    load_pose_dirs_predictor,
)


class MHR(_model.MHR):
    """MHR body model, loaded from the FBX assets with PyMomentum.

    `from mhr import MHR` gives the same model without PyMomentum, from the
    converted assets.
    """

    def __init__(
        self,
        character: pym_geometry.Character,
        pose_correctives_model: MHRPoseCorrectivesModel | None,
        device: torch.device,
    ) -> None:
        # Note that this call also instantiates the identity and face expressions model
        super().__init__(
            torch_character.Character(character).to(device), pose_correctives_model
        )
        # The PyMomentum character, for tools that use PyMomentum directly
        self.character = character

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
        folder: str | os.PathLike | None = None,
        device: torch.device = "cuda",
        lod: LOD = 1,
        wants_pose_correctives: bool = True,
    ) -> "MHR":
        """Load character and model parameterization, and create full model.

        ``folder`` is the folder with the MHR asset files. If it is None,
        get_default_asset_folder() picks it when this method is called.
        Raises FileNotFoundError if an asset file is missing.
        """

        if folder is None:
            searched = get_asset_folder_candidates()
            folder = get_default_asset_folder()
        else:
            folder = Path(folder)
            searched = [folder]

        # Create character by fetching rig and model parameterization paths
        fbx_path = get_mhr_fbx_path(folder, lod)
        model_path = get_mhr_model_path(folder)
        blendshapes_path = get_mhr_blendshapes_path(folder, lod)
        corrective_activation_path = (
            get_corrective_activation_path(folder) if wants_pose_correctives else None
        )
        required_paths = [fbx_path, model_path, blendshapes_path]
        if corrective_activation_path is not None:
            required_paths.append(corrective_activation_path)
        missing = [path for path in required_paths if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(asset_not_found_message(missing, searched))

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

        # Create full model with the correctives
        return MHR._create_model(
            character, blendshapes_path, corrective_activation_path, device
        )


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
