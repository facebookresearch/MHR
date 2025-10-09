# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the TODO license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch

from .utils import SparseLinear


FACE_EXPR_COMPONENTS_NAME = "facial_expression_components"
IDENTITY_MEAN_NAME = "identity_mean"
IDENTITY_COMPONENTS_NAME = "identity_components"
POSE_CORRECTIVES_SPARSE_MASK_NAME = "posedirs_sparse_mask"
POSE_CORRECTIVES_STATE_PREFIX = "posedirs_state_dict"


def get_proto_fbx_path(lod: int) -> str:
    """Return the path to the PROTO fbx file."""

    script_dir = Path(__file__).parent
    asset_path = script_dir.parent / "assets" / f"rig_{lod}.fbx"
    return str(asset_path)


def get_proto_model_path() -> str:
    """Return the path to the PROTO model definition file (same across LODs)."""

    script_dir = Path(__file__).parent
    asset_path = script_dir.parent / "assets" / "model_definition.model"
    return str(asset_path)


def get_proto_correctives_path(lod: int) -> str:
    """Return the path to the file storing identity blendshapes, facial expression blendshapes, and pose-dependent correctives."""

    script_dir = Path(__file__).parent
    asset_path = script_dir.parent / "assets" / "correctives_{lod}.npz"
    return str(asset_path)


def load_pose_dirs_predictor(
    data: Dict[str, np.ndarray], load_with_cuda: bool
) -> torch.nn.Sequential:
    """Extract pose correctives data and build the pose correctives predictor."""

    posedirs = torch.nn.Sequential(
        SparseLinear(
            125 * 6,
            125 * 24,
            torch.from_numpy(data[POSE_CORRECTIVES_SPARSE_MASK_NAME]),
            bias=False,
            load_with_cuda=load_with_cuda,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(
            125 * 24, 18439 * 3, bias=False
        ),  # TODO: Hard-coded numbers should be removed, deal with LODs!
    )
    state_dict = {k: torch.tensor(data[k]) for k in data.keys() if k.startswith(POSE_CORRECTIVES_STATE_PREFIX)}
    posedirs.load_state_dict(state_dict)
    for posedir in posedirs.parameters():
        posedir.requires_grad = False

    return posedirs


def has_facial_expressions(data: Dict[str, np.ndarray]) -> bool:
    """Check if the data contains facial expression blendshapes."""

    return FACE_EXPR_COMPONENTS_NAME in data


def has_pose_correctives(data: Dict[str, np.ndarray]) -> bool:
    """Check if the data contains pose-depdendent correctivesß."""

    return POSE_CORRECTIVES_SPARSE_MASK_NAME in data


def load_blendshapes(
    data: Dict[str, np.ndarray], is_identity: bool
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Load and reshape identity/facial expressions blendshapes."""

    if is_identity:
        mean_shape = torch.from_numpy(data[IDENTITY_MEAN_NAME].reshape((-1, 3)))
        n_components = data[IDENTITY_COMPONENTS_NAME].shape[0]
        components = torch.from_numpy(
            data[IDENTITY_COMPONENTS_NAME].reshape((n_components, -1, 3))
        )
        return mean_shape, components
    else:
        n_components = data[FACE_EXPR_COMPONENTS_NAME].shape[0]
        components = torch.from_numpy(
            data[FACE_EXPR_COMPONENTS_NAME].reshape((n_components, -1, 3))
        )
        return components
