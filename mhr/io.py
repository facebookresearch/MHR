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


from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .utils import SparseLinear


POSE_CORRECTIVES_SPARSE_MASK_NAME = "posedirs_sparse_mask"
POSE_CORRECTIVES_COMPONENTS_NAME = "corrective_blendshapes"


def get_default_asset_folder() -> Path:
    """Return the path to the default MHR asset folder."""

    return Path(__file__).parent.parent / "assets"


def get_mhr_fbx_path(folder: Path, lod: int) -> str:
    """Return the path to the MHR fbx file."""

    asset_path = folder / f"lod{lod}.fbx"
    return str(asset_path)


def get_mhr_model_path(folder: Path) -> str:
    """Return the path to the MHR model definition file (same across LODs)."""

    asset_path = folder / "compact_v6_1.model"
    return str(asset_path)


def get_mhr_blendshapes_path(folder: Path, lod: int) -> str:
    """Return the path to the file storing pose-dependent blendshapes."""

    asset_path = folder / f"corrective_blendshapes_lod{lod}.npz"
    return str(asset_path)


def get_corrective_activation_path(folder: Path) -> str:
    """Return the path to the file storing activations for the pose-dependent correctives."""

    asset_path = folder / "corrective_activation.npz"
    return str(asset_path)


def load_pose_dirs_predictor(
    blendshapes_data: Dict[str, np.ndarray],
    corrective_activation_data: Dict[str, np.ndarray],
    load_with_cuda: bool,
) -> torch.nn.Sequential:
    """Extract pose correctives data and build the pose correctives predictor."""

    n_components = blendshapes_data[POSE_CORRECTIVES_COMPONENTS_NAME].shape[0]
    n_verts = blendshapes_data[POSE_CORRECTIVES_COMPONENTS_NAME].shape[1]
    state_dict = {
        "0.sparse_indices": torch.from_numpy(
            corrective_activation_data["0.sparse_indices"]
        ),
        "0.sparse_weight": torch.from_numpy(
            corrective_activation_data["0.sparse_weight"]
        ),
    }
    state_dict["2.weight"] = torch.from_numpy(
        blendshapes_data[POSE_CORRECTIVES_COMPONENTS_NAME].reshape((n_components, -1)).T
    )

    posedirs = torch.nn.Sequential(
        SparseLinear(
            125
            * 6,  # num joints minus the 2 global ones (125) x 6D rotation representation
            125 * 24,  # 24 is a hyperparameter
            torch.from_numpy(corrective_activation_data["posedirs_sparse_mask"]),
            bias=False,
            load_with_cuda=load_with_cuda,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(125 * 24, n_verts * 3, bias=False),
    )

    posedirs.load_state_dict(state_dict)
    for posedir in posedirs.parameters():
        posedir.requires_grad = False

    return posedirs


def has_pose_corrective_blendshapes(data: Dict[str, np.ndarray]) -> bool:
    """Check if the data contains pose-dependent correctives."""

    return POSE_CORRECTIVES_COMPONENTS_NAME in data
