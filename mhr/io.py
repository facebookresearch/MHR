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
from typing import Dict, Union

import numpy as np
import torch

from .utils import SparseLinear


POSE_CORRECTIVES_SPARSE_MASK_NAME = "posedirs_sparse_mask"
POSE_CORRECTIVES_COMPONENTS_NAME = "corrective_blendshapes"

# Same variable and default as the --dest option of mhr-download-assets.
ASSETS_DEST_ENV = "MHR_ASSETS_DEST"
MODEL_FILE_NAME = "compact_v6_1.model"
# The file that marks a folder of converted assets (see mhr/_rig_schema.py).
RIG_FILE_NAME = "rig.npz"

PathLike = Union[str, os.PathLike]


def _package_asset_folder() -> Path:
    """Return the assets folder next to the mhr package (source checkouts)."""

    return Path(__file__).resolve().parent.parent / "assets"


def _download_asset_folder() -> Path:
    """Return the folder that mhr-download-assets extracts to by default."""

    dest = os.environ.get(ASSETS_DEST_ENV) or "."
    return Path(dest).absolute() / "assets"


def get_asset_folder_candidates() -> list[Path]:
    """Return the folders that get_default_asset_folder() searches, in order.

    1. ``assets`` next to the ``mhr`` package, for a source or editable checkout.
    2. ``$MHR_ASSETS_DEST/assets``, if ``MHR_ASSETS_DEST`` is set.
    3. ``assets`` in the current working directory.

    2 and 3 are where ``mhr-download-assets`` extracts the assets by default.
    """

    candidates = []
    for folder in (_package_asset_folder(), _download_asset_folder(), Path.cwd() / "assets"):
        if folder not in candidates:
            candidates.append(folder)
    return candidates


def get_default_asset_folder(marker: str = MODEL_FILE_NAME) -> Path:
    """Return the default MHR asset folder.

    This is the first folder from get_asset_folder_candidates() that contains
    the file ``marker``: the MHR model file for the FBX assets, or
    RIG_FILE_NAME for the converted assets. If no folder contains it, this is
    the folder that ``mhr-download-assets`` extracts to by default.
    """

    for folder in get_asset_folder_candidates():
        if (folder / marker).is_file():
            return folder
    return _download_asset_folder()


def asset_not_found_message(
    missing: list[str],
    searched: list[Path],
    download_command: str = "mhr-download-assets",
) -> str:
    """Return an error message for missing asset files, with the ways to fix it."""

    lines = ["MHR asset files not found:"]
    lines += [f"  {path}" for path in missing]
    lines.append("Searched these asset folders:")
    lines += [f"  {folder}" for folder in searched]
    lines += [
        "To fix it, do one of the following:",
        f"  - Run {download_command} in your working directory. It extracts the assets to ./assets.",
        f"  - Set {ASSETS_DEST_ENV} to the directory that contains the extracted assets folder.",
        "  - Pass folder=<path to the assets folder> to MHR.from_files.",
    ]
    return "\n".join(lines)


def get_mhr_fbx_path(folder: PathLike, lod: int) -> str:
    """Return the path to the MHR fbx file."""

    asset_path = Path(folder) / f"lod{lod}.fbx"
    return str(asset_path)


def get_mhr_rig_path(folder: PathLike) -> str:
    """Return the path to the converted rig data that all LODs share."""

    return str(Path(folder) / RIG_FILE_NAME)


def get_mhr_lod_path(folder: PathLike, lod: int) -> str:
    """Return the path to the converted data of one LOD."""

    return str(Path(folder) / f"lod{lod}.npz")


def get_mhr_model_path(folder: PathLike) -> str:
    """Return the path to the MHR model definition file (same across LODs)."""

    asset_path = Path(folder) / MODEL_FILE_NAME
    return str(asset_path)


def get_mhr_blendshapes_path(folder: PathLike, lod: int) -> str:
    """Return the path to the file storing pose-dependent blendshapes."""

    asset_path = Path(folder) / f"corrective_blendshapes_lod{lod}.npz"
    return str(asset_path)


def get_corrective_activation_path(folder: PathLike) -> str:
    """Return the path to the file storing activations for the pose-dependent correctives."""

    asset_path = Path(folder) / "corrective_activation.npz"
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
