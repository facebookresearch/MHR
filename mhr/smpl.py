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

"""Load standard SMPL models without requiring Chumpy."""

from pathlib import Path
from typing import Any

import numpy as np


_NUM_SMPL_VERTICES = 6890
_NUM_SMPL_POSE_BLENDSHAPES = 207
_REQUIRED_SMPL_KEYS = {
    "J_regressor",
    "f",
    "kintree_table",
    "posedirs",
    "shapedirs",
    "v_template",
    "weights",
}


def _reshape_blendshape_basis(
    name: str, basis: np.ndarray, num_vertices: int
) -> np.ndarray:
    """Normalize a blendshape basis to [vertices, 3, components]."""

    basis = np.asarray(basis)
    if basis.ndim == 3 and basis.shape[:2] == (num_vertices, 3):
        return basis

    vertex_dimensions = num_vertices * 3
    if basis.ndim == 2:
        if basis.shape[0] == vertex_dimensions:
            return basis.reshape(num_vertices, 3, -1)
        if basis.shape[1] == vertex_dimensions:
            return basis.T.reshape(num_vertices, 3, -1)

    raise ValueError(
        f"Unsupported SMPL {name} shape {basis.shape}. Expected "
        f"({num_vertices}, 3, components), ({vertex_dimensions}, components), "
        f"or (components, {vertex_dimensions})."
    )


def _normalize_smpl_model_data(
    model_data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate and normalize data loaded from a standard SMPL NPZ file."""

    missing_keys = sorted(_REQUIRED_SMPL_KEYS - model_data.keys())
    if missing_keys:
        raise ValueError(
            "SMPL NPZ file is missing required keys: " + ", ".join(missing_keys)
        )

    normalized = dict(model_data)
    v_template = np.asarray(normalized["v_template"])
    if v_template.shape != (_NUM_SMPL_VERTICES, 3):
        raise ValueError(
            "Only standard SMPL models with 6890 vertices are supported; "
            f"v_template has shape {v_template.shape}."
        )
    normalized["v_template"] = v_template

    normalized["shapedirs"] = _reshape_blendshape_basis(
        "shapedirs", normalized["shapedirs"], _NUM_SMPL_VERTICES
    )
    if normalized["shapedirs"].shape[2] < 10:
        raise ValueError(
            "SMPL shapedirs must contain at least 10 shape components; "
            f"found {normalized['shapedirs'].shape[2]}."
        )

    normalized["posedirs"] = _reshape_blendshape_basis(
        "posedirs", normalized["posedirs"], _NUM_SMPL_VERTICES
    )
    if normalized["posedirs"].shape[2] != _NUM_SMPL_POSE_BLENDSHAPES:
        raise ValueError(
            "SMPL posedirs must contain 207 pose blendshapes; "
            f"found {normalized['posedirs'].shape[2]}."
        )

    return normalized


def load_smpl_model(model_path: str | Path, **kwargs: Any) -> Any:
    """Load a Chumpy-free SMPL PKL or a standard SMPL NPZ model.

    The ``smplx.SMPL`` class reads PKL files directly but does not read SMPL
    NPZ files. For NPZ input, this function loads the arrays in memory and
    normalizes common flattened blendshape layouts before constructing the
    model.
    """

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL model file not found: {model_path}")

    suffix = model_path.suffix.lower()
    if suffix not in {".npz", ".pkl"}:
        raise ValueError(
            f"Unsupported SMPL model extension '{model_path.suffix}'. "
            "Expected a .npz or .pkl file."
        )

    try:
        from smplx import SMPL
        from smplx.utils import Struct
    except ModuleNotFoundError as error:
        if error.name != "smplx":
            raise
        raise RuntimeError(
            "SMPL loading requires the optional smplx package. Install the "
            "tested version with `pixi add --pypi smplx==0.1.28`."
        ) from error

    if suffix == ".npz":
        with np.load(model_path) as model_archive:
            model_data = {key: model_archive[key] for key in model_archive.files}
        model_data = _normalize_smpl_model_data(model_data)
        return SMPL(
            model_path=str(model_path),
            data_struct=Struct(**model_data),
            **kwargs,
        )

    try:
        return SMPL(model_path=str(model_path), **kwargs)
    except ModuleNotFoundError as error:
        if error.name is None or error.name.split(".")[0] != "chumpy":
            raise
        raise RuntimeError(
            "This legacy SMPL PKL contains Chumpy objects. Use the official "
            "SMPL NPZ model instead; MHR loads it directly without Chumpy."
        ) from error


__all__ = ["load_smpl_model"]
