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

"""Load SMPL model files for the conversion tool without requiring Chumpy.

``smplx.SMPL`` reads pickle files only, and it assumes ``posedirs`` and
``shapedirs`` are stored as ``(vertices, 3, components)``. Some SMPL ``.npz``
files store these bases as flattened 2D arrays instead. A flattened array has
two possible element orders, and the shape alone does not tell them apart:

* ``"vertex_major"``: ``x0, y0, z0, x1, y1, z1, ...``
* ``"coordinate_major"``: ``x0, x1, ..., y0, y1, ..., z0, z1, ...``

Reading a basis in the wrong order still produces a model, but with wrong
vertices. This module therefore detects the order from the data: blendshape
offsets vary smoothly over the mesh, so the correct order gives a much smaller
difference between the offsets of vertices that share an edge. When the two
orders are not clearly separated, loading fails and the caller must pass
``blendshape_layout``.
"""

from pathlib import Path
from typing import Any

import numpy as np


BLENDSHAPE_LAYOUTS = ("vertex_major", "coordinate_major")

# The correct order must be at least this many times smoother than the other.
MIN_LAYOUT_ROUGHNESS_RATIO = 2.0

_NUM_SMPL_VERTICES = 6890
_NUM_SMPLX_VERTICES = 10475
_SMPLX_HAND_KEYS = ("hands_componentsl", "hands_componentsr")
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


def _mesh_edges(faces: np.ndarray, num_vertices: int) -> np.ndarray:
    """Return the unique undirected edges of a triangle mesh."""

    faces = np.asarray(faces)
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or faces.shape[0] == 0
        or not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0
        or faces.max() >= num_vertices
        or np.unique(faces).size != num_vertices
    ):
        raise ValueError(
            f"SMPL faces must be an integer array shaped (faces, 3) that uses "
            f"every vertex index in [0, {num_vertices}); got shape "
            f"{faces.shape} and dtype {faces.dtype}."
        )
    faces = faces.astype(np.int64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def _roughness(basis: np.ndarray, edges: np.ndarray) -> float:
    """Mean offset change along mesh edges, relative to the mean offset size.

    ``basis`` is shaped ``(vertices, 3, components)``.
    """

    edge_change = 0.0
    magnitude = 0.0
    # Process components in chunks to bound memory for large bases.
    for start in range(0, basis.shape[2], 32):
        chunk = basis[:, :, start : start + 32].astype(np.float64)
        edge_change += np.linalg.norm(
            chunk[edges[:, 0]] - chunk[edges[:, 1]], axis=1
        ).mean(axis=0).sum()
        magnitude += np.linalg.norm(chunk, axis=1).mean(axis=0).sum()
    if magnitude == 0.0:
        return float("nan")
    return float(edge_change / magnitude)


def _reshape_blendshape_basis(
    name: str,
    basis: np.ndarray,
    faces: np.ndarray,
    num_vertices: int,
    layout: str | None = None,
) -> np.ndarray:
    """Return a blendshape basis shaped ``(vertices, 3, components)``.

    A 3D basis must already have that shape. A 2D basis may be shaped
    ``(3 * vertices, components)`` or ``(components, 3 * vertices)``. Its
    element order is ``layout`` when given; otherwise it is detected from the
    mesh, and a ``ValueError`` is raised when the detection is not conclusive.
    """

    if layout is not None and layout not in BLENDSHAPE_LAYOUTS:
        raise ValueError(
            f"Unknown blendshape layout {layout!r}; expected one of "
            f"{', '.join(BLENDSHAPE_LAYOUTS)}."
        )

    basis = np.asarray(basis)
    if basis.ndim == 3 and basis.shape[:2] == (num_vertices, 3):
        return basis

    vertex_dimensions = num_vertices * 3
    flat = None
    if basis.ndim == 2:
        if basis.shape[0] == vertex_dimensions:
            flat = basis
        elif basis.shape[1] == vertex_dimensions:
            flat = basis.T
    if flat is None:
        raise ValueError(
            f"Unsupported SMPL {name} shape {basis.shape}. Expected "
            f"({num_vertices}, 3, components), ({vertex_dimensions}, components), "
            f"or (components, {vertex_dimensions})."
        )

    candidates = {
        "vertex_major": flat.reshape(num_vertices, 3, -1),
        "coordinate_major": flat.reshape(3, num_vertices, -1).transpose(1, 0, 2),
    }
    if layout is not None:
        return candidates[layout]
    if np.array_equal(candidates["vertex_major"], candidates["coordinate_major"]):
        # Both orders give the same basis, so the order does not matter.
        return candidates["vertex_major"]

    edges = _mesh_edges(faces, num_vertices)
    roughness = {key: _roughness(value, edges) for key, value in candidates.items()}
    smooth, rough = sorted(roughness, key=lambda key: roughness[key])
    ratio = roughness[rough] / roughness[smooth] if roughness[smooth] > 0 else np.inf
    if not np.isfinite(roughness[smooth]) or not ratio >= MIN_LAYOUT_ROUGHNESS_RATIO:
        raise ValueError(
            f"Cannot determine the element order of the flattened SMPL {name} "
            f"array shaped {basis.shape}: mesh-edge roughness is "
            f"{roughness['vertex_major']:.3g} for vertex_major and "
            f"{roughness['coordinate_major']:.3g} for coordinate_major (the "
            f"correct order must be at least {MIN_LAYOUT_ROUGHNESS_RATIO:g}x "
            "smoother). Pass blendshape_layout='vertex_major' (x0, y0, z0, "
            "x1, ...) or blendshape_layout='coordinate_major' (x0, x1, ..., "
            "y0, ...) to state the order used by the file."
        )
    return candidates[smooth]


def _normalize_smpl_model_data(
    model_data: dict[str, np.ndarray],
    blendshape_layout: str | None = None,
) -> dict[str, np.ndarray]:
    """Validate and normalize data loaded from an SMPL ``.npz`` file."""

    missing_keys = sorted(_REQUIRED_SMPL_KEYS - model_data.keys())
    if missing_keys:
        raise ValueError(
            "SMPL NPZ file is missing required keys: " + ", ".join(missing_keys)
        )

    normalized = dict(model_data)
    v_template = np.asarray(normalized["v_template"])
    if v_template.shape != (_NUM_SMPL_VERTICES, 3):
        raise ValueError(
            "Only SMPL models with 6890 vertices are supported; "
            f"v_template has shape {v_template.shape}."
        )
    normalized["v_template"] = v_template

    for name in ("shapedirs", "posedirs"):
        normalized[name] = _reshape_blendshape_basis(
            name,
            normalized[name],
            normalized["f"],
            _NUM_SMPL_VERTICES,
            blendshape_layout,
        )

    if normalized["shapedirs"].shape[2] < 10:
        raise ValueError(
            "SMPL shapedirs must contain at least 10 shape components; "
            f"found {normalized['shapedirs'].shape[2]}."
        )
    if normalized["posedirs"].shape[2] != _NUM_SMPL_POSE_BLENDSHAPES:
        raise ValueError(
            "SMPL posedirs must contain 207 pose blendshapes; "
            f"found {normalized['posedirs'].shape[2]}."
        )

    return normalized


def load_smpl_model(
    model_path: str | Path,
    *,
    blendshape_layout: str | None = None,
    **kwargs: Any,
) -> Any:
    """Load an SMPL ``.npz`` file or a Chumpy-free ``.pkl`` file.

    An ``.npz`` file is loaded in memory, validated, and its ``posedirs`` and
    ``shapedirs`` are converted to ``(6890, 3, components)`` before an
    ``smplx.SMPL`` model is built. ``blendshape_layout`` states the element
    order of flattened 2D bases (``"vertex_major"`` or
    ``"coordinate_major"``); leave it as ``None`` to detect the order, which
    raises a ``ValueError`` when the data do not decide it. A ``.pkl`` file is
    passed to ``smplx.SMPL`` unchanged and does not accept
    ``blendshape_layout``. ``kwargs`` are forwarded to
    ``smplx.SMPL``.
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

    if suffix == ".pkl" and blendshape_layout is not None:
        raise ValueError(
            "blendshape_layout applies only to .npz files; .pkl files are "
            "passed to smplx.SMPL unchanged."
        )

    from smplx import SMPL
    from smplx.utils import Struct

    if suffix == ".npz":
        with np.load(model_path) as model_archive:
            model_data = {key: model_archive[key] for key in model_archive.files}
        model_data = _normalize_smpl_model_data(model_data, blendshape_layout)
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
            "This SMPL .pkl file contains Chumpy objects, which cannot be "
            "loaded in the supported Python environments. Use an SMPL .npz "
            "model file instead."
        ) from error


def load_smplx_model(model_path: str | Path, **kwargs: Any) -> Any:
    """Load an SMPL-X ``.npz`` or ``.pkl`` model file with ``smplx.SMPLX``.

    The file is checked first: an SMPL file passed by mistake raises a
    ``ValueError`` that names the problem, instead of an error from inside
    ``smplx`` such as ``'Struct' object has no attribute 'hands_componentsl'``.
    The file extension selects how ``smplx`` reads the file. ``kwargs`` are
    forwarded to ``smplx.SMPLX``.
    """

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL-X model file not found: {model_path}")

    suffix = model_path.suffix.lower()
    if suffix == ".npz":
        with np.load(model_path, allow_pickle=True) as model_archive:
            keys = set(model_archive.files)
            v_template_shape = (
                model_archive["v_template"].shape if "v_template" in keys else None
            )
    elif suffix == ".pkl":
        import pickle

        try:
            with open(model_path, "rb") as model_file:
                model_data = pickle.load(model_file, encoding="latin1")
        except ModuleNotFoundError as error:
            if error.name is None or error.name.split(".")[0] != "chumpy":
                raise
            raise RuntimeError(
                "This SMPL-X .pkl file contains Chumpy objects, which cannot be "
                "loaded in the supported Python environments. Use an SMPL-X "
                ".npz model file instead."
            ) from error
        keys = set(model_data)
        v_template_shape = (
            np.shape(model_data["v_template"]) if "v_template" in keys else None
        )
    else:
        raise ValueError(
            f"Unsupported SMPL-X model extension '{model_path.suffix}'. "
            "Expected a .npz or .pkl file."
        )

    missing_keys = [key for key in _SMPLX_HAND_KEYS if key not in keys]
    if missing_keys or v_template_shape != (_NUM_SMPLX_VERTICES, 3):
        raise ValueError(
            f"{model_path} is not an SMPL-X model file: its v_template has shape "
            f"{v_template_shape} instead of ({_NUM_SMPLX_VERTICES}, 3)"
            + (f" and it has no {', '.join(missing_keys)}" if missing_keys else "")
            + ". Use an SMPL-X model file such as SMPLX_NEUTRAL.npz; SMPL files "
            "load with load_smpl_model."
        )

    from smplx import SMPLX

    return SMPLX(model_path=str(model_path), ext=suffix[1:], **kwargs)
