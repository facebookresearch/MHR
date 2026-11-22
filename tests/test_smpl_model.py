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

"""Tests for the SMPL model loader of the SMPL-MHR conversion tool."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "mhr_smpl_conversion"
    / "smpl_model.py"
)
_spec = importlib.util.spec_from_file_location("mhr_smpl_model", _MODULE_PATH)
smpl_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smpl_model)

_NUM_SMPL_VERTICES = 6890


def _grid_mesh(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, faces) of a triangulated rows x cols grid."""

    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, rows), np.linspace(0.0, 1.0, cols), indexing="ij"
    )
    grid_positions = np.stack([x.ravel(), y.ravel()], axis=1)
    index = np.arange(rows * cols).reshape(rows, cols)
    a = index[:-1, :-1].ravel()
    b = index[:-1, 1:].ravel()
    c = index[1:, :-1].ravel()
    d = index[1:, 1:].ravel()
    grid_faces = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    # Shuffle the vertex order so that neighboring indices are not neighboring
    # vertices; order detection must then rely on the faces.
    order = np.random.default_rng(2).permutation(rows * cols)
    new_index = np.empty_like(order)
    new_index[order] = np.arange(rows * cols)
    return grid_positions[order], new_index[grid_faces]


def _smooth_basis(positions: np.ndarray, num_components: int) -> np.ndarray:
    """Return a non-zero basis shaped (vertices, 3, components) that varies
    smoothly over the mesh, like real blendshape offsets."""

    rng = np.random.default_rng(0)
    frequency = rng.uniform(0.1, 0.5, size=(2, 3, num_components))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(3, num_components))
    angle = (
        positions[:, 0, None, None] * frequency[0]
        + positions[:, 1, None, None] * frequency[1]
    )
    return 0.01 * np.sin(2.0 * np.pi * angle + phase)


def _flatten(basis: np.ndarray, layout: str) -> np.ndarray:
    """Flatten (vertices, 3, components) to (3 * vertices, components)."""

    if layout == "vertex_major":
        return basis.reshape(-1, basis.shape[2])
    return basis.transpose(1, 0, 2).reshape(-1, basis.shape[2])


@pytest.mark.parametrize("layout", smpl_model.BLENDSHAPE_LAYOUTS)
@pytest.mark.parametrize("components_first", [True, False])
def test_detect_flattened_blendshape_order(
    layout: str, components_first: bool
) -> None:
    positions, faces = _grid_mesh(20, 24)
    basis = _smooth_basis(positions, num_components=6)
    flat = _flatten(basis, layout)
    if components_first:
        flat = flat.T

    reshaped = smpl_model._reshape_blendshape_basis(
        "posedirs", flat, faces, num_vertices=len(positions)
    )

    np.testing.assert_array_equal(reshaped, basis)


def test_ambiguous_blendshape_order_raises() -> None:
    positions, faces = _grid_mesh(20, 24)
    # Random offsets are equally rough in both orders.
    noise = np.random.default_rng(1).normal(size=(len(positions) * 3, 6))

    with pytest.raises(ValueError, match="blendshape_layout"):
        smpl_model._reshape_blendshape_basis(
            "posedirs", noise.T, faces, num_vertices=len(positions)
        )


@pytest.mark.parametrize("layout", smpl_model.BLENDSHAPE_LAYOUTS)
def test_explicit_blendshape_layout_is_used(layout: str) -> None:
    positions, faces = _grid_mesh(20, 24)
    noise = np.random.default_rng(1).normal(size=(len(positions), 3, 6))
    flat = _flatten(noise, layout)

    reshaped = smpl_model._reshape_blendshape_basis(
        "posedirs", flat.T, faces, num_vertices=len(positions), layout=layout
    )

    np.testing.assert_array_equal(reshaped, noise)


def test_explicit_blendshape_layout_overrides_detection() -> None:
    positions, faces = _grid_mesh(20, 24)
    basis = _smooth_basis(positions, num_components=6)
    flat = _flatten(basis, "vertex_major")

    reshaped = smpl_model._reshape_blendshape_basis(
        "posedirs",
        flat,
        faces,
        num_vertices=len(positions),
        layout="coordinate_major",
    )

    np.testing.assert_array_equal(
        reshaped, flat.reshape(3, len(positions), -1).transpose(1, 0, 2)
    )


def _mixed_basis(
    positions: np.ndarray, noise_weight: float, layout: str
) -> np.ndarray:
    """Return a flattened smooth basis with added noise."""

    basis = _smooth_basis(positions, num_components=6)
    noise = np.random.default_rng(3).normal(size=basis.shape)
    basis = basis + noise_weight * np.abs(basis).mean() * noise
    return _flatten(basis, layout).T


def _roughness_ratio(flat: np.ndarray, faces: np.ndarray) -> float:
    num_vertices = flat.shape[1] // 3
    edges = smpl_model._mesh_edges(faces, num_vertices)
    roughness = sorted(
        smpl_model._roughness(candidate, edges)
        for candidate in (
            flat.T.reshape(num_vertices, 3, -1),
            flat.T.reshape(3, num_vertices, -1).transpose(1, 0, 2),
        )
    )
    return roughness[1] / roughness[0]


@pytest.mark.parametrize("layout", smpl_model.BLENDSHAPE_LAYOUTS)
def test_roughness_ratio_threshold(layout: str) -> None:
    positions, faces = _grid_mesh(20, 24)
    weak = _mixed_basis(positions, noise_weight=0.8, layout=layout)
    clear = _mixed_basis(positions, noise_weight=0.5, layout=layout)
    # The weak case is less than 2x smoother in the correct order; the clear
    # case is more than 2x smoother.
    assert 1.5 < _roughness_ratio(weak, faces) < 2.0
    assert 2.0 < _roughness_ratio(clear, faces) < 3.0

    with pytest.raises(ValueError, match="blendshape_layout"):
        smpl_model._reshape_blendshape_basis(
            "posedirs", weak, faces, num_vertices=len(positions)
        )
    reshaped = smpl_model._reshape_blendshape_basis(
        "posedirs", clear, faces, num_vertices=len(positions)
    )
    expected = smpl_model._reshape_blendshape_basis(
        "posedirs", clear, faces, num_vertices=len(positions), layout=layout
    )
    np.testing.assert_array_equal(reshaped, expected)


def test_zero_basis_needs_no_order() -> None:
    positions, faces = _grid_mesh(4, 4)
    zeros = np.zeros((2, len(positions) * 3))

    reshaped = smpl_model._reshape_blendshape_basis(
        "posedirs", zeros, faces, num_vertices=len(positions)
    )

    np.testing.assert_array_equal(reshaped, np.zeros((len(positions), 3, 2)))


def test_nan_basis_raises() -> None:
    positions, faces = _grid_mesh(20, 24)
    flat = _flatten(_smooth_basis(positions, num_components=6), "vertex_major")
    flat[0, 0] = np.nan

    with pytest.raises(ValueError, match="blendshape_layout"):
        smpl_model._reshape_blendshape_basis(
            "posedirs", flat, faces, num_vertices=len(positions)
        )


def test_uint32_faces_are_accepted() -> None:
    positions, faces = _grid_mesh(20, 24)
    basis = _smooth_basis(positions, num_components=6)

    reshaped = smpl_model._reshape_blendshape_basis(
        "posedirs",
        _flatten(basis, "coordinate_major"),
        faces.astype(np.uint32),
        num_vertices=len(positions),
    )

    np.testing.assert_array_equal(reshaped, basis)


def test_faces_must_cover_every_vertex() -> None:
    positions, faces = _grid_mesh(20, 24)
    basis = _smooth_basis(positions, num_components=6)

    with pytest.raises(ValueError, match="every vertex index"):
        smpl_model._reshape_blendshape_basis(
            "posedirs",
            _flatten(basis, "coordinate_major"),
            faces[:1],
            num_vertices=len(positions),
        )


def test_unknown_blendshape_layout_raises() -> None:
    positions, faces = _grid_mesh(4, 4)
    basis = _smooth_basis(positions, num_components=2)

    with pytest.raises(ValueError, match="Unknown blendshape layout"):
        smpl_model._reshape_blendshape_basis(
            "posedirs", basis, faces, num_vertices=len(positions), layout="xyz"
        )


def _smpl_model_data(posedirs: np.ndarray, shapedirs: np.ndarray, faces) -> dict:
    return {
        "v_template": np.zeros((_NUM_SMPL_VERTICES, 3)),
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "J_regressor": np.zeros((24, _NUM_SMPL_VERTICES)),
        "weights": np.zeros((_NUM_SMPL_VERTICES, 24)),
        "kintree_table": np.zeros((2, 24), dtype=np.int64),
        "f": faces,
    }


@pytest.mark.parametrize("layout", smpl_model.BLENDSHAPE_LAYOUTS)
def test_normalize_smpl_model_data(layout: str) -> None:
    # A 106 x 65 grid has the 6890 vertices of the SMPL mesh.
    positions, faces = _grid_mesh(106, 65)
    posedirs = _smooth_basis(positions, num_components=207)
    shapedirs = _smooth_basis(positions, num_components=10)
    model_data = _smpl_model_data(
        _flatten(posedirs, layout).T, _flatten(shapedirs, layout).T, faces
    )

    normalized = smpl_model._normalize_smpl_model_data(model_data)

    np.testing.assert_array_equal(normalized["posedirs"], posedirs)
    np.testing.assert_array_equal(normalized["shapedirs"], shapedirs)


def test_normalize_keeps_3d_blendshapes() -> None:
    positions, faces = _grid_mesh(106, 65)
    posedirs = _smooth_basis(positions, num_components=207)
    shapedirs = _smooth_basis(positions, num_components=10)
    model_data = _smpl_model_data(posedirs, shapedirs, faces)

    normalized = smpl_model._normalize_smpl_model_data(model_data)

    np.testing.assert_array_equal(normalized["posedirs"], posedirs)
    np.testing.assert_array_equal(normalized["shapedirs"], shapedirs)


def test_blendshape_layout_rejected_for_pkl(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"")

    with pytest.raises(ValueError, match="only to .npz files"):
        smpl_model.load_smpl_model(model_path, blendshape_layout="vertex_major")


def test_reject_nonstandard_smpl_topology() -> None:
    model_data = {
        key: np.empty(0)
        for key in (
            "J_regressor",
            "f",
            "kintree_table",
            "posedirs",
            "shapedirs",
            "weights",
        )
    }
    model_data["v_template"] = np.zeros((10, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="6890 vertices"):
        smpl_model._normalize_smpl_model_data(model_data)
