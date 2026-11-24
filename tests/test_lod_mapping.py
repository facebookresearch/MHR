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

"""Tests for the LOD 1 -> LOD N mappings of tools/mhr_LOD_conversion.

The real-asset test runs when MHR_TEST_ASSETS names the release asset folder.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

_TOOL_DIR = Path(__file__).resolve().parents[1] / "tools" / "mhr_LOD_conversion"
_spec = importlib.util.spec_from_file_location(
    "mhr_lod_compute_mapping", _TOOL_DIR / "compute_mapping.py"
)
compute_mapping_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compute_mapping_module)

ASSETS = os.environ.get("MHR_TEST_ASSETS")

# Every LOD vertex lies within 0.22 cm of the LOD 1 rest surface.
MAX_MAPPING_DISTANCE_CM = 0.3


def _brute_force_closest(point, a, b, c, samples=401):
    u, v = np.meshgrid(np.linspace(0, 1, samples), np.linspace(0, 1, samples))
    keep = u + v <= 1
    u, v = u[keep], v[keep]
    candidates = (1 - u - v)[:, None] * a + u[:, None] * b + v[:, None] * c
    return candidates[np.argmin(np.linalg.norm(candidates - point, axis=1))]


@pytest.mark.parametrize(
    "point",
    [
        [0.2, 0.2, 1.0],  # above the interior
        [-1.0, -1.0, 0.5],  # vertex a region
        [3.0, -0.5, 0.0],  # vertex b region
        [-0.5, 3.0, -0.3],  # vertex c region
        [0.5, -1.0, 0.2],  # edge ab region
        [-1.0, 0.5, 0.2],  # edge ac region
        [1.0, 1.0, -0.4],  # edge bc region
    ],
)
def test_closest_point_matches_brute_force(point):
    a, b, c = (np.array(p, dtype=float) for p in ([0, 0, 0], [2, 0, 0], [0, 2, 0]))
    point = np.array(point, dtype=float)
    closest, bary = compute_mapping_module.closest_points_on_triangles(
        point, a, b, c
    )

    np.testing.assert_allclose(bary.sum(), 1.0, atol=1e-12)
    assert (bary >= -1e-12).all()
    np.testing.assert_allclose(closest, bary[0] * a + bary[1] * b + bary[2] * c)
    np.testing.assert_allclose(
        closest, _brute_force_closest(point, a, b, c), atol=5e-3
    )


def test_mapping_returns_the_closest_triangle():
    # One target lies on the flat triangle, the other next to the raised one.
    source_vertices = np.array(
        [[0, 0, 0], [10, 0, 0], [0, 1, 0], [5, 0.5, 3.0]], dtype=float
    )
    source_faces = np.array([[0, 1, 2], [1, 3, 2]])
    target_vertices = np.array([[1.0, 0.2, 0.0], [5.0, 0.6, 2.9]])

    mapping = compute_mapping_module.compute_mapping(
        source_vertices, source_faces, target_vertices, num_candidate_vertices=4
    )

    mapped = np.einsum(
        "ijk,ij->ik",
        source_vertices[source_faces[mapping["triangle_ids"]]],
        mapping["baryc_coords"],
    )
    np.testing.assert_array_equal(mapping["triangle_ids"], [0, 1])
    np.testing.assert_allclose(mapped[0], target_vertices[0], atol=1e-12)
    assert mapping["triangle_ids"].dtype == np.int64
    assert mapping["baryc_coords"].dtype == np.float64


@pytest.mark.parametrize(
    "degenerate_face",
    [[0, 1, 2], [0, 0, 1]],  # corners on one line; a repeated corner
)
def test_zero_area_source_triangles_do_not_win(degenerate_face):
    source_vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 1], [1, 1, 1], [0, 2, 1]],
        dtype=float,
    )
    source_faces = np.array([degenerate_face, [3, 4, 5]])
    rng = np.random.default_rng(0)
    # Targets on the good triangle (z = 1).
    u, v = rng.uniform(0, 0.5, size=(2, 64))
    target_vertices = np.stack([u, 1 + v, np.ones_like(u)], axis=1)

    mapping = compute_mapping_module.compute_mapping(
        source_vertices, source_faces, target_vertices
    )

    np.testing.assert_array_equal(mapping["triangle_ids"], 1)
    assert np.isfinite(mapping["baryc_coords"]).all()


def test_batches_give_the_same_mapping(monkeypatch):
    rng = np.random.default_rng(0)
    source_vertices = rng.normal(size=(200, 3))
    source_faces = rng.integers(0, 200, size=(300, 3))
    target_vertices = rng.normal(size=(50, 3))
    whole = compute_mapping_module.compute_mapping(
        source_vertices, source_faces, target_vertices
    )

    monkeypatch.setattr(compute_mapping_module, "BATCH_SIZE", 7)
    batched = compute_mapping_module.compute_mapping(
        source_vertices, source_faces, target_vertices
    )

    for key in whole:
        np.testing.assert_array_equal(whole[key], batched[key])


@pytest.mark.skipif(ASSETS is None, reason="set MHR_TEST_ASSETS to the asset folder")
@pytest.mark.parametrize("target_lod", [0, 2, 3, 4, 5, 6])
def test_mapping_files_put_every_vertex_on_the_lod1_surface(target_lod):
    pytest.importorskip("pymomentum.geometry")
    import torch

    from mhr.mhr import MHR

    def rest_mesh(lod):
        model = MHR.from_files(
            folder=ASSETS,
            device=torch.device("cpu"),
            lod=lod,
            wants_pose_correctives=False,
        )
        return np.asarray(model.character.mesh.vertices), np.asarray(
            model.character.mesh.faces
        )

    source_vertices, source_faces = rest_mesh(1)
    target_vertices, _ = rest_mesh(target_lod)
    with np.load(_TOOL_DIR / f"lod1_to_lod{target_lod}_mapping.npz") as mapping:
        triangle_ids = mapping["triangle_ids"]
        baryc_coords = mapping["baryc_coords"]

    assert triangle_ids.shape == (len(target_vertices),)
    mapped = np.einsum(
        "ijk,ij->ik", source_vertices[source_faces[triangle_ids]], baryc_coords
    )
    distance = np.linalg.norm(mapped - target_vertices, axis=1)
    worst = int(np.argmax(distance))
    assert distance.max() < MAX_MAPPING_DISTANCE_CM, (
        f"LOD {target_lod} vertex {worst} at {target_vertices[worst]} maps "
        f"{distance[worst]:.2f} cm away from itself; "
        f"{int((distance >= MAX_MAPPING_DISTANCE_CM).sum())} vertices exceed "
        f"{MAX_MAPPING_DISTANCE_CM} cm"
    )
