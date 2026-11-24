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

"""Compute the LOD 1 -> LOD N barycentric mappings used by example.py.

Each target vertex is mapped to the closest point on the source rest mesh:
the triangle that holds that point and its barycentric coordinates.

    python compute_mapping.py --target-lod 5 --target-lod 6
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# Target vertices processed at once; keeps peak memory under 2 GB for LOD 0.
BATCH_SIZE = 4096

# Source-mesh vertices whose incident triangles are searched for each target
# vertex. The closest triangle always touches one of the nearest vertices
# unless the source mesh has triangles much larger than the target spacing.
NUM_CANDIDATE_VERTICES = 64


def closest_points_on_triangles(
    points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the closest point on each triangle (a, b, c) to each point.

    All inputs have shape (..., 3). Returns the closest points and their
    barycentric coordinates (..., 3), following Ericson, Real-Time Collision
    Detection, section 5.1.5.
    """
    ab, ac, ap = b - a, c - a, points - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = points - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = points - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    with np.errstate(divide="ignore", invalid="ignore"):
        # Interior of the triangle.
        denom = va + vb + vc
        v = vb / denom
        w = vc / denom
        bary = np.stack([1 - v - w, v, w], axis=-1)

        # Edges.
        t_ab = d1 / (d1 - d3)
        t_ac = d2 / (d2 - d6)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))

    zeros = np.zeros_like(d1)
    ones = np.ones_like(d1)
    cases = [
        # (condition, barycentric coordinates); later cases take precedence.
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), (zeros, 1 - t_bc, t_bc)),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), (1 - t_ac, zeros, t_ac)),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), (1 - t_ab, t_ab, zeros)),
        ((d6 >= 0) & (d5 <= d6), (zeros, zeros, ones)),
        ((d3 >= 0) & (d4 <= d3), (zeros, ones, zeros)),
        ((d1 <= 0) & (d2 <= 0), (ones, zeros, zeros)),
    ]
    for condition, coords in cases:
        bary = np.where(condition[..., None], np.stack(coords, axis=-1), bary)

    closest = bary[..., :1] * a + bary[..., 1:2] * b + bary[..., 2:] * c
    return closest, bary


def compute_mapping(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    num_candidate_vertices: int = NUM_CANDIDATE_VERTICES,
) -> dict[str, np.ndarray]:
    """Map each target vertex to the closest point on the source mesh."""
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    target_vertices = np.asarray(target_vertices, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)

    # Triangles incident to each source vertex, padded with -1.
    incident = [[] for _ in range(len(source_vertices))]
    for face_id, face in enumerate(source_faces):
        for vertex_id in face:
            incident[vertex_id].append(face_id)
    width = max(len(faces) for faces in incident)
    vertex_faces = np.full((len(source_vertices), width), -1, dtype=np.int64)
    for vertex_id, faces in enumerate(incident):
        vertex_faces[vertex_id, : len(faces)] = faces

    k = min(num_candidate_vertices, len(source_vertices))
    tree = cKDTree(source_vertices)
    triangle_ids = np.empty(len(target_vertices), dtype=np.int64)
    baryc_coords = np.empty((len(target_vertices), 3), dtype=np.float64)
    for start in range(0, len(target_vertices), BATCH_SIZE):
        targets = target_vertices[start : start + BATCH_SIZE]
        _, nearest = tree.query(targets, k=k)
        nearest = nearest.reshape(len(targets), k)
        candidates = vertex_faces[nearest].reshape(len(targets), -1)
        valid = candidates >= 0
        safe = np.where(valid, candidates, 0)

        triangles = source_vertices[source_faces[safe]]  # (T, C, 3, 3)
        closest, bary = closest_points_on_triangles(
            targets[:, None, :],
            triangles[..., 0, :],
            triangles[..., 1, :],
            triangles[..., 2, :],
        )
        distances = np.linalg.norm(closest - targets[:, None, :], axis=-1)
        # Padding and zero-area triangles (NaN distances) never win.
        distances = np.where(valid & np.isfinite(distances), distances, np.inf)
        best = np.argmin(distances, axis=1)
        rows = np.arange(len(targets))
        triangle_ids[start : start + len(targets)] = safe[rows, best]
        baryc_coords[start : start + len(targets)] = bary[rows, best]
    return {"triangle_ids": triangle_ids, "baryc_coords": baryc_coords}


def main() -> None:
    import torch

    from mhr.mhr import MHR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lod", type=int, default=1)
    parser.add_argument("--target-lod", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    def rest_mesh(lod: int) -> tuple[np.ndarray, np.ndarray]:
        model = MHR.from_files(
            lod=lod, device=torch.device("cpu"), wants_pose_correctives=False
        )
        mesh = model.character.mesh
        return np.asarray(mesh.vertices), np.asarray(mesh.faces)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_vertices, source_faces = rest_mesh(args.source_lod)
    for target_lod in args.target_lod:
        target_vertices, _ = rest_mesh(target_lod)
        mapping = compute_mapping(source_vertices, source_faces, target_vertices)
        mapped = np.einsum(
            "ijk,ij->ik",
            source_vertices[source_faces[mapping["triangle_ids"]]],
            mapping["baryc_coords"],
        )
        error = np.linalg.norm(mapped - target_vertices, axis=1)
        path = args.output_dir / f"lod{args.source_lod}_to_lod{target_lod}_mapping.npz"
        np.savez(path, **mapping)
        print(
            f"{path.name}: {len(target_vertices)} vertices, distance to the "
            f"LOD {args.source_lod} surface mean {error.mean():.3f} cm, "
            f"max {error.max():.3f} cm"
        )


if __name__ == "__main__":
    main()
