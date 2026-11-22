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

"""A two-joint, two-vertex rig in the converted asset format, for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mhr._rig_schema import RIG_SCHEMA_VERSION

NUM_PARAMETERS = 204 + 45 + 72


def rig_arrays(schema_version: int = RIG_SCHEMA_VERSION) -> dict[str, np.ndarray]:
    parameter_transform = np.zeros((14, NUM_PARAMETERS), dtype=np.float32)
    parameter_transform[0, 0] = 1  # parameter 0 translates the root along x
    return {
        "schema_version": np.array(schema_version, dtype=np.int64),
        "parameter_transform": parameter_transform,
        "pose_parameters": np.zeros(NUM_PARAMETERS, dtype=np.bool_),
        "rigid_parameters": np.zeros(NUM_PARAMETERS, dtype=np.bool_),
        "scaling_parameters": np.zeros(NUM_PARAMETERS, dtype=np.bool_),
        "active_joints": np.ones(2, dtype=np.bool_),
        "parameter_names": np.asarray(
            [f"parameter_{i}" for i in range(NUM_PARAMETERS)]
        ),
        "joint_offsets": np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        "joint_prerotations": np.asarray(
            [[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32
        ),
        "joint_parents": np.asarray([-1, 0], dtype=np.int32),
        "joint_names": np.asarray(["root", "child"]),
        "fk_indices": np.asarray([[1], [0]], dtype=np.int64),
        "fk_sizes": np.asarray([1], dtype=np.int64),
        "inverse_bind_pose": np.asarray(
            [[0, 0, 0, 0, 0, 0, 1, 1], [-1, 0, 0, 0, 0, 0, 1, 1]], dtype=np.float32
        ),
        "limit_min": np.zeros(0, dtype=np.float32),
        "limit_max": np.zeros(0, dtype=np.float32),
        "limit_weight": np.zeros(0, dtype=np.float32),
        "limit_parameter_index": np.zeros(0, dtype=np.int32),
    }


def lod_arrays(
    lod: int = 1, schema_version: int = RIG_SCHEMA_VERSION
) -> dict[str, np.ndarray]:
    skin_indices = np.zeros((2, 8), dtype=np.int32)
    skin_indices[1, 0] = 1  # vertex 1 follows the child joint
    skin_weights = np.zeros((2, 8), dtype=np.float32)
    skin_weights[:, 0] = 1
    shape_vectors = np.zeros((117, 2, 3), dtype=np.float32)
    shape_vectors[0, :, 1] = [1, 2]  # first identity shape moves along y
    shape_vectors[45, :, 2] = [1, 2]  # first expression shape moves along z
    return {
        "schema_version": np.array(schema_version, dtype=np.int64),
        "lod": np.array(lod, dtype=np.int64),
        "base_shape": np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.float32),
        "shape_vectors": shape_vectors,
        "faces": np.asarray([[0, 1, 1]], dtype=np.int32),
        "texcoords": np.zeros((2, 2), dtype=np.float32),
        "texcoord_faces": np.asarray([[0, 1, 1]], dtype=np.int32),
        "skin_indices": skin_indices,
        "skin_weights": skin_weights,
    }


def write_assets(folder: Path, lod: int = 1) -> None:
    # Uncompressed, like the files that scripts/convert_assets.py writes.
    np.savez(folder / "rig.npz", **rig_arrays())
    np.savez(folder / f"lod{lod}.npz", **lod_arrays(lod))
