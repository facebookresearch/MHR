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

"""Layout of the converted MHR rig assets (rig.npz and lod<N>.npz).

rig.npz holds the data that every LOD shares: the skeleton, the parameter
transform, the parameter limits, the inverse bind pose and the pose
corrective activations. lod<N>.npz holds one LOD: the mesh, the blend shapes,
the skinning weights and the pose corrective blend shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

RIG_SCHEMA_VERSION = 1
PARAMETERS_PER_JOINT = 7
LODS = tuple(range(7))

RIG_ARRAYS = (
    "schema_version",
    "parameter_transform",
    "pose_parameters",
    "rigid_parameters",
    "scaling_parameters",
    "active_joints",
    "parameter_names",
    "joint_offsets",
    "joint_prerotations",
    "joint_parents",
    "joint_names",
    "fk_indices",
    "fk_sizes",
    "inverse_bind_pose",
    "limit_min",
    "limit_max",
    "limit_weight",
    "limit_parameter_index",
)

LOD_ARRAYS = (
    "schema_version",
    "lod",
    "base_shape",
    "shape_vectors",
    "faces",
    "texcoords",
    "texcoord_faces",
    "skin_indices",
    "skin_weights",
)


def _require(data: Mapping[str, Any], path: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")


def validate_rig_assets(
    rig: Mapping[str, np.ndarray],
    lod: Mapping[str, np.ndarray],
    rig_path: Path,
    lod_path: Path,
    expected_lod: int,
) -> None:
    """Raise ValueError if rig.npz and lod<N>.npz do not match the schema."""

    _require(rig, rig_path, RIG_ARRAYS)
    _require(lod, lod_path, LOD_ARRAYS)
    for path, data in ((rig_path, rig), (lod_path, lod)):
        if int(data["schema_version"].item()) != RIG_SCHEMA_VERSION:
            raise ValueError(f"unsupported MHR asset schema in {path}")
    if int(lod["lod"].item()) != expected_lod:
        raise ValueError(f"{lod_path} contains data for the wrong LOD")

    joint_count = rig["joint_offsets"].shape[0]
    vertex_count = lod["base_shape"].shape[0]
    if rig["joint_offsets"].shape != (joint_count, 3):
        raise ValueError(f"invalid joint offset shape in {rig_path}")
    if rig["parameter_transform"].shape[0] != joint_count * PARAMETERS_PER_JOINT:
        raise ValueError(f"invalid parameter transform shape in {rig_path}")
    if rig["joint_prerotations"].shape != (joint_count, 4):
        raise ValueError(f"invalid joint prerotation shape in {rig_path}")
    if rig["joint_parents"].shape != (joint_count,):
        raise ValueError(f"invalid joint parent shape in {rig_path}")
    if rig["inverse_bind_pose"].shape != (joint_count, 8):
        raise ValueError(f"invalid inverse bind pose shape in {rig_path}")
    if int(rig["fk_sizes"].sum()) != rig["fk_indices"].shape[1]:
        raise ValueError(f"invalid FK schedule in {rig_path}")
    if lod["base_shape"].ndim != 2 or lod["base_shape"].shape[1] != 3:
        raise ValueError(f"invalid base shape dimensions in {lod_path}")
    if lod["shape_vectors"].shape[1:] != (vertex_count, 3):
        raise ValueError(f"invalid blend shape dimensions in {lod_path}")
    if lod["faces"].ndim != 2 or lod["faces"].shape[1] != 3:
        raise ValueError(f"invalid face dimensions in {lod_path}")
    if lod["skin_indices"].shape != lod["skin_weights"].shape:
        raise ValueError(f"skinning index/weight shape mismatch in {lod_path}")
    if lod["skin_indices"].shape[0] != vertex_count:
        raise ValueError(f"skinning vertex count mismatch in {lod_path}")
    if np.any(lod["skin_indices"] < 0) or np.any(lod["skin_indices"] >= joint_count):
        raise ValueError(f"skinning joint index out of range in {lod_path}")
