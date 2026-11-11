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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


RIG_SCHEMA_VERSION = 1
PARAMETERS_PER_JOINT = 7


def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _rotate_vector(q: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(q, dim=-1)
    axis = q[..., :3]
    scalar = q[..., 3:]
    axis_cross_value = torch.cross(axis, value, dim=-1)
    return value + 2 * (
        scalar * axis_cross_value + torch.cross(axis, axis_cross_value, dim=-1)
    )


def _euler_xyz_to_quaternion(euler_xyz: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = euler_xyz.unbind(-1)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    return torch.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dim=-1,
    )


def _state_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    while first.ndim < second.ndim:
        first = first.unsqueeze(0)
    while second.ndim < first.ndim:
        second = second.unsqueeze(0)
    first_t, first_q, first_s = first[..., :3], first[..., 3:7], first[..., 7:]
    second_t, second_q, second_s = (
        second[..., :3],
        second[..., 3:7],
        second[..., 7:],
    )
    first_q = torch.nn.functional.normalize(first_q, dim=-1)
    second_q = torch.nn.functional.normalize(second_q, dim=-1)
    return torch.cat(
        (
            first_t + first_s * _rotate_vector(first_q, second_t),
            _quaternion_multiply(first_q, second_q),
            first_s * second_s,
        ),
        dim=-1,
    )


def _transform_points(state: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    while state.ndim < points.ndim:
        state = state.unsqueeze(0)
    return state[..., :3] + _rotate_vector(state[..., 3:7], state[..., 7:] * points)


def _tensor(data: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = torch.from_numpy(np.array(data, copy=True))
    return result.to(dtype=dtype) if dtype is not None else result


def _require(data: Any, path: Path, *names: str) -> None:
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")


class _Skeleton(torch.nn.Module):
    def __init__(
        self,
        offsets: torch.Tensor,
        prerotations: torch.Tensor,
        parents: torch.Tensor,
        pmi: torch.Tensor,
        pmi_sizes: list[int],
        joint_names: list[str],
    ) -> None:
        super().__init__()
        self.register_buffer("joint_translation_offsets", offsets)
        self.register_buffer("joint_prerotations", prerotations)
        self.register_buffer("pmi", pmi)
        self.register_buffer("joint_parents", parents)
        self._pmi_buffer_sizes = pmi_sizes
        self.joint_names = joint_names

    def forward(self, joint_parameters: torch.Tensor) -> torch.Tensor:
        if joint_parameters.ndim == 2:
            joint_parameters = joint_parameters.reshape(
                joint_parameters.shape[0], -1, PARAMETERS_PER_JOINT
            )
        if joint_parameters.ndim != 3 or joint_parameters.shape[-1] != 7:
            raise ValueError("joint parameters must have shape (..., joints, 7)")

        local = torch.cat(
            (
                joint_parameters[..., :3] + self.joint_translation_offsets.unsqueeze(0),
                _quaternion_multiply(
                    self.joint_prerotations.unsqueeze(0),
                    _euler_xyz_to_quaternion(joint_parameters[..., 3:6]),
                ),
                torch.exp(0.6931471824645996 * joint_parameters[..., 6:]),
            ),
            dim=-1,
        )

        dtype = local.dtype
        result = local.double()
        for indices in self.pmi.split(self._pmi_buffer_sizes, dim=1):
            child, parent = indices[0], indices[1]
            combined = _state_multiply(
                result.index_select(-2, parent), result.index_select(-2, child)
            )
            result = result.index_copy(-2, child, combined)
        return result.to(dtype)


class _Mesh(torch.nn.Module):
    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        texcoords: torch.Tensor,
        texcoord_faces: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("rest_vertices", vertices)
        self.register_buffer("faces", faces)
        self.register_buffer("texcoords", texcoords)
        self.register_buffer("texcoord_faces", texcoord_faces)


class _BlendShape(torch.nn.Module):
    def __init__(self, base_shape: torch.Tensor, shape_vectors: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("base_shape", base_shape)
        self.register_buffer("shape_vectors", shape_vectors)

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self.base_shape + torch.einsum(
            "nvd,...n->...vd", self.shape_vectors, coefficients
        )


class _ParameterTransform(torch.nn.Module):
    def __init__(
        self,
        transform: torch.Tensor,
        pose_parameters: torch.Tensor,
        rigid_parameters: torch.Tensor,
        scaling_parameters: torch.Tensor,
        active_joints: torch.Tensor,
        parameter_names: list[str],
    ) -> None:
        super().__init__()
        self.register_buffer("parameter_transform", transform)
        self.register_buffer("pose_parameters", pose_parameters)
        self.register_buffer("rigid_parameters", rigid_parameters)
        self.register_buffer("scaling_parameters", scaling_parameters)
        self.register_buffer("active_joints", active_joints)
        self.parameter_names = parameter_names

    def forward(self, model_parameters: torch.Tensor) -> torch.Tensor:
        return torch.einsum("dn,...n->...d", self.parameter_transform, model_parameters)


class _ParameterLimits(torch.nn.Module):
    def __init__(
        self,
        minimum: torch.Tensor,
        maximum: torch.Tensor,
        weight: torch.Tensor,
        parameter_index: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("minmax_min", minimum)
        self.register_buffer("minmax_max", maximum)
        self.register_buffer("minmax_weight", weight)
        self.register_buffer("minmax_parameter_index", parameter_index)


class _LinearBlendSkinning(torch.nn.Module):
    def __init__(
        self,
        inverse_bind_pose: torch.Tensor,
        skin_indices: torch.Tensor,
        skin_weights: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("skin_indices", skin_indices)
        self.register_buffer("skin_weights", skin_weights)
        mask = skin_weights.flatten() > 1e-5
        self.register_buffer("inverse_bind_pose", inverse_bind_pose)
        self.register_buffer("skin_indices_flattened", skin_indices.flatten()[mask])
        self.register_buffer("skin_weights_flattened", skin_weights.flatten()[mask])
        vertex_indices = (
            torch.arange(skin_indices.shape[0], dtype=torch.long)[:, None]
            .expand_as(skin_indices)
            .reshape(-1)
        )
        self.register_buffer("vert_indices_flattened", vertex_indices[mask])
        self.num_vertices = skin_indices.shape[0]

    def forward(
        self, skeleton_state: torch.Tensor, rest_vertices: torch.Tensor
    ) -> torch.Tensor:
        if rest_vertices.shape[-2:] != (self.num_vertices, 3):
            raise ValueError(
                f"expected (..., {self.num_vertices}, 3) rest vertices, "
                f"got {tuple(rest_vertices.shape)}"
            )
        while rest_vertices.ndim < skeleton_state.ndim:
            rest_vertices = rest_vertices.unsqueeze(0)
        rest_vertices = rest_vertices.expand(
            *skeleton_state.shape[:-2], self.num_vertices, 3
        )

        joint_state = _state_multiply(skeleton_state, self.inverse_bind_pose)
        transformed = _transform_points(
            joint_state.index_select(-2, self.skin_indices_flattened),
            rest_vertices.index_select(-2, self.vert_indices_flattened),
        )
        weights = self.skin_weights_flattened.reshape(
            *((1,) * (transformed.ndim - 2)), -1, 1
        )
        return torch.zeros_like(rest_vertices).index_add(
            -2,
            self.vert_indices_flattened,
            transformed * weights,
        )


class TorchCharacter(torch.nn.Module):
    """Minimal differentiable character runtime used by MHR."""

    def __init__(
        self,
        *,
        skeleton: _Skeleton,
        mesh: _Mesh,
        parameter_transform: _ParameterTransform,
        linear_blend_skinning: _LinearBlendSkinning,
        blend_shape: _BlendShape,
        parameter_limits: _ParameterLimits,
    ) -> None:
        super().__init__()
        self.skeleton = skeleton
        self.mesh = mesh
        self.parameter_transform = parameter_transform
        self.linear_blend_skinning = linear_blend_skinning
        self.blend_shape = blend_shape
        self.parameter_limits = parameter_limits

    def model_parameters_to_joint_parameters(
        self, model_parameters: torch.Tensor
    ) -> torch.Tensor:
        return self.parameter_transform(model_parameters)

    def joint_parameters_to_skeleton_state(
        self, joint_parameters: torch.Tensor
    ) -> torch.Tensor:
        return self.skeleton(joint_parameters)

    def skin_points(
        self, skel_state: torch.Tensor, rest_vertex_positions: torch.Tensor
    ) -> torch.Tensor:
        return self.linear_blend_skinning(skel_state, rest_vertex_positions)

    @classmethod
    def from_files(
        cls, rig_path: Path, lod_path: Path, expected_lod: int
    ) -> tuple["TorchCharacter", object]:
        with (
            np.load(rig_path, allow_pickle=False) as rig,
            np.load(lod_path, allow_pickle=False) as lod,
        ):
            _validate_assets(rig, lod, rig_path, lod_path, expected_lod)
            character = cls(
                skeleton=_Skeleton(
                    _tensor(rig["joint_offsets"], torch.float32),
                    _tensor(rig["joint_prerotations"], torch.float32),
                    _tensor(rig["joint_parents"], torch.int32),
                    _tensor(rig["fk_indices"], torch.int64),
                    [int(value) for value in rig["fk_sizes"]],
                    [str(value) for value in rig["joint_names"]],
                ),
                mesh=_Mesh(
                    _tensor(lod["base_shape"], torch.float32),
                    _tensor(lod["faces"], torch.int32),
                    _tensor(lod["texcoords"], torch.float32),
                    _tensor(lod["texcoord_faces"], torch.int32),
                ),
                parameter_transform=_ParameterTransform(
                    _tensor(rig["parameter_transform"], torch.float32),
                    _tensor(rig["pose_parameters"], torch.bool),
                    _tensor(rig["rigid_parameters"], torch.bool),
                    _tensor(rig["scaling_parameters"], torch.bool),
                    _tensor(rig["active_joints"], torch.bool),
                    [str(value) for value in rig["parameter_names"]],
                ),
                linear_blend_skinning=_LinearBlendSkinning(
                    _tensor(rig["inverse_bind_pose"], torch.float32),
                    _tensor(lod["skin_indices"], torch.int32),
                    _tensor(lod["skin_weights"], torch.float32),
                ),
                blend_shape=_BlendShape(
                    _tensor(lod["base_shape"], torch.float32),
                    _tensor(lod["shape_vectors"], torch.float32),
                ),
                parameter_limits=_ParameterLimits(
                    _tensor(rig["limit_min"], torch.float32),
                    _tensor(rig["limit_max"], torch.float32),
                    _tensor(rig["limit_weight"], torch.float32),
                    _tensor(rig["limit_parameter_index"], torch.int32),
                ),
            )
            view = _compatibility_view(rig, lod)
        return character, view


def _validate_assets(
    rig: Any, lod: Any, rig_path: Path, lod_path: Path, expected_lod: int
) -> None:
    _require(
        rig,
        rig_path,
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
    _require(
        lod,
        lod_path,
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
    for path, data in ((rig_path, rig), (lod_path, lod)):
        if int(data["schema_version"].item()) != RIG_SCHEMA_VERSION:
            raise ValueError(f"unsupported MHR asset schema in {path}")
    if int(lod["lod"].item()) != expected_lod:
        raise ValueError(f"{lod_path} contains data for the wrong LOD")

    joint_count = rig["joint_offsets"].shape[0]
    vertex_count = lod["base_shape"].shape[0]
    if rig["parameter_transform"].shape[0] != joint_count * PARAMETERS_PER_JOINT:
        raise ValueError(f"invalid parameter transform shape in {rig_path}")
    if rig["joint_prerotations"].shape != (joint_count, 4):
        raise ValueError(f"invalid joint prerotation shape in {rig_path}")
    if rig["joint_parents"].shape != (joint_count,):
        raise ValueError(f"invalid joint parent shape in {rig_path}")
    if rig["inverse_bind_pose"].shape != (joint_count, 8):
        raise ValueError(f"invalid inverse bind pose shape in {rig_path}")
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
    if int(rig["fk_sizes"].sum()) != rig["fk_indices"].shape[1]:
        raise ValueError(f"invalid FK schedule in {rig_path}")
    if np.any(lod["skin_indices"] < 0) or np.any(lod["skin_indices"] >= joint_count):
        raise ValueError(f"skinning joint index out of range in {lod_path}")


def _compatibility_view(rig: Any, lod: Any) -> object:
    parameter_transform = SimpleNamespace(
        size=int(rig["parameter_transform"].shape[1]),
        names=[str(value) for value in rig["parameter_names"]],
        transform=_tensor(rig["parameter_transform"], torch.float32),
        pose_parameters=_tensor(rig["pose_parameters"], torch.bool),
        rigid_parameters=_tensor(rig["rigid_parameters"], torch.bool),
        scaling_parameters=_tensor(rig["scaling_parameters"], torch.bool),
        active_joints=_tensor(rig["active_joints"], torch.bool),
    )
    return SimpleNamespace(
        mesh=SimpleNamespace(
            vertices=np.array(lod["base_shape"], copy=True),
            faces=np.array(lod["faces"], copy=True),
            texcoords=np.array(lod["texcoords"], copy=True),
            texcoord_faces=np.array(lod["texcoord_faces"], copy=True),
        ),
        skeleton=SimpleNamespace(
            joint_names=[str(value) for value in rig["joint_names"]],
            joint_parents=[int(value) for value in rig["joint_parents"]],
            offsets=np.array(rig["joint_offsets"], copy=True),
            pre_rotations=np.array(rig["joint_prerotations"], copy=True),
        ),
        parameter_transform=parameter_transform,
        skin_weights=SimpleNamespace(
            index=np.array(lod["skin_indices"], copy=True),
            weight=np.array(lod["skin_weights"], copy=True),
        ),
        blend_shape=SimpleNamespace(
            base_shape=np.array(lod["base_shape"], copy=True),
            shape_vectors=np.array(lod["shape_vectors"], copy=True),
            n_shapes=int(lod["shape_vectors"].shape[0]),
        ),
    )
