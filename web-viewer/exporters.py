"""Server-side mesh exporters for the MHR web viewer."""

from __future__ import annotations

from dataclasses import dataclass
import json
import struct

import numpy as np
import trimesh

SUPPORTED_EXPORT_FORMATS = ("obj", "glb", "usd")
RIGGED_EXPORT_FORMATS = ("glb", "usd")


@dataclass(frozen=True)
class ExportResult:
    data: bytes
    extension: str
    mimetype: str


@dataclass(frozen=True)
class RigData:
    joint_names: tuple[str, ...]
    joint_parents: np.ndarray
    skeleton_state: np.ndarray
    joint_indices: np.ndarray
    joint_weights: np.ndarray


@dataclass(frozen=True)
class _ValidatedRig:
    joint_names: tuple[str, ...]
    joint_parents: np.ndarray
    global_transforms: np.ndarray
    local_transforms: np.ndarray
    inverse_bind_matrices: np.ndarray
    joint_indices: np.ndarray
    joint_weights: np.ndarray


def normalize_export_format(file_format: str) -> str:
    """Return a supported lowercase extension without a leading dot."""

    normalized = file_format.strip().lower().lstrip(".")
    if normalized not in SUPPORTED_EXPORT_FORMATS:
        supported = ", ".join(SUPPORTED_EXPORT_FORMATS)
        raise ValueError(f"Unsupported export format. Choose one of: {supported}.")
    return normalized


def _validated_mesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    vertex_data = np.ascontiguousarray(vertices, dtype=np.float32)
    face_data = np.ascontiguousarray(faces, dtype=np.int64)
    if vertex_data.ndim != 2 or vertex_data.shape[1] != 3:
        raise ValueError(
            f"Expected vertices with shape (N, 3), got {vertex_data.shape}."
        )
    if face_data.ndim != 2 or face_data.shape[1] != 3:
        raise ValueError(f"Expected faces with shape (M, 3), got {face_data.shape}.")
    if not np.isfinite(vertex_data).all():
        raise ValueError("Mesh vertices contain non-finite values.")
    if face_data.size and (
        int(face_data.min()) < 0 or int(face_data.max()) >= len(vertex_data)
    ):
        raise ValueError("Mesh faces contain an invalid vertex index.")

    mesh = trimesh.Trimesh(
        vertices=vertex_data,
        faces=face_data,
        process=False,
        validate=False,
        metadata={"units": "m", "up_axis": "Y"},
    )
    mesh.units = "m"
    return mesh


def _skeleton_state_to_matrices(skeleton_state: np.ndarray) -> np.ndarray:
    translations = skeleton_state[:, :3]
    quaternions = skeleton_state[:, 3:7].copy()
    scales = skeleton_state[:, 7]
    quaternion_norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(quaternion_norms <= 1e-8):
        raise ValueError("Rig contains a zero-length joint quaternion.")
    quaternions /= quaternion_norms

    x, y, z, w = quaternions.T
    rotations = np.empty((len(skeleton_state), 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    rotations *= scales[:, np.newaxis, np.newaxis]

    matrices = np.repeat(
        np.eye(4, dtype=np.float64)[np.newaxis, :, :],
        len(skeleton_state),
        axis=0,
    )
    matrices[:, :3, :3] = rotations
    matrices[:, :3, 3] = translations
    return matrices


def _validated_rig(rig: RigData, vertex_count: int) -> _ValidatedRig:
    names = tuple(str(name) for name in rig.joint_names)
    parents = np.ascontiguousarray(rig.joint_parents, dtype=np.int32)
    skeleton_state = np.ascontiguousarray(rig.skeleton_state, dtype=np.float64)
    joint_indices = np.ascontiguousarray(rig.joint_indices, dtype=np.uint16)
    joint_weights = np.ascontiguousarray(rig.joint_weights, dtype=np.float32)
    joint_count = len(names)

    if len(set(names)) != joint_count:
        raise ValueError("Rig joint names must be unique.")
    if parents.shape != (joint_count,):
        raise ValueError("Rig parent array does not match the joint count.")
    if skeleton_state.shape != (joint_count, 8):
        raise ValueError("Expected skeleton state with shape (J, 8).")
    if joint_indices.shape != (vertex_count, 4):
        raise ValueError("Expected four joint indices per mesh vertex.")
    if joint_weights.shape != (vertex_count, 4):
        raise ValueError("Expected four joint weights per mesh vertex.")
    if not np.isfinite(skeleton_state).all():
        raise ValueError("Rig skeleton state contains non-finite values.")
    if not np.isfinite(joint_weights).all() or np.any(joint_weights < 0):
        raise ValueError("Rig joint weights must be finite and non-negative.")
    if joint_indices.size and int(joint_indices.max()) >= joint_count:
        raise ValueError("Rig contains a joint index outside the skeleton.")
    for index, parent in enumerate(parents):
        if parent < -1 or parent >= index:
            raise ValueError("Rig parents must precede their children in joint order.")

    weight_sums = joint_weights.sum(axis=1, keepdims=True)
    if np.any(weight_sums <= 1e-8):
        raise ValueError("Every mesh vertex must have a joint influence.")
    joint_weights = joint_weights / weight_sums

    global_transforms = _skeleton_state_to_matrices(skeleton_state)
    if np.any(np.abs(np.linalg.det(global_transforms[:, :3, :3])) <= 1e-10):
        raise ValueError("Rig contains a non-invertible joint transform.")
    inverse_bind_matrices = np.linalg.inv(global_transforms)
    local_transforms = global_transforms.copy()
    for index, parent in enumerate(parents):
        if parent >= 0:
            local_transforms[index] = (
                inverse_bind_matrices[parent] @ global_transforms[index]
            )

    return _ValidatedRig(
        joint_names=names,
        joint_parents=parents,
        global_transforms=global_transforms,
        local_transforms=local_transforms,
        inverse_bind_matrices=inverse_bind_matrices,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
    )


def _export_obj(mesh: trimesh.Trimesh) -> ExportResult:
    payload = trimesh.exchange.obj.export_obj(
        mesh,
        include_normals=True,
        include_color=False,
        include_texture=False,
    )
    header = "# MHR mesh export\n# units: meters\n# up_axis: Y\n"
    return ExportResult(
        data=(header + payload).encode("utf-8"),
        extension="obj",
        mimetype="text/plain; charset=utf-8",
    )


class _GlbBuilder:
    def __init__(self) -> None:
        self.payload = bytearray()
        self.buffer_views: list[dict[str, object]] = []
        self.accessors: list[dict[str, object]] = []

    def add_accessor(
        self,
        values: np.ndarray,
        *,
        component_type: int,
        accessor_type: str,
        target: int | None = None,
        include_bounds: bool = False,
        matrix_column_major: bool = False,
    ) -> int:
        while len(self.payload) % 4:
            self.payload.append(0)
        offset = len(self.payload)
        array = np.ascontiguousarray(values)
        if matrix_column_major:
            array = np.ascontiguousarray(
                np.stack(
                    [matrix.reshape(-1, order="F") for matrix in array],
                    axis=0,
                )
            )
        raw = array.tobytes()
        self.payload.extend(raw)
        view: dict[str, object] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(raw),
        }
        if target is not None:
            view["target"] = target
        self.buffer_views.append(view)

        accessor: dict[str, object] = {
            "bufferView": len(self.buffer_views) - 1,
            "byteOffset": 0,
            "componentType": component_type,
            "count": int(len(array)),
            "type": accessor_type,
        }
        if include_bounds:
            accessor["min"] = np.min(values, axis=0).astype(float).tolist()
            accessor["max"] = np.max(values, axis=0).astype(float).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def build(self, document: dict[str, object]) -> bytes:
        document["buffers"] = [{"byteLength": len(self.payload)}]
        document["bufferViews"] = self.buffer_views
        document["accessors"] = self.accessors
        json_payload = json.dumps(
            document,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        json_payload += b" " * ((-len(json_payload)) % 4)
        binary_payload = bytes(self.payload)
        binary_payload += b"\0" * ((-len(binary_payload)) % 4)
        total_length = 12 + 8 + len(json_payload) + 8 + len(binary_payload)
        return b"".join(
            (
                struct.pack("<4sII", b"glTF", 2, total_length),
                struct.pack("<II", len(json_payload), 0x4E4F534A),
                json_payload,
                struct.pack("<II", len(binary_payload), 0x004E4942),
                binary_payload,
            )
        )


def _export_glb(mesh: trimesh.Trimesh, rig: _ValidatedRig) -> ExportResult:
    builder = _GlbBuilder()
    positions = np.asarray(mesh.vertices, dtype="<f4")
    normals = np.asarray(mesh.vertex_normals, dtype="<f4")
    faces = np.asarray(mesh.faces, dtype="<u4").reshape(-1)
    joints = np.asarray(rig.joint_indices, dtype="<u2")
    weights = np.asarray(rig.joint_weights, dtype="<f4")
    inverse_binds = np.asarray(rig.inverse_bind_matrices, dtype="<f4")

    position_accessor = builder.add_accessor(
        positions,
        component_type=5126,
        accessor_type="VEC3",
        target=34962,
        include_bounds=True,
    )
    normal_accessor = builder.add_accessor(
        normals,
        component_type=5126,
        accessor_type="VEC3",
        target=34962,
    )
    joints_accessor = builder.add_accessor(
        joints,
        component_type=5123,
        accessor_type="VEC4",
        target=34962,
    )
    weights_accessor = builder.add_accessor(
        weights,
        component_type=5126,
        accessor_type="VEC4",
        target=34962,
    )
    index_accessor = builder.add_accessor(
        faces,
        component_type=5125,
        accessor_type="SCALAR",
        target=34963,
    )
    inverse_bind_accessor = builder.add_accessor(
        inverse_binds,
        component_type=5126,
        accessor_type="MAT4",
        matrix_column_major=True,
    )

    nodes: list[dict[str, object]] = []
    for index, (name, matrix) in enumerate(
        zip(rig.joint_names, rig.local_transforms, strict=True)
    ):
        node: dict[str, object] = {
            "name": name,
            "matrix": matrix.reshape(-1, order="F").astype(float).tolist(),
        }
        children = np.flatnonzero(rig.joint_parents == index).tolist()
        if children:
            node["children"] = [int(child) for child in children]
        nodes.append(node)

    mesh_node_index = len(nodes)
    nodes.append({"name": "MHR_Mesh", "mesh": 0, "skin": 0})
    root_nodes = [int(index) for index in np.flatnonzero(rig.joint_parents < 0)]
    root_nodes.append(mesh_node_index)
    document: dict[str, object] = {
        "asset": {
            "version": "2.0",
            "generator": "MHR Web Viewer",
        },
        "scene": 0,
        "scenes": [{"name": "MHR", "nodes": root_nodes}],
        "nodes": nodes,
        "meshes": [
            {
                "name": "MHR_Mesh",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                            "JOINTS_0": joints_accessor,
                            "WEIGHTS_0": weights_accessor,
                        },
                        "indices": index_accessor,
                        "mode": 4,
                    }
                ],
            }
        ],
        "skins": [
            {
                "name": "MHR_Armature",
                "inverseBindMatrices": inverse_bind_accessor,
                "joints": list(range(len(rig.joint_names))),
                "skeleton": root_nodes[0],
            }
        ],
    }
    payload = builder.build(document)
    return ExportResult(
        data=payload,
        extension="glb",
        mimetype="model/gltf-binary",
    )


def _joint_paths(
    names: tuple[str, ...],
    parents: np.ndarray,
) -> list[str]:
    paths: list[str] = []
    for index, name in enumerate(names):
        parent = int(parents[index])
        paths.append(name if parent < 0 else f"{paths[parent]}/{name}")
    return paths


def _export_usd(mesh: trimesh.Trimesh, rig: _ValidatedRig) -> ExportResult:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt
    except ImportError as error:
        raise RuntimeError(
            "USD export requires the usd-core Python package."
        ) from error

    layer = Sdf.Layer.CreateAnonymous("mhr.usda")
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    skel_root = UsdSkel.Root.Define(stage, "/MHR")
    skeleton = UsdSkel.Skeleton.Define(stage, "/MHR/Skeleton")
    joint_paths = _joint_paths(rig.joint_names, rig.joint_parents)
    skeleton.CreateJointsAttr(Vt.TokenArray(joint_paths))
    skeleton.CreateBindTransformsAttr(
        Vt.Matrix4dArray(
            [Gf.Matrix4d(matrix.T.tolist()) for matrix in rig.global_transforms]
        )
    )
    skeleton.CreateRestTransformsAttr(
        Vt.Matrix4dArray(
            [Gf.Matrix4d(matrix.T.tolist()) for matrix in rig.local_transforms]
        )
    )

    usd_mesh = UsdGeom.Mesh.Define(stage, "/MHR/Mesh")
    usd_mesh.CreatePointsAttr(
        Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32))
    )
    usd_mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, dtype=np.int32))
    )
    usd_mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype=np.int32).reshape(-1))
    )
    usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    normals = usd_mesh.CreateNormalsAttr(
        Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals, dtype=np.float32))
    )
    normals.SetMetadata("interpolation", UsdGeom.Tokens.vertex)

    binding = UsdSkel.BindingAPI.Apply(usd_mesh.GetPrim())
    binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateGeomBindTransformAttr().Set(Gf.Matrix4d(1.0))
    joint_indices = binding.CreateJointIndicesPrimvar(False, 4)
    joint_indices.Set(
        Vt.IntArray.FromNumpy(np.asarray(rig.joint_indices, dtype=np.int32).reshape(-1))
    )
    joint_indices.SetElementSize(4)
    joint_weights = binding.CreateJointWeightsPrimvar(False, 4)
    joint_weights.Set(
        Vt.FloatArray.FromNumpy(
            np.asarray(rig.joint_weights, dtype=np.float32).reshape(-1)
        )
    )
    joint_weights.SetElementSize(4)
    stage.SetDefaultPrim(skel_root.GetPrim())

    return ExportResult(
        data=stage.GetRootLayer().ExportToString().encode("utf-8"),
        extension="usd",
        mimetype="model/vnd.usd",
    )


def export_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    file_format: str,
    rig: RigData | None = None,
) -> ExportResult:
    """Export one immutable mesh snapshot in the requested format."""

    normalized_format = normalize_export_format(file_format)
    mesh = _validated_mesh(vertices, faces)
    validated_rig = _validated_rig(rig, len(mesh.vertices)) if rig is not None else None
    if normalized_format in RIGGED_EXPORT_FORMATS and validated_rig is None:
        raise ValueError(f"{normalized_format.upper()} export requires rig data.")
    if normalized_format == "obj":
        return _export_obj(mesh)
    assert validated_rig is not None
    if normalized_format == "glb":
        return _export_glb(mesh, validated_rig)
    return _export_usd(mesh, validated_rig)
