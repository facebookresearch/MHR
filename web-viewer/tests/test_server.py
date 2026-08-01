from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom, UsdSkel
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from server import DEFAULT_MODEL_PATH, create_app  # noqa: E402


def _parse_glb(payload: bytes) -> tuple[dict[str, object], bytes]:
    magic, version, total_length = struct.unpack_from("<4sII", payload, 0)
    assert magic == b"glTF"
    assert version == 2
    assert total_length == len(payload)
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    json_start = 20
    document = json.loads(
        payload[json_start : json_start + json_length].decode("utf-8")
    )
    binary_header = json_start + json_length
    binary_length, binary_type = struct.unpack_from(
        "<II",
        payload,
        binary_header,
    )
    assert binary_type == 0x004E4942
    binary_start = binary_header + 8
    return document, payload[binary_start : binary_start + binary_length]


def _glb_accessor(
    document: dict[str, object],
    binary: bytes,
    accessor_index: int,
) -> np.ndarray:
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    component_dtypes = {
        5121: np.dtype("<u1"),
        5123: np.dtype("<u2"),
        5125: np.dtype("<u4"),
        5126: np.dtype("<f4"),
    }
    component_counts = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }
    dtype = component_dtypes[accessor["componentType"]]
    components = component_counts[accessor["type"]]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    values = np.frombuffer(
        binary,
        dtype=dtype,
        count=int(accessor["count"]) * components,
        offset=offset,
    ).reshape(int(accessor["count"]), components)
    if accessor["type"] == "MAT4":
        return values.reshape(-1, 4, 4).transpose(0, 2, 1)
    return values


@pytest.fixture(scope="module")
def client():
    app = create_app(DEFAULT_MODEL_PATH)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_health_and_metadata(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json["status"] == "ok"
    assert health.json["vertices"] == 18439
    assert health.json["faces"] == 36874

    metadata = client.get("/api/metadata")
    assert metadata.status_code == 200
    assert len(metadata.json["groups"]["pose"]) == 204
    assert len(metadata.json["groups"]["identity"]) == 45
    assert len(metadata.json["groups"]["expression"]) == 72
    assert metadata.json["model"]["jointCount"] == 127
    assert metadata.json["model"]["exportFormats"] == ["obj", "glb", "usd"]
    assert metadata.json["model"]["riggedExportFormats"] == ["glb", "usd"]
    assert metadata.json["model"]["snapToGround"] is True
    root_ty = next(
        item for item in metadata.json["groups"]["pose"] if item["name"] == "root_ty"
    )
    assert root_ty["managed"] is True


def test_topology_and_geometry_payload_sizes(client):
    metadata = client.get("/api/metadata").json
    topology = client.get("/api/topology")
    geometry = client.get("/api/geometry")

    expected_topology_bytes = metadata["model"]["faceCount"] * 3 * 4
    expected_geometry_bytes = (
        (metadata["model"]["vertexCount"] + metadata["model"]["jointCount"]) * 3 * 4
    )

    assert topology.status_code == 200
    assert len(topology.data) == expected_topology_bytes
    assert geometry.status_code == 200
    assert len(geometry.data) == expected_geometry_bytes


def test_pose_deformation_changes_vertices(client):
    metadata = client.get("/api/metadata").json
    knee = next(
        item for item in metadata["groups"]["pose"] if item["name"] == "l_knee_bend"
    )
    neutral = np.frombuffer(client.get("/api/geometry").data, dtype="<f4").copy()

    response = client.post(
        "/api/deform",
        json={
            "action": "set",
            "group": "pose",
            "index": knee["index"],
            "value": 1.0,
        },
    )
    deformed = np.frombuffer(response.data, dtype="<f4")
    vertex_floats = metadata["model"]["vertexCount"] * 3

    assert response.status_code == 200
    assert float(np.abs(deformed[:vertex_floats] - neutral[:vertex_floats]).max()) > 0
    assert float(response.headers["X-MHR-Inference-Ms"]) >= 0


@pytest.mark.parametrize(
    ("group", "index", "value"),
    [
        ("identity", 0, 1.5),
        ("expression", 0, 0.5),
    ],
)
def test_identity_and_expression_deformation(client, group, index, value):
    response = client.post(
        "/api/deform",
        json={
            "action": "set",
            "group": group,
            "index": index,
            "value": value,
        },
    )
    assert response.status_code == 200
    assert response.mimetype == "application/octet-stream"


def test_reset_all_and_validation(client):
    reset = client.post(
        "/api/deform",
        json={"action": "reset_all", "group": "pose", "index": 0},
    )
    assert reset.status_code == 200

    invalid_group = client.post(
        "/api/deform",
        json={"action": "set", "group": "unknown", "index": 0, "value": 0},
    )
    assert invalid_group.status_code == 400

    invalid_index = client.post(
        "/api/deform",
        json={"action": "set", "group": "pose", "index": 999, "value": 0},
    )
    assert invalid_index.status_code == 400


def test_pose_correctives_can_be_toggled(client):
    client.post(
        "/api/deform",
        json={"action": "reset_all", "group": "pose", "index": 0},
    )
    metadata = client.get("/api/metadata").json
    head_lean = next(
        item for item in metadata["groups"]["pose"] if item["name"] == "head_lean"
    )
    corrected = client.post(
        "/api/deform",
        json={
            "action": "set",
            "group": "pose",
            "index": head_lean["index"],
            "value": 0.2,
        },
    )
    uncorrected_config = client.post(
        "/api/configure",
        json={"applyCorrectives": False},
    )
    uncorrected = client.get("/api/geometry")

    corrected_values = np.frombuffer(corrected.data, dtype="<f4")
    uncorrected_values = np.frombuffer(uncorrected.data, dtype="<f4")
    assert uncorrected_config.status_code == 200
    assert uncorrected_config.json["model"]["applyCorrectives"] is False
    assert float(np.abs(corrected_values - uncorrected_values).max()) > 0

    restored = client.post(
        "/api/configure",
        json={"applyCorrectives": True},
    )
    assert restored.status_code == 200


def test_lod_selection_updates_topology_and_geometry(client):
    configured = client.post("/api/configure", json={"lod": 6})
    assert configured.status_code == 200
    assert configured.json["model"]["lod"] == 6
    assert configured.json["model"]["vertexCount"] == 595
    assert configured.json["model"]["faceCount"] == 1186

    topology = client.get("/api/topology")
    geometry = client.get("/api/geometry")
    expected_geometry_bytes = (
        (
            configured.json["model"]["vertexCount"]
            + configured.json["model"]["jointCount"]
        )
        * 3
        * 4
    )
    assert len(topology.data) == 1186 * 3 * 4
    assert len(geometry.data) == expected_geometry_bytes
    assert topology.headers["Cache-Control"] == "no-store"

    restored = client.post("/api/configure", json={"lod": 1})
    assert restored.status_code == 200
    assert restored.json["model"]["vertexCount"] == 18439


@pytest.mark.parametrize(
    "payload",
    [
        {"lod": -1},
        {"lod": 7},
        {"lod": "high"},
        {"applyCorrectives": "yes"},
        {"snapToGround": "yes"},
    ],
)
def test_model_configuration_validation(client, payload):
    response = client.post("/api/configure", json=payload)
    assert response.status_code == 400


def test_snap_to_ground_is_driven_by_root_ty(client):
    client.post(
        "/api/configure",
        json={"snapToGround": True},
    )
    client.post(
        "/api/deform",
        json={"action": "reset_all", "group": "pose", "index": 0},
    )
    metadata = client.get("/api/metadata").json
    upper_legs = next(
        item for item in metadata["groups"]["pose"] if item["name"] == "scale_uplegs"
    )
    lower_legs = next(
        item for item in metadata["groups"]["pose"] if item["name"] == "scale_lowlegs"
    )
    root_ty = next(
        item for item in metadata["groups"]["pose"] if item["name"] == "root_ty"
    )

    client.post(
        "/api/deform",
        json={
            "action": "set",
            "group": "pose",
            "index": upper_legs["index"],
            "value": upper_legs["max"],
        },
    )
    snapped = client.post(
        "/api/deform",
        json={
            "action": "set",
            "group": "pose",
            "index": lower_legs["index"],
            "value": lower_legs["max"],
        },
    )
    vertex_count = metadata["model"]["vertexCount"]
    snapped_vertices = np.frombuffer(
        snapped.data,
        dtype="<f4",
        count=vertex_count * 3,
    ).reshape(-1, 3)
    snapped_metadata = client.get("/api/metadata").json
    snapped_root_ty = next(
        item for item in snapped_metadata["groups"]["pose"] if item["name"] == "root_ty"
    )

    assert np.min(snapped_vertices[:, 1]) == pytest.approx(0.0, abs=1e-6)
    assert snapped_root_ty["value"] != pytest.approx(0.0, abs=1e-4)
    assert snapped_root_ty["managed"] is True
    assert float(snapped.headers["X-MHR-Root-TY"]) == pytest.approx(
        snapped_root_ty["value"],
        abs=1e-6,
    )

    managed_update = client.post(
        "/api/deform",
        json={
            "action": "set",
            "group": "pose",
            "index": root_ty["index"],
            "value": 1.0,
        },
    )
    assert managed_update.status_code == 400
    assert "managed" in managed_update.json["error"]

    unsnapped = client.post(
        "/api/configure",
        json={"snapToGround": False},
    )
    unsnapped_geometry = np.frombuffer(
        client.get("/api/geometry").data,
        dtype="<f4",
        count=vertex_count * 3,
    ).reshape(-1, 3)
    unsnapped_root_ty = next(
        item for item in unsnapped.json["groups"]["pose"] if item["name"] == "root_ty"
    )
    assert unsnapped.json["model"]["snapToGround"] is False
    assert unsnapped_root_ty["managed"] is False
    assert unsnapped_root_ty["value"] == pytest.approx(0.0, abs=1e-6)
    assert abs(float(np.min(unsnapped_geometry[:, 1]))) > 1e-3

    client.post(
        "/api/deform",
        json={"action": "reset_all", "group": "pose", "index": 0},
    )
    restored = client.post(
        "/api/configure",
        json={"snapToGround": True},
    )
    assert restored.status_code == 200


def test_obj_export_matches_current_mesh(client):
    metadata = client.get("/api/metadata").json
    geometry = np.frombuffer(client.get("/api/geometry").data, dtype="<f4")
    expected_vertices = geometry[: metadata["model"]["vertexCount"] * 3].reshape(-1, 3)
    response = client.get("/api/export/obj")
    exported = trimesh.load(
        BytesIO(response.data),
        file_type="obj",
        force="mesh",
        process=False,
    )

    assert response.status_code == 200
    assert response.data.startswith(b"# MHR mesh export")
    assert response.headers["X-MHR-LOD"] == str(metadata["model"]["lod"])
    assert response.headers["Content-Disposition"].endswith('.obj"')
    assert len(exported.vertices) == metadata["model"]["vertexCount"]
    assert len(exported.faces) == metadata["model"]["faceCount"]
    np.testing.assert_allclose(exported.vertices, expected_vertices, atol=1e-6)


def test_glb_export_matches_current_mesh(client):
    metadata = client.get("/api/metadata").json
    geometry = np.frombuffer(client.get("/api/geometry").data, dtype="<f4")
    expected_vertices = geometry[: metadata["model"]["vertexCount"] * 3].reshape(-1, 3)
    response = client.get("/api/export/glb")
    exported = trimesh.load(
        BytesIO(response.data),
        file_type="glb",
        force="mesh",
        process=False,
    )

    assert response.status_code == 200
    assert response.data[:4] == b"glTF"
    assert response.mimetype == "model/gltf-binary"
    assert response.headers["Content-Disposition"].endswith('.glb"')
    assert len(exported.vertices) == metadata["model"]["vertexCount"]
    assert len(exported.faces) == metadata["model"]["faceCount"]
    np.testing.assert_allclose(exported.vertices, expected_vertices, atol=1e-6)


def test_usd_export_matches_current_mesh(client):
    metadata = client.get("/api/metadata").json
    geometry = np.frombuffer(client.get("/api/geometry").data, dtype="<f4")
    expected_vertices = geometry[: metadata["model"]["vertexCount"] * 3].reshape(-1, 3)
    response = client.get("/api/export/usd")
    layer = Sdf.Layer.CreateAnonymous("test.usda")
    assert layer.ImportFromString(response.data.decode("utf-8"))
    stage = Usd.Stage.Open(layer)
    exported = UsdGeom.Mesh(stage.GetPrimAtPath("/MHR/Mesh"))

    assert response.status_code == 200
    assert response.data.startswith(b"#usda")
    assert response.mimetype == "model/vnd.usd"
    assert response.headers["Content-Disposition"].endswith('.usd"')
    assert len(exported.GetPointsAttr().Get()) == metadata["model"]["vertexCount"]
    assert (
        len(exported.GetFaceVertexCountsAttr().Get()) == metadata["model"]["faceCount"]
    )
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    np.testing.assert_allclose(
        np.asarray(exported.GetPointsAttr().Get()),
        expected_vertices,
        atol=1e-6,
    )


def test_export_validation(client):
    response = client.get("/api/export/fbx")
    assert response.status_code == 400
    assert "Unsupported export format" in response.json["error"]

    response = client.get("/api/export/obj?ground=yes")
    assert response.status_code == 400
    assert "has been removed" in response.json["error"]


def test_glb_export_contains_armature_and_lbs_weights(client):
    metadata = client.get("/api/metadata").json
    response = client.get("/api/export/glb")
    document, binary = _parse_glb(response.data)
    primitive = document["meshes"][0]["primitives"][0]
    skin = document["skins"][0]

    assert len(document["skins"]) == 1
    assert len(skin["joints"]) == metadata["model"]["jointCount"]
    assert "JOINTS_0" in primitive["attributes"]
    assert "WEIGHTS_0" in primitive["attributes"]

    joint_indices = _glb_accessor(
        document,
        binary,
        primitive["attributes"]["JOINTS_0"],
    )
    joint_weights = _glb_accessor(
        document,
        binary,
        primitive["attributes"]["WEIGHTS_0"],
    )
    inverse_binds = _glb_accessor(
        document,
        binary,
        skin["inverseBindMatrices"],
    )
    assert joint_indices.shape == (metadata["model"]["vertexCount"], 4)
    assert joint_weights.shape == (metadata["model"]["vertexCount"], 4)
    assert int(joint_indices.max()) < metadata["model"]["jointCount"]
    np.testing.assert_allclose(
        joint_weights.sum(axis=1),
        1.0,
        atol=1e-6,
    )
    local_transforms = np.asarray(
        [
            np.asarray(document["nodes"][node_index]["matrix"]).reshape(4, 4).T
            for node_index in skin["joints"]
        ]
    )
    global_transforms = local_transforms.copy()
    for index, parent in enumerate(metadata["jointParents"]):
        if parent >= 0:
            global_transforms[index] = (
                global_transforms[parent] @ local_transforms[index]
            )
    np.testing.assert_allclose(
        global_transforms @ inverse_binds,
        np.broadcast_to(
            np.eye(4),
            (metadata["model"]["jointCount"], 4, 4),
        ),
        atol=2e-5,
    )


def test_usd_export_contains_armature_and_lbs_weights(client):
    metadata = client.get("/api/metadata").json
    response = client.get("/api/export/usd")
    layer = Sdf.Layer.CreateAnonymous("rigged-test.usda")
    assert layer.ImportFromString(response.data.decode("utf-8"))
    stage = Usd.Stage.Open(layer)
    root = UsdSkel.Root(stage.GetPrimAtPath("/MHR"))
    skeleton = UsdSkel.Skeleton(stage.GetPrimAtPath("/MHR/Skeleton"))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/MHR/Mesh"))
    binding = UsdSkel.BindingAPI(mesh.GetPrim())

    assert root
    assert skeleton
    expected_joint_paths = []
    for index, name in enumerate(metadata["jointNames"]):
        parent = metadata["jointParents"][index]
        expected_joint_paths.append(
            name if parent < 0 else f"{expected_joint_paths[parent]}/{name}"
        )
    exported_joint_paths = [str(path) for path in skeleton.GetJointsAttr().Get()]
    assert exported_joint_paths == expected_joint_paths
    assert len(exported_joint_paths) == metadata["model"]["jointCount"]
    assert (
        len(skeleton.GetBindTransformsAttr().Get()) == metadata["model"]["jointCount"]
    )
    assert (
        len(skeleton.GetRestTransformsAttr().Get()) == metadata["model"]["jointCount"]
    )
    assert binding.GetSkeletonRel().GetTargets() == [skeleton.GetPath()]

    joint_indices_primvar = binding.GetJointIndicesPrimvar()
    joint_weights_primvar = binding.GetJointWeightsPrimvar()
    joint_indices = np.asarray(
        joint_indices_primvar.Get(),
        dtype=np.int32,
    ).reshape(-1, 4)
    joint_weights = np.asarray(
        joint_weights_primvar.Get(),
        dtype=np.float32,
    ).reshape(-1, 4)
    assert joint_indices_primvar.GetElementSize() == 4
    assert joint_weights_primvar.GetElementSize() == 4
    assert joint_indices.shape == (metadata["model"]["vertexCount"], 4)
    assert joint_weights.shape == (metadata["model"]["vertexCount"], 4)
    assert int(joint_indices.max()) < metadata["model"]["jointCount"]
    np.testing.assert_allclose(
        joint_weights.sum(axis=1),
        1.0,
        atol=1e-6,
    )
    cache = UsdSkel.Cache()
    assert cache.Populate(root, Usd.PrimDefaultPredicate)
    skeleton_query = cache.GetSkelQuery(skeleton)
    assert skeleton_query
    assert (
        len(skeleton_query.ComputeSkinningTransforms())
        == metadata["model"]["jointCount"]
    )


def test_all_lods_have_normalized_four_influence_skinning(client):
    engine = client.application.config["MHR_ENGINE"]
    metadata = client.get("/api/metadata").json
    expected_counts = {
        entry["lod"]: entry["vertexCount"]
        for entry in metadata["model"]["supportedLods"]
    }

    with engine.lock:
        for lod, vertex_count in expected_counts.items():
            joint_indices, joint_weights = engine._skinning_for_lod(lod)
            assert joint_indices.shape == (vertex_count, 4)
            assert joint_weights.shape == (vertex_count, 4)
            assert int(joint_indices.max()) < metadata["model"]["jointCount"]
            assert np.all(joint_weights >= 0)
            np.testing.assert_allclose(
                joint_weights.sum(axis=1),
                1.0,
                atol=1e-6,
            )


@pytest.mark.parametrize("file_format", ["obj", "glb", "usd"])
def test_snapped_export_uses_grounded_model_state(client, file_format):
    response = client.get(f"/api/export/{file_format}")

    assert response.status_code == 200
    assert response.headers["X-MHR-Snap-To-Ground"] == "true"
    assert "-grounded." in response.headers["Content-Disposition"]

    if file_format == "usd":
        layer = Sdf.Layer.CreateAnonymous("grounded-test.usda")
        assert layer.ImportFromString(response.data.decode("utf-8"))
        stage = Usd.Stage.Open(layer)
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/MHR/Mesh"))
        vertices = np.asarray(mesh.GetPointsAttr().Get())
    else:
        mesh = trimesh.load(
            BytesIO(response.data),
            file_type=file_format,
            force="mesh",
            process=False,
        )
        vertices = np.asarray(mesh.vertices)

    assert np.min(vertices[:, 1]) == pytest.approx(0.0, abs=1e-6)
