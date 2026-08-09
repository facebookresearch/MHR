"""Local Three.js web viewer backend for the Momentum Human Rig model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import webbrowser

from flask import Flask, Response, jsonify, request, send_from_directory
import numpy as np
import torch

from exporters import (
    RIGGED_EXPORT_FORMATS,
    RigData,
    SUPPORTED_EXPORT_FORMATS,
    export_mesh,
    normalize_export_format,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT.parent / "assets" / "mhr_model.pt"
STATIC_ROOT = PROJECT_ROOT / "static"
THREE_ROOT = PROJECT_ROOT / "node_modules" / "three"
LOD_MAPPING_ROOT = PROJECT_ROOT.parent / "tools" / "mhr_LOD_conversion"
LOD_TOPOLOGY_PATH = PROJECT_ROOT / "data" / "lod_topology.npz"
SUPPORTED_LODS = tuple(range(7))
ROOT_TY_NAME = "root_ty"
ROOT_TY_SOURCE_UNITS_PER_VALUE = 10.0
IDENTITY_RANGE = (-3.0, 3.0)
EXPRESSION_RANGE = (-1.0, 1.0)


def _pose_category(name: str) -> str:
    if name.startswith("root_"):
        return "Root"
    if name.startswith("spine"):
        return "Spine"
    if name.startswith(("neck_", "head_")):
        return "Neck & head"
    if name.startswith(("r_clavicle", "r_uparm", "r_elbow", "r_lowarm", "r_wrist")):
        return "Right arm"
    if name.startswith(("l_clavicle", "l_uparm", "l_elbow", "l_lowarm", "l_wrist")):
        return "Left arm"
    if name.startswith(
        (
            "r_upleg",
            "r_knee",
            "r_lowleg",
            "r_foot",
            "r_ball",
            "r_subtalar",
            "r_talocrural",
        )
    ):
        return "Right leg"
    if name.startswith(
        (
            "l_upleg",
            "l_knee",
            "l_lowleg",
            "l_foot",
            "l_ball",
            "l_subtalar",
            "l_talocrural",
        )
    ):
        return "Left leg"
    if name.startswith(("r_thumb", "r_index", "r_middle", "r_ring", "r_pinky")):
        return "Right hand"
    if name.startswith(("l_thumb", "l_index", "l_middle", "l_ring", "l_pinky")):
        return "Left hand"
    if name.startswith("scale_"):
        return "Proportions"
    return "Flexible body"


@dataclass(frozen=True)
class GeometrySnapshot:
    vertices: np.ndarray
    joints: np.ndarray
    skeleton_state: np.ndarray
    revision: int
    inference_ms: float
    root_ty: float

    def to_bytes(self) -> bytes:
        return (
            self.vertices.astype("<f4", copy=False).tobytes()
            + self.joints.astype("<f4", copy=False).tobytes()
        )


class MHREngine:
    """Thread-safe state and inference wrapper around the TorchScript model."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MHR model not found at {model_path}. "
                "Pass --model with the path to mhr_model.pt."
            )

        self.model_path = model_path
        self.lock = threading.RLock()
        self.model = torch.jit.load(str(model_path), map_location="cpu").eval()
        self.lod = 1
        self.apply_correctives = True
        self.identity_count = int(self.model.get_num_identity_blendshapes())

        all_names = list(self.model.get_parameter_names())
        self.pose_names = all_names[: -self.identity_count]
        all_limits = (
            self.model.get_parameter_limits().detach().cpu().numpy().astype(np.float32)
        )
        self.pose_limits = all_limits[: len(self.pose_names)]
        self.pose_values = np.zeros(len(self.pose_names), dtype=np.float32)
        self.identity_values = np.zeros(self.identity_count, dtype=np.float32)
        self.expression_values = np.zeros(72, dtype=np.float32)
        self.identity_limits = np.tile(
            np.asarray([IDENTITY_RANGE], dtype=np.float32),
            (self.identity_count, 1),
        )
        self.expression_limits = np.tile(
            np.asarray([EXPRESSION_RANGE], dtype=np.float32),
            (len(self.expression_values), 1),
        )
        self.root_ty_index = self.pose_names.index(ROOT_TY_NAME)
        self.unsnapped_root_ty = float(self.pose_values[self.root_ty_index])
        self.snap_to_ground = True

        self.source_faces = (
            self.model.character_torch.mesh.faces.detach()
            .cpu()
            .numpy()
            .astype(np.uint32)
        )
        source_joint_indices, source_joint_weights = self.model.get_lbsw()
        self.source_joint_indices = (
            source_joint_indices.detach().cpu().numpy().astype(np.uint16)
        )
        self.source_joint_weights = (
            source_joint_weights.detach().cpu().numpy().astype(np.float32)
        )
        self.source_vertex_count = int(self.source_joint_indices.shape[0])
        self.lod_faces, self.lod_mappings = self._load_lod_assets()
        self.faces = self.lod_faces[self.lod]
        self.joint_names = list(self.model.get_joint_names())
        self.joint_parents = [
            int(value) for value in self.model.character_torch.skeleton.joint_parents
        ]
        self.lod_skinning: dict[int, tuple[np.ndarray, np.ndarray]] = {
            1: self._top_four_skinning(
                self.source_joint_indices,
                self.source_joint_weights,
            )
        }
        self.influence_mapping = self._build_influence_mapping()
        self.revision = 0
        self.snapshot = self._infer()

    def _load_lod_assets(
        self,
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, tuple[np.ndarray, np.ndarray]],
    ]:
        if not LOD_TOPOLOGY_PATH.is_file():
            raise FileNotFoundError(
                f"LOD topology data not found at {LOD_TOPOLOGY_PATH}. "
                "Run tools/build_lod_topology.py from the viewer folder."
            )

        lod_faces = {1: self.source_faces}
        lod_mappings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        with np.load(LOD_TOPOLOGY_PATH) as topology:
            for lod in SUPPORTED_LODS:
                if lod == 1:
                    continue
                faces_key = f"lod{lod}_faces"
                if faces_key not in topology:
                    raise ValueError(f"LOD topology data is missing {faces_key}.")
                lod_faces[lod] = topology[faces_key].astype(np.uint32)

                mapping_path = LOD_MAPPING_ROOT / f"lod1_to_lod{lod}_mapping.npz"
                if not mapping_path.is_file():
                    raise FileNotFoundError(
                        f"LOD conversion mapping not found at {mapping_path}."
                    )
                with np.load(mapping_path) as mapping:
                    triangle_ids = mapping["triangle_ids"].astype(np.int64)
                    barycentric = mapping["baryc_coords"].astype(np.float32)
                source_indices = self.source_faces[triangle_ids].astype(np.int64)
                if int(lod_faces[lod].max()) >= len(source_indices):
                    raise ValueError(
                        f"LOD {lod} topology references a vertex outside its "
                        "conversion mapping."
                    )
                lod_mappings[lod] = (source_indices, barycentric)
        return lod_faces, lod_mappings

    def _build_influence_mapping(self) -> dict[str, list[str]]:
        num_joints = len(self.joint_names)
        influence_matrix = (
            self.model.get_parameter_transform().detach().cpu().numpy().astype(bool).T
        )
        result: dict[str, list[str]] = {}
        for name, joint_mask in zip(self.pose_names, influence_matrix, strict=False):
            influenced = joint_mask.reshape(num_joints, 7).any(axis=1)
            result[name] = [
                self.joint_names[index] for index in np.flatnonzero(influenced)
            ]
        return result

    @staticmethod
    def _top_four_skinning(
        joint_indices: np.ndarray,
        joint_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(joint_weights, axis=1)[:, -4:][:, ::-1]
        indices = np.take_along_axis(joint_indices, order, axis=1)
        weights = np.take_along_axis(joint_weights, order, axis=1)
        weight_sums = weights.sum(axis=1, keepdims=True)
        if np.any(weight_sums <= 1e-8):
            raise ValueError("Every MHR vertex must have a skin influence.")
        return (
            np.ascontiguousarray(indices, dtype=np.uint16),
            np.ascontiguousarray(weights / weight_sums, dtype=np.float32),
        )

    def _skinning_for_lod(
        self,
        lod: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        cached = self.lod_skinning.get(lod)
        if cached is not None:
            return cached

        source_indices, barycentric = self.lod_mappings[lod]
        vertex_count = len(source_indices)
        joint_count = len(self.joint_names)
        dense_weights = np.zeros(
            (vertex_count, joint_count),
            dtype=np.float32,
        )
        rows = np.arange(vertex_count, dtype=np.int64)[:, np.newaxis]
        for corner in range(3):
            source_vertices = source_indices[:, corner]
            corner_indices = self.source_joint_indices[source_vertices]
            corner_weights = (
                self.source_joint_weights[source_vertices]
                * barycentric[:, corner, np.newaxis]
            )
            np.add.at(
                dense_weights,
                (
                    np.broadcast_to(rows, corner_indices.shape).reshape(-1),
                    corner_indices.astype(np.int64, copy=False).reshape(-1),
                ),
                corner_weights.reshape(-1),
            )
        dense_weights = np.maximum(dense_weights, 0.0)
        dense_indices = np.broadcast_to(
            np.arange(joint_count, dtype=np.uint16),
            dense_weights.shape,
        )
        result = self._top_four_skinning(dense_indices, dense_weights)
        self.lod_skinning[lod] = result
        return result

    def _infer(self) -> GeometrySnapshot:
        started = time.perf_counter()
        source_vertices, skeleton = self._evaluate_model()
        render_vertices = self._render_lod(source_vertices)

        if self.snap_to_ground:
            lowest_y = float(np.min(render_vertices[:, 1]))
            correction = -lowest_y / ROOT_TY_SOURCE_UNITS_PER_VALUE
            if not np.isclose(correction, 0.0, atol=1e-7):
                minimum, maximum = self.pose_limits[self.root_ty_index]
                current = float(self.pose_values[self.root_ty_index])
                self.pose_values[self.root_ty_index] = np.clip(
                    current + correction,
                    minimum,
                    maximum,
                )
                source_vertices, skeleton = self._evaluate_model()
                render_vertices = self._render_lod(source_vertices)

        self.revision += 1
        export_skeleton = skeleton.copy()
        export_skeleton[:, :3] /= 100.0
        return GeometrySnapshot(
            vertices=render_vertices / 100.0,
            joints=export_skeleton[:, :3],
            skeleton_state=export_skeleton,
            revision=self.revision,
            inference_ms=(time.perf_counter() - started) * 1000.0,
            root_ty=float(self.pose_values[self.root_ty_index]),
        )

    def _evaluate_model(self) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            vertices, skeleton = self.model(
                identity_coeffs=torch.from_numpy(
                    self.identity_values[np.newaxis, :].copy()
                ),
                model_parameters=torch.from_numpy(
                    self.pose_values[np.newaxis, :].copy()
                ),
                face_expr_coeffs=torch.from_numpy(
                    self.expression_values[np.newaxis, :].copy()
                ),
                apply_correctives=self.apply_correctives,
            )
        return (
            vertices[0].detach().cpu().numpy().astype(np.float32),
            skeleton[0].detach().cpu().numpy().astype(np.float32),
        )

    def _render_lod(self, source_vertices: np.ndarray) -> np.ndarray:
        if self.lod == 1:
            return source_vertices
        source_indices, barycentric = self.lod_mappings[self.lod]
        triangle_vertices = source_vertices[source_indices]
        return np.einsum("ijk,ij->ik", triangle_vertices, barycentric).astype(
            np.float32
        )

    def metadata(self) -> dict[str, object]:
        pose_parameters = []
        for index, name in enumerate(self.pose_names):
            minimum, maximum = self.pose_limits[index]
            managed = index == self.root_ty_index and self.snap_to_ground
            pose_parameters.append(
                {
                    "index": index,
                    "name": name,
                    "label": name,
                    "category": _pose_category(name),
                    "min": float(minimum),
                    "max": float(maximum),
                    "value": float(self.pose_values[index]),
                    "joints": self.influence_mapping.get(name, []),
                    "managed": managed,
                    "lockedReason": ("Managed by Snap to ground" if managed else None),
                }
            )

        return {
            "model": {
                "path": str(self.model_path),
                "lod": self.lod,
                "supportedLods": [
                    {
                        "lod": lod,
                        "vertexCount": (
                            int(self.snapshot.vertices.shape[0])
                            if lod == self.lod
                            else (
                                self.source_vertex_count
                                if lod == 1
                                else int(self.lod_mappings[lod][0].shape[0])
                            )
                        ),
                        "faceCount": int(self.lod_faces[lod].shape[0]),
                    }
                    for lod in SUPPORTED_LODS
                ],
                "applyCorrectives": self.apply_correctives,
                "snapToGround": self.snap_to_ground,
                "exportFormats": list(SUPPORTED_EXPORT_FORMATS),
                "riggedExportFormats": list(RIGGED_EXPORT_FORMATS),
                "vertexCount": int(self.snapshot.vertices.shape[0]),
                "faceCount": int(self.faces.shape[0]),
                "jointCount": len(self.joint_names),
            },
            "groups": {
                "pose": pose_parameters,
                "identity": self._latent_parameters(
                    "identity",
                    "Identity",
                    self.identity_values,
                    self.identity_limits,
                ),
                "expression": self._latent_parameters(
                    "expression",
                    "Expression",
                    self.expression_values,
                    self.expression_limits,
                ),
            },
            "jointNames": self.joint_names,
            "jointParents": self.joint_parents,
            "revision": self.revision,
        }

    @staticmethod
    def _latent_parameters(
        name_prefix: str,
        label_prefix: str,
        values: np.ndarray,
        limits: np.ndarray,
    ) -> list[dict[str, object]]:
        return [
            {
                "index": index,
                "name": f"{name_prefix}_{index + 1:02d}",
                "label": f"{label_prefix} component {index + 1:02d}",
                "category": label_prefix,
                "min": float(limits[index, 0]),
                "max": float(limits[index, 1]),
                "value": float(value),
                "joints": [],
            }
            for index, value in enumerate(values)
        ]

    def configure(self, payload: dict[str, object]) -> GeometrySnapshot:
        requested_lod = payload.get("lod", self.lod)
        try:
            lod = int(requested_lod)
        except (TypeError, ValueError) as error:
            raise ValueError("LOD must be an integer from 0 through 6.") from error
        if lod not in SUPPORTED_LODS:
            raise ValueError("LOD must be an integer from 0 through 6.")

        apply_correctives = payload.get("applyCorrectives", self.apply_correctives)
        if not isinstance(apply_correctives, bool):
            raise ValueError("applyCorrectives must be true or false.")

        snap_to_ground = payload.get("snapToGround", self.snap_to_ground)
        if not isinstance(snap_to_ground, bool):
            raise ValueError("snapToGround must be true or false.")

        with self.lock:
            changed = (
                lod != self.lod
                or apply_correctives != self.apply_correctives
                or snap_to_ground != self.snap_to_ground
            )
            if snap_to_ground != self.snap_to_ground:
                if snap_to_ground:
                    self.unsnapped_root_ty = float(self.pose_values[self.root_ty_index])
                else:
                    self.pose_values[self.root_ty_index] = self.unsnapped_root_ty
            self.lod = lod
            self.faces = self.lod_faces[lod]
            self.apply_correctives = apply_correctives
            self.snap_to_ground = snap_to_ground
            if changed:
                self.snapshot = self._infer()
            return self.snapshot

    def update(self, payload: dict[str, object]) -> GeometrySnapshot:
        group = str(payload.get("group", "pose"))
        action = str(payload.get("action", "set"))
        index_value = payload.get("index", 0)

        try:
            index = int(index_value)
        except (TypeError, ValueError) as error:
            raise ValueError("Parameter index must be an integer.") from error

        with self.lock:
            values, limits = self._group_state(group)

            if action in {"set", "reset_current"}:
                if index < 0 or index >= len(values):
                    raise ValueError(f"Parameter index {index} is out of range.")
                if (
                    group == "pose"
                    and index == self.root_ty_index
                    and self.snap_to_ground
                ):
                    raise ValueError(
                        "root_ty is managed while Snap to ground is enabled."
                    )
                requested = 0.0
                if action == "set":
                    try:
                        requested = float(payload.get("value", 0.0))
                    except (TypeError, ValueError) as error:
                        raise ValueError("Parameter value must be numeric.") from error
                values[index] = np.clip(requested, *limits[index])
                if group == "pose" and index == self.root_ty_index:
                    self.unsnapped_root_ty = float(values[index])
            elif action == "reset_group":
                values.fill(0.0)
                if group == "pose":
                    self.unsnapped_root_ty = 0.0
            elif action == "reset_all":
                self.pose_values.fill(0.0)
                self.identity_values.fill(0.0)
                self.expression_values.fill(0.0)
                self.unsnapped_root_ty = 0.0
            else:
                raise ValueError(f"Unknown action: {action}")

            self.snapshot = self._infer()
            return self.snapshot

    def _group_state(self, group: str) -> tuple[np.ndarray, np.ndarray]:
        if group == "pose":
            return self.pose_values, self.pose_limits
        if group == "identity":
            return self.identity_values, self.identity_limits
        if group == "expression":
            return self.expression_values, self.expression_limits
        raise ValueError(f"Unknown parameter group: {group}")


def _geometry_response(snapshot: GeometrySnapshot) -> Response:
    response = Response(snapshot.to_bytes(), mimetype="application/octet-stream")
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-MHR-Revision"] = str(snapshot.revision)
    response.headers["X-MHR-Inference-Ms"] = f"{snapshot.inference_ms:.3f}"
    response.headers["X-MHR-Root-TY"] = f"{snapshot.root_ty:.8f}"
    return response


def _json_request() -> dict[str, object]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON request body.")
    return payload


def create_app(model_path: Path = DEFAULT_MODEL_PATH) -> Flask:
    engine = MHREngine(model_path.resolve())
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index() -> Response:
        return send_from_directory(STATIC_ROOT, "index.html")

    @app.get("/static/<path:filename>")
    def static_file(filename: str) -> Response:
        return send_from_directory(STATIC_ROOT, filename)

    @app.get("/vendor/three/<path:filename>")
    def three_vendor(filename: str) -> Response:
        if not THREE_ROOT.is_dir():
            return (
                jsonify(
                    {
                        "error": "Three.js is not installed. Run npm install in the viewer folder."
                    }
                ),
                503,
            )
        return send_from_directory(THREE_ROOT, filename)

    @app.get("/api/metadata")
    def metadata() -> Response:
        return jsonify(engine.metadata())

    @app.get("/api/topology")
    def topology() -> Response:
        with engine.lock:
            payload = engine.faces.astype("<u4", copy=False).tobytes()
        response = Response(payload, mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/geometry")
    def geometry() -> Response:
        return _geometry_response(engine.snapshot)

    @app.post("/api/deform")
    def deform() -> tuple[Response, int] | Response:
        try:
            snapshot = engine.update(_json_request())
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        return _geometry_response(snapshot)

    @app.post("/api/configure")
    def configure() -> tuple[Response, int] | Response:
        try:
            engine.configure(_json_request())
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(engine.metadata())

    @app.get("/api/export/<file_format>")
    def export_current_mesh(file_format: str) -> tuple[Response, int] | Response:
        if "ground" in request.args:
            return (
                jsonify(
                    {
                        "error": (
                            "The ground query parameter has been removed. Configure "
                            "snapToGround before exporting."
                        )
                    }
                ),
                400,
            )

        try:
            normalized_format = normalize_export_format(file_format)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        with engine.lock:
            vertices = engine.snapshot.vertices.copy()
            faces = engine.faces.copy()
            lod = engine.lod
            revision = engine.revision
            snap_to_ground = engine.snap_to_ground
            rig = None
            if normalized_format in RIGGED_EXPORT_FORMATS:
                joint_indices, joint_weights = engine._skinning_for_lod(lod)
                rig = RigData(
                    joint_names=tuple(engine.joint_names),
                    joint_parents=np.asarray(
                        engine.joint_parents,
                        dtype=np.int32,
                    ),
                    skeleton_state=engine.snapshot.skeleton_state.copy(),
                    joint_indices=joint_indices.copy(),
                    joint_weights=joint_weights.copy(),
                )
        try:
            result = export_mesh(vertices, faces, normalized_format, rig)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except RuntimeError as error:
            return jsonify({"error": str(error)}), 503

        ground_suffix = "-grounded" if snap_to_ground else ""
        filename = f"mhr-lod{lod}-r{revision}{ground_suffix}.{result.extension}"
        response = Response(result.data, mimetype=result.mimetype)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        response.headers["X-MHR-LOD"] = str(lod)
        response.headers["X-MHR-Revision"] = str(revision)
        response.headers["X-MHR-Snap-To-Ground"] = str(snap_to_ground).lower()
        return response

    @app.get("/health")
    def health() -> Response:
        return jsonify(
            {
                "status": "ok",
                "model": str(engine.model_path),
                "vertices": int(engine.snapshot.vertices.shape[0]),
                "faces": int(engine.faces.shape[0]),
                "joints": int(engine.snapshot.joints.shape[0]),
                "lod": engine.lod,
                "applyCorrectives": engine.apply_correctives,
                "snapToGround": engine.snap_to_ground,
                "revision": engine.revision,
            }
        )

    app.config["MHR_ENGINE"] = engine
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to mhr_model.pt (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.model)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"MHR Three.js Viewer: {url}")
    print(f"Model: {args.model.resolve()}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
