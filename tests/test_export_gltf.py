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

"""Tests for the glTF export tool."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

# The tool needs pymomentum, which is not a dependency of the mhr package.
pym_geometry = pytest.importorskip("pymomentum.geometry")
torch_character = pytest.importorskip("pymomentum.torch.character")

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "mhr_export" / "export_gltf.py"
)
_spec = importlib.util.spec_from_file_location("mhr_export_gltf", _MODULE_PATH)
export_gltf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_gltf)

_NUM_IDENTITY = 3
_NUM_FACE = 2
_NUM_FRAMES = 4


def _character_with_blend_shapes() -> pym_geometry.Character:
    """Synthetic test character with identity and face blendshapes appended
    to its model parameters, like ``export_gltf.load_character`` returns."""

    character = pym_geometry.create_test_character()
    rng = np.random.default_rng(0)
    n_verts = character.mesh.n_vertices
    # The mesh is the blendshape base, as in the MHR assets.
    blend_shape = pym_geometry.BlendShape.from_tensors(
        np.asarray(character.mesh.vertices, dtype=np.float32),
        rng.random((_NUM_IDENTITY + _NUM_FACE, n_verts, 3)).astype(np.float32),
    )
    return character.with_blend_shape(blend_shape)


def _posed_vertices(
    character: pym_geometry.Character, motion: np.ndarray
) -> np.ndarray:
    """Skinned vertices of every frame, blendshapes included."""

    rig = torch_character.Character(character)
    model_parameters = torch.from_numpy(motion)
    num_blend = character.blend_shape.shape_vectors.shape[0]
    rest = rig.blend_shape.forward(model_parameters[:, -num_blend:])
    skel_state = rig.joint_parameters_to_skeleton_state(
        rig.model_parameters_to_joint_parameters(model_parameters)
    )
    return rig.skin_points(skel_state=skel_state, rest_vertex_positions=rest).numpy()


def _inputs(character: pym_geometry.Character):
    rng = np.random.default_rng(1)
    num_model = character.parameter_transform.size - _NUM_IDENTITY - _NUM_FACE
    model_parameters = rng.normal(scale=0.3, size=(_NUM_FRAMES, num_model))
    identity_coeffs = rng.normal(size=_NUM_IDENTITY)
    face_expr_coeffs = rng.normal(size=(_NUM_FRAMES, _NUM_FACE))
    return model_parameters, identity_coeffs, face_expr_coeffs


def test_build_motion_order_and_broadcast():
    character = _character_with_blend_shapes()
    model_parameters, identity_coeffs, face_expr_coeffs = _inputs(character)

    motion = export_gltf.build_motion(
        character,
        model_parameters,
        identity_coeffs,
        face_expr_coeffs,
        num_identity=_NUM_IDENTITY,
    )

    num_model = model_parameters.shape[1]
    assert motion.shape == (_NUM_FRAMES, character.parameter_transform.size)
    assert motion.dtype == np.float32
    np.testing.assert_allclose(motion[:, :num_model], model_parameters, rtol=1e-6)
    np.testing.assert_allclose(
        motion[:, num_model : num_model + _NUM_IDENTITY],
        np.tile(identity_coeffs, (_NUM_FRAMES, 1)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        motion[:, num_model + _NUM_IDENTITY :], face_expr_coeffs, rtol=1e-6
    )

    no_face = export_gltf.build_motion(
        character, model_parameters, identity_coeffs, num_identity=_NUM_IDENTITY
    )
    assert not no_face[:, num_model + _NUM_IDENTITY :].any()


@pytest.mark.parametrize(
    "bad",
    [
        {"model_parameters": np.zeros((_NUM_FRAMES, 1))},
        {"identity_coeffs": np.zeros((2, _NUM_IDENTITY))},
        {"identity_coeffs": np.zeros(_NUM_IDENTITY - 1), "face_expr_coeffs": None},
        {"face_expr_coeffs": np.zeros((_NUM_FRAMES, _NUM_FACE + 1))},
    ],
)
def test_build_motion_rejects_wrong_shapes(bad):
    character = _character_with_blend_shapes()
    model_parameters, identity_coeffs, face_expr_coeffs = _inputs(character)
    kwargs = {
        "model_parameters": model_parameters,
        "identity_coeffs": identity_coeffs,
        "face_expr_coeffs": face_expr_coeffs,
    }
    kwargs.update(bad)
    with pytest.raises(ValueError):
        export_gltf.build_motion(character, **kwargs, num_identity=_NUM_IDENTITY)


def test_build_motion_rejects_zero_frames():
    character = _character_with_blend_shapes()
    model_parameters, identity_coeffs, face_expr_coeffs = _inputs(character)
    with pytest.raises(ValueError, match="no frames"):
        export_gltf.build_motion(
            character,
            model_parameters[:0],
            identity_coeffs,
            face_expr_coeffs[:0],
            num_identity=_NUM_IDENTITY,
        )


def test_gltf_round_trip(tmp_path):
    character = _character_with_blend_shapes()
    motion = export_gltf.build_motion(
        character, *_inputs(character), num_identity=_NUM_IDENTITY
    )
    output = tmp_path / "motion.glb"

    export_gltf.save_gltf(character, output, motion, fps=24.0)
    loaded, loaded_motion, _, fps = pym_geometry.Character.load_gltf_with_motion(
        str(output)
    )

    assert fps == 24.0
    np.testing.assert_allclose(loaded_motion, motion, atol=1e-6)
    assert loaded.parameter_transform.names == character.parameter_transform.names
    assert loaded.skeleton.joint_names == character.skeleton.joint_names
    np.testing.assert_allclose(
        np.asarray(loaded.skeleton.offsets),
        np.asarray(character.skeleton.offsets),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        loaded.blend_shape.shape_vectors,
        character.blend_shape.shape_vectors,
        atol=1e-5,
    )
    # Skinning weights and blendshapes survive: every frame poses the same mesh.
    np.testing.assert_allclose(
        _posed_vertices(loaded, loaded_motion),
        _posed_vertices(character, motion),
        atol=1e-4,
    )


def test_cli_reports_missing_assets(tmp_path):
    with pytest.raises(FileNotFoundError, match="lod1.fbx"):
        export_gltf.main(
            ["--assets", str(tmp_path), "--output", str(tmp_path / "a.glb")]
        )
