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

"""Export MHR parameters to a glTF binary (.glb) file.

The file holds the skinned MHR mesh, its skeleton, the identity and face
expression blendshapes as morph targets, and one animation frame per row of
the input parameters. It can be opened in Blender and other glTF viewers.

Pose correctives are not exported; see README.md.
"""

import argparse
from pathlib import Path

import numpy as np

from mhr._pymomentum_version import check_pymomentum_version

check_pymomentum_version()

import pymomentum.geometry as pym_geometry

from mhr.io import get_default_asset_folder, get_mhr_fbx_path, get_mhr_model_path

NUM_IDENTITY_BLENDSHAPES = 45


def load_character(folder: Path, lod: int) -> pym_geometry.Character:
    """Load the MHR rig of one LOD with its blendshapes as model parameters.

    The parameters of the returned character are ordered
    [model_parameters | identity_coeffs | face_expr_coeffs], as in ``mhr.MHR``.
    """

    fbx_path = Path(get_mhr_fbx_path(folder, lod))
    model_path = Path(get_mhr_model_path(folder))
    for path in (fbx_path, model_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; download the MHR assets first "
                "(see the main README)."
            )
    character = pym_geometry.Character.load_fbx(
        str(fbx_path), str(model_path), load_blendshapes=True
    )
    # Append one model parameter per blendshape (identity, then face expression).
    return character.with_blend_shape(character.blend_shape)


def build_motion(
    character: pym_geometry.Character,
    model_parameters: np.ndarray,
    identity_coeffs: np.ndarray,
    face_expr_coeffs: np.ndarray | None = None,
    num_identity: int = NUM_IDENTITY_BLENDSHAPES,
) -> np.ndarray:
    """Stack MHR inputs into the [frames, parameters] motion that glTF export takes.

    Shapes follow ``MHR.forward``: one identity row is repeated for every
    frame; face expression coefficients need one row per frame.

    :param character: Character from ``load_character``.
    :param model_parameters: [frames, n] rigid, pose and scale parameters.
    :param identity_coeffs: [k] or [1, k] for one identity across frames, or
        [frames, k], with ``k == num_identity``.
    :param face_expr_coeffs: [frames, m] face expression coefficients, or None
        for zeros. ``num_identity + m`` must equal the number of blendshapes.
    :param num_identity: Number of identity blendshapes; the rest of the
        character's blendshapes are face expressions.
    """

    model_parameters = np.atleast_2d(np.asarray(model_parameters, dtype=np.float32))
    num_frames = model_parameters.shape[0]
    if num_frames == 0:
        raise ValueError("model_parameters has no frames")
    num_blendshapes = character.blend_shape.shape_vectors.shape[0]
    num_model_parameters = character.parameter_transform.size - num_blendshapes
    if model_parameters.shape[1] != num_model_parameters:
        raise ValueError(
            f"model_parameters has {model_parameters.shape[1]} columns, "
            f"expected {num_model_parameters}"
        )

    identity_coeffs = np.atleast_2d(np.asarray(identity_coeffs, dtype=np.float32))
    if identity_coeffs.shape[1] != num_identity:
        raise ValueError(
            f"identity_coeffs has {identity_coeffs.shape[1]} columns, "
            f"expected {num_identity}"
        )
    if identity_coeffs.shape[0] == 1:
        identity_coeffs = np.repeat(identity_coeffs, num_frames, axis=0)
    if identity_coeffs.shape[0] != num_frames:
        raise ValueError(
            f"identity_coeffs has {identity_coeffs.shape[0]} rows, "
            f"expected 1 or {num_frames}"
        )

    num_face = num_blendshapes - num_identity
    if face_expr_coeffs is None:
        face_expr_coeffs = np.zeros((num_frames, num_face), dtype=np.float32)
    face_expr_coeffs = np.atleast_2d(np.asarray(face_expr_coeffs, dtype=np.float32))
    if face_expr_coeffs.shape != (num_frames, num_face):
        raise ValueError(
            f"face_expr_coeffs has shape {face_expr_coeffs.shape}, "
            f"expected {(num_frames, num_face)} "
            f"({num_blendshapes} blendshapes in total)"
        )

    return np.concatenate([model_parameters, identity_coeffs, face_expr_coeffs], axis=1)


def save_gltf(
    character: pym_geometry.Character,
    output_path: Path,
    motion: np.ndarray,
    fps: float = 30.0,
) -> None:
    """Write the character and its motion to a .glb or .gltf file."""

    pym_geometry.Character.save_gltf(
        path=str(output_path),
        character=character,
        motion=(character.parameter_transform.names, motion),
        fps=fps,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            ".npz file with model_parameters [frames, 204], identity_coeffs "
            "[45], [1, 45] or [frames, 45], and optionally face_expr_coeffs "
            "[frames, 72]. Without it, one frame of the mean shape in the rest "
            "pose is exported."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .glb file.")
    parser.add_argument(
        "--lod", type=int, default=1, choices=range(7), help="Mesh LOD."
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Animation frame rate.")
    parser.add_argument(
        "--assets",
        type=Path,
        default=get_default_asset_folder(),
        help="Folder with the MHR assets (lod<N>.fbx, compact_v6_1.model).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    character = load_character(args.assets, args.lod)
    num_blendshapes = character.blend_shape.shape_vectors.shape[0]
    num_model_parameters = character.parameter_transform.size - num_blendshapes

    if args.input is None:
        model_parameters = np.zeros((1, num_model_parameters), dtype=np.float32)
        identity_coeffs = np.zeros(NUM_IDENTITY_BLENDSHAPES, dtype=np.float32)
        face_expr_coeffs = None
    else:
        with np.load(args.input) as data:
            model_parameters = data["model_parameters"]
            identity_coeffs = data["identity_coeffs"]
            face_expr_coeffs = (
                data["face_expr_coeffs"] if "face_expr_coeffs" in data else None
            )

    motion = build_motion(
        character, model_parameters, identity_coeffs, face_expr_coeffs
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_gltf(character, args.output, motion, fps=args.fps)
    print(f"Saved {motion.shape[0]} frame(s) at {args.fps:g} fps to {args.output}")


if __name__ == "__main__":
    main()
