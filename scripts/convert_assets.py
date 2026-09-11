#!/usr/bin/env python3
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

"""Convert legacy MHR FBX assets into the portable tensor format."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile

from pathlib import Path
from typing import Iterable

import numpy as np

from mhr._torch_rig import RIG_SCHEMA_VERSION
from mhr.mhr import (
    NUM_FACE_EXPRESSION_BLENDSHAPES,
    NUM_IDENTITY_BLENDSHAPES,
    set_blendshape_parameter_sets,
)


LODS = tuple(range(7))


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a reproducible compressed NPZ archive."""

    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=6)


def _write_bundle(path: Path, files: Iterable[Path]) -> None:
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for source in sorted(files, key=lambda item: item.name):
            info = zipfile.ZipInfo(source.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with source.open("rb") as src, archive.open(info, "w") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _character(assets: Path, lod: int):
    import pymomentum.geometry as geometry
    from pymomentum.torch.character import Character

    character = geometry.Character.load_fbx(
        str(assets / f"lod{lod}.fbx"),
        str(assets / "compact_v6_1.model"),
        load_blendshapes=True,
    )
    expected_shapes = NUM_IDENTITY_BLENDSHAPES + NUM_FACE_EXPRESSION_BLENDSHAPES
    if character.blend_shape.shape_vectors.shape[0] != expected_shapes:
        raise ValueError(
            f"LOD {lod} contains {character.blend_shape.shape_vectors.shape[0]} "
            f"blend shapes; expected {expected_shapes}"
        )
    character = character.with_blend_shape(character.blend_shape)
    set_blendshape_parameter_sets(character)
    return character, Character(character)


def _rig_arrays(
    character: object,
    torch_character: object,
    activation_path: Path,
    momentum_version: str,
) -> dict[str, np.ndarray]:
    parameter_transform = torch_character.parameter_transform.parameter_transform
    active_joints = getattr(
        torch_character.parameter_transform,
        "active_joints",
        parameter_transform.ne(0).any(dim=1).reshape(-1, 7).any(dim=1),
    )
    arrays = {
        "schema_version": np.array(RIG_SCHEMA_VERSION, dtype=np.int64),
        "source_momentum_version": np.array(momentum_version),
        "parameter_transform": _as_numpy(parameter_transform),
        "pose_parameters": _as_numpy(
            torch_character.parameter_transform.pose_parameters
        ),
        "rigid_parameters": _as_numpy(
            torch_character.parameter_transform.rigid_parameters
        ),
        "scaling_parameters": _as_numpy(
            torch_character.parameter_transform.scaling_parameters
        ),
        "active_joints": _as_numpy(active_joints),
        "parameter_names": np.asarray(
            character.parameter_transform.names, dtype=np.str_
        ),
        "joint_offsets": _as_numpy(torch_character.skeleton.joint_translation_offsets),
        "joint_prerotations": _as_numpy(torch_character.skeleton.joint_prerotations),
        "joint_parents": _as_numpy(torch_character.skeleton.joint_parents),
        "joint_names": np.asarray(character.skeleton.joint_names, dtype=np.str_),
        "fk_indices": _as_numpy(torch_character.skeleton.pmi),
        "fk_sizes": np.asarray(
            torch_character.skeleton._pmi_buffer_sizes, dtype=np.int64
        ),
        "inverse_bind_pose": _as_numpy(
            torch_character.linear_blend_skinning.inverse_bind_pose
        ),
        "limit_min": _as_numpy(torch_character.parameter_limits.minmax_min),
        "limit_max": _as_numpy(torch_character.parameter_limits.minmax_max),
        "limit_weight": _as_numpy(torch_character.parameter_limits.minmax_weight),
        "limit_parameter_index": _as_numpy(
            torch_character.parameter_limits.minmax_parameter_index
        ),
    }
    with np.load(activation_path, allow_pickle=False) as activation:
        for name in activation.files:
            if name in arrays:
                raise ValueError(f"duplicate rig array name: {name}")
            arrays[name] = np.array(activation[name], copy=True)
    return arrays


def _lod_arrays(
    character: object,
    torch_character: object,
    corrective_path: Path,
    lod: int,
) -> dict[str, np.ndarray]:
    arrays = {
        "schema_version": np.array(RIG_SCHEMA_VERSION, dtype=np.int64),
        "lod": np.array(lod, dtype=np.int64),
        "base_shape": _as_numpy(torch_character.blend_shape.base_shape),
        "shape_vectors": _as_numpy(torch_character.blend_shape.shape_vectors),
        "faces": _as_numpy(torch_character.mesh.faces),
        "texcoords": _as_numpy(torch_character.mesh.texcoords),
        "texcoord_faces": _as_numpy(torch_character.mesh.texcoord_faces),
        "skin_indices": _as_numpy(character.skin_weights.index).astype(
            np.int32, copy=False
        ),
        "skin_weights": _as_numpy(character.skin_weights.weight).astype(
            np.float32, copy=False
        ),
    }
    with np.load(corrective_path, allow_pickle=False) as correctives:
        for name in correctives.files:
            if name in arrays:
                raise ValueError(f"duplicate LOD array name: {name}")
            arrays[name] = np.array(correctives[name], copy=True)
    return arrays


def _assert_shared_rig(
    reference: dict[str, np.ndarray], actual: dict[str, np.ndarray]
) -> None:
    if reference.keys() != actual.keys():
        raise ValueError("LOD rigs contain different shared fields")
    for name in reference:
        if not np.array_equal(reference[name], actual[name]):
            raise ValueError(f"shared rig field {name!r} differs between LODs")


def convert_assets(
    assets: Path,
    output: Path,
    lods: Iterable[int],
    momentum_version: str,
) -> Path:
    lods = tuple(sorted(set(lods)))
    if not lods or any(lod not in LODS for lod in lods):
        raise ValueError("LODs must be selected from 0 through 6")
    output.mkdir(parents=True, exist_ok=True)

    activation_path = assets / "corrective_activation.npz"
    license_path = assets / "LICENSE.txt"
    for required in (assets / "compact_v6_1.model", activation_path, license_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    rig_reference: dict[str, np.ndarray] | None = None
    lod_paths: dict[int, Path] = {}
    source_hashes: dict[str, str] = {}
    for lod in lods:
        fbx_path = assets / f"lod{lod}.fbx"
        corrective_path = assets / f"corrective_blendshapes_lod{lod}.npz"
        for required in (fbx_path, corrective_path):
            if not required.is_file():
                raise FileNotFoundError(required)

        character, torch_character = _character(assets, lod)
        rig = _rig_arrays(character, torch_character, activation_path, momentum_version)
        if rig_reference is None:
            rig_reference = rig
        else:
            _assert_shared_rig(rig_reference, rig)

        lod_path = output / f"lod{lod}.npz"
        _write_npz(
            lod_path,
            _lod_arrays(character, torch_character, corrective_path, lod),
        )
        lod_paths[lod] = lod_path
        source_hashes[fbx_path.name] = _sha256(fbx_path)
        source_hashes[corrective_path.name] = _sha256(corrective_path)

    assert rig_reference is not None
    rig_path = output / "rig.npz"
    _write_npz(rig_path, rig_reference)
    source_hashes["compact_v6_1.model"] = _sha256(assets / "compact_v6_1.model")
    source_hashes[activation_path.name] = _sha256(activation_path)

    common_bundle = output / f"mhr-assets-v{RIG_SCHEMA_VERSION}-common.zip"
    _write_bundle(common_bundle, (rig_path, license_path))
    bundles: dict[str, dict[str, object]] = {}
    for lod, lod_path in lod_paths.items():
        bundle = output / f"mhr-assets-v{RIG_SCHEMA_VERSION}-lod{lod}.zip"
        _write_bundle(bundle, (lod_path, license_path))
        bundles[str(lod)] = {
            "filename": bundle.name,
            "sha256": _sha256(bundle),
            "size": bundle.stat().st_size,
        }

    manifest = {
        "schema_version": RIG_SCHEMA_VERSION,
        "common": {
            "filename": common_bundle.name,
            "sha256": _sha256(common_bundle),
            "size": common_bundle.stat().st_size,
        },
        "lods": bundles,
        "source_sha256": dict(sorted(source_hashes.items())),
    }
    manifest_path = output / f"mhr-assets-v{RIG_SCHEMA_VERSION}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lod", action="append", type=int, choices=LODS)
    parser.add_argument("--momentum-version", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = convert_assets(
        args.assets,
        args.output,
        args.lod if args.lod is not None else LODS,
        args.momentum_version,
    )
    print(manifest)


if __name__ == "__main__":
    main()
