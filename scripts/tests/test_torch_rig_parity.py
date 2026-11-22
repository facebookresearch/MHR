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

"""Parity of the pure-PyTorch rig with the PyMomentum rig on the real assets.

Set MHR_TEST_ASSETS to the folder of the release assets. The test converts the
assets, runs MHR once with PyMomentum's torch character and once with
TorchCharacter, and compares the vertices, the skeleton state and the input
gradients on every LOD.
"""

import copy
import gc
import os

from pathlib import Path

import pytest
import torch

pytest.importorskip("pymomentum.geometry")

from mhr._rig_schema import LODS
from mhr._torch_rig import TorchCharacter
from mhr.mhr import MHR
from scripts.convert_assets import convert_assets

ASSETS = os.environ.get("MHR_TEST_ASSETS")

# Both rigs evaluate in float32, and their operations differ in order, so the
# results agree to float32 rounding, not bit for bit. Vertices are in cm.
VERTEX_ATOL = 1e-3
# Gradients reach about 1e2, so compare them relative to their largest entry.
GRADIENT_RTOL = 1e-5


def _with_rig(model: MHR, rig: TorchCharacter) -> MHR:
    """A shallow copy of model that evaluates with rig."""

    converted = copy.copy(model)
    converted._modules = dict(model._modules)
    converted.character_torch = rig
    return converted


def _inputs(seed: int, pose_scale: float, batch_size: int = 4) -> list[torch.Tensor]:
    """Random inputs at the demo's scale; pose_scale scales the pose
    parameters (1: up to 0.1 rad, 30: up to 3 rad)."""

    generator = torch.Generator().manual_seed(seed)
    return [
        0.8 * torch.randn(batch_size, 45, generator=generator),
        pose_scale * 0.2 * (torch.rand(batch_size, 204, generator=generator) - 0.5),
        0.3 * torch.randn(batch_size, 72, generator=generator),
    ]


@pytest.mark.skipif(ASSETS is None, reason="set MHR_TEST_ASSETS to the asset folder")
def test_torch_rig_matches_pymomentum_on_every_lod(tmp_path: Path) -> None:
    assets = Path(ASSETS)
    convert_assets(assets, tmp_path, LODS, "test")

    for lod in LODS:
        legacy = MHR.from_files(assets, device=torch.device("cpu"), lod=lod)
        rig = TorchCharacter.from_files(
            tmp_path / "rig.npz", tmp_path / f"lod{lod}.npz", lod
        )
        converted = _with_rig(legacy, rig)
        assert set(rig.state_dict()) >= {
            "blend_shape.base_shape",
            "blend_shape.shape_vectors",
            "skeleton.joint_translation_offsets",
            "linear_blend_skinning.inverse_bind_pose",
        }

        for pose_scale in (1.0, 30.0):
            values = _inputs(seed=lod, pose_scale=pose_scale)
            legacy_inputs = [value.clone().requires_grad_() for value in values]
            converted_inputs = [value.clone().requires_grad_() for value in values]
            legacy_vertices, legacy_skeleton = legacy(*legacy_inputs)
            weights = torch.randn_like(legacy_vertices)
            (legacy_vertices * weights).sum().backward()
            converted_vertices, converted_skeleton = converted(*converted_inputs)
            (converted_vertices * weights).sum().backward()

            torch.testing.assert_close(
                converted_vertices, legacy_vertices, atol=VERTEX_ATOL, rtol=0
            )
            torch.testing.assert_close(
                converted_skeleton, legacy_skeleton, atol=VERTEX_ATOL, rtol=0
            )

            for converted_input, legacy_input in zip(converted_inputs, legacy_inputs):
                scale = legacy_input.grad.abs().max()
                torch.testing.assert_close(
                    converted_input.grad,
                    legacy_input.grad,
                    atol=GRADIENT_RTOL * scale,
                    rtol=0,
                )

        del legacy, rig, converted, legacy_vertices, converted_vertices
        gc.collect()
