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

"""Tests for the staged PyMomentum fitting of the SMPL-MHR conversion tool."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# The tool needs pymomentum, which is not a dependency of the mhr package.
pytest.importorskip("pymomentum.solver")

_TOOL_DIR = Path(__file__).resolve().parents[1] / "tools" / "mhr_smpl_conversion"
sys.path.insert(0, str(_TOOL_DIR))
try:
    _spec = importlib.util.spec_from_file_location(
        "mhr_pymomentum_fitting", _TOOL_DIR / "pymomentum_fitting.py"
    )
    pymomentum_fitting = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(pymomentum_fitting)
finally:
    sys.path.remove(str(_TOOL_DIR))

_NUM_RIG = pymomentum_fitting._NUM_RIG_PARAMETERS
_NUM_IDENTITY = 45
_NUM_EXPRESSION = 72
_NUM_JOINTS = 6
_NUM_VERTICES = 40
_PARTS = ["spine", "neck", "head", "clavicle", "knee", "index", "thumb", "scale_leg"]


def _fake_character(numpy_parameter_transform: bool = False) -> SimpleNamespace:
    """A stand-in for the MHR character with the attributes the stage setup reads.

    It has the MHR parameter layout (204 rig parameters, 45 identity and 72
    face expression blendshapes) on a small skeleton and mesh. Recent
    PyMomentum releases return the parameter transform as NumPy arrays, older
    ones as torch tensors; numpy_parameter_transform selects the former.
    """

    rng = np.random.default_rng(0)
    rig_names = ["root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"]
    rig_names += [f"{_PARTS[i % len(_PARTS)]}{i}_rx" for i in range(_NUM_RIG - 6)]
    names = rig_names + [f"blend_{i}" for i in range(_NUM_IDENTITY + _NUM_EXPRESSION)]

    transform = torch.zeros(_NUM_JOINTS * 7, len(names))
    for i in range(_NUM_RIG):
        transform[rng.integers(_NUM_JOINTS * 7), i] = 1.0
    rigid = torch.zeros(len(names), dtype=torch.bool)
    rigid[:6] = True
    if numpy_parameter_transform:
        transform, rigid = transform.numpy(), rigid.numpy()

    index = rng.integers(1, _NUM_JOINTS, size=(_NUM_VERTICES, 2))
    weight = np.full((_NUM_VERTICES, 2), 0.5)
    return SimpleNamespace(
        parameter_transform=SimpleNamespace(
            names=names, transform=transform, rigid_parameters=rigid
        ),
        skin_weights=SimpleNamespace(index=index, weight=weight),
        mesh=SimpleNamespace(vertices=rng.random((_NUM_VERTICES, 3))),
    )


def _fitter(recorded: list, numpy_parameter_transform: bool = False):
    """A fitter on the fake character whose solver records each stage's active
    parameters instead of solving."""

    fitter = object.__new__(pymomentum_fitting.PyMomentumModelFitting)
    character = _fake_character(numpy_parameter_transform)
    num_parameters = len(character.parameter_transform.names)
    fitter._mhr_model = SimpleNamespace(character=character)
    fitter._bs_character = character
    fitter._num_blendshapes = _NUM_IDENTITY
    fitter._num_expression_blendshapes = _NUM_EXPRESSION
    fitter._num_vertices = _NUM_VERTICES
    fitter._num_parameters = num_parameters
    fitter._solved_parameters = torch.zeros(num_parameters, dtype=torch.float64)
    fitter._constant_parameter_mask = torch.zeros(num_parameters, dtype=torch.bool)
    fitter._subsampled_vertex_mask = torch.ones(_NUM_VERTICES, dtype=torch.bool)
    fitter._head_mask = torch.ones(_NUM_VERTICES, dtype=torch.float64)
    fitter._hand_mask = torch.ones(_NUM_VERTICES, dtype=torch.float64)
    fitter._body_mask = torch.ones(_NUM_VERTICES, dtype=torch.float64)
    fitter._optimize_one_stage = lambda target, stage: recorded.append(
        (stage.info, stage.active_parameter_mask.clone())
    )
    return fitter


def _identity_mask(fitter) -> torch.Tensor:
    """Scale parameters and identity blendshapes, as the single-identity mode
    of the conversion holds constant."""

    names = fitter._bs_character.parameter_transform.names
    mask = torch.tensor(["scale" in name for name in names])
    mask[_NUM_RIG : _NUM_RIG + _NUM_IDENTITY] = True
    return mask


_TARGET = torch.zeros(_NUM_VERTICES, 3)


@pytest.mark.parametrize("exclude_expression", [False, True])
def test_repeated_fit_matches_fresh_fit(exclude_expression):
    # Fit once with the identity held constant, then again without constants.
    reused_stages = []
    reused = _fitter(reused_stages)
    mask = _identity_mask(reused)
    reused.reset()
    reused.set_constant_parameters(mask, torch.zeros(int(mask.sum())))
    reused.fit(_TARGET, exclude_expression=exclude_expression)
    first_fit = list(reused_stages)
    reused_stages.clear()
    reused.reset()
    reused.fit(_TARGET, exclude_expression=exclude_expression)

    fresh_stages = []
    fresh = _fitter(fresh_stages)
    fresh.reset()
    fresh.fit(_TARGET, exclude_expression=exclude_expression)

    assert [info for info, _ in reused_stages] == [info for info, _ in fresh_stages]
    for (info, reused_mask), (_, fresh_mask) in zip(reused_stages, fresh_stages):
        assert torch.equal(reused_mask, fresh_mask), info

    # The constant parameters were still excluded while they were constant.
    face_stage = next(m for info, m in first_fit if info.startswith("Stage 1.0"))
    assert not (face_stage & mask).any()
    fresh_face = next(m for info, m in fresh_stages if info.startswith("Stage 1.0"))
    assert (fresh_face & mask).any()


def test_fit_after_tracking_fit_matches_fresh_fit():
    # A tracking fit continues from the previous frame without reset(), so
    # the face parameters that stage 2 holds constant are still constant.
    reused_stages = []
    reused = _fitter(reused_stages)
    reused.reset()
    reused.fit(_TARGET)
    reused.fit(_TARGET, skip_global_stages=True)
    reused_stages.clear()
    reused.reset()
    reused.fit(_TARGET)

    fresh_stages = []
    fresh = _fitter(fresh_stages)
    fresh.reset()
    fresh.fit(_TARGET)

    assert len(reused_stages) == len(fresh_stages)
    for (info, reused_mask), (_, fresh_mask) in zip(reused_stages, fresh_stages):
        assert torch.equal(reused_mask, fresh_mask), info


def test_fit_leaves_stage_masks_unchanged():
    fitter = _fitter([])
    stages = fitter._create_hierarchical_masks(exclude_expression=False)
    before = [stage.active_parameter_mask.clone() for stage in stages]
    mask = _identity_mask(fitter)
    fitter.set_constant_parameters(mask, torch.zeros(int(mask.sum())))

    fitter.fit(_TARGET, exclude_expression=False)

    # fit() reads the same cached stages (same arguments, same cache entry).
    assert fitter._create_hierarchical_masks(exclude_expression=False) is stages
    for stage, mask_before in zip(stages, before):
        assert torch.equal(stage.active_parameter_mask, mask_before), stage.info


def test_numpy_parameter_transform_gives_the_same_stages():
    torch_stages, numpy_stages = [], []
    _fitter(torch_stages).fit(_TARGET)
    _fitter(numpy_stages, numpy_parameter_transform=True).fit(_TARGET)

    assert [info for info, _ in numpy_stages] == [info for info, _ in torch_stages]
    for (info, numpy_mask), (_, torch_mask) in zip(numpy_stages, torch_stages):
        assert torch.equal(numpy_mask, torch_mask), info
