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

"""Constant parameters in the PyMomentum fitting of the SMPL-MHR conversion tool."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# The tool needs pymomentum, which is not a dependency of the mhr package.
pytest.importorskip("pymomentum.solver")

_TOOL_DIR = Path(__file__).resolve().parents[1] / "tools" / "mhr_smpl_conversion"
sys.path.insert(0, str(_TOOL_DIR))
try:
    _spec = importlib.util.spec_from_file_location(
        "mhr_smpl_pymomentum_fitting", _TOOL_DIR / "pymomentum_fitting.py"
    )
    pymomentum_fitting = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(pymomentum_fitting)
finally:
    sys.path.remove(str(_TOOL_DIR))

_NUM_PARAMETERS = 8


def _mask(*indices: int) -> torch.Tensor:
    mask = torch.zeros(_NUM_PARAMETERS, dtype=torch.bool)
    mask[list(indices)] = True
    return mask


def _fitter(stages):
    """A fitter that runs the given stages; its solver adds 1 to every active
    parameter."""

    fitter = object.__new__(pymomentum_fitting.PyMomentumModelFitting)
    fitter._solved_parameters = torch.zeros(_NUM_PARAMETERS, dtype=torch.float64)
    fitter._constant_parameter_mask = torch.zeros(_NUM_PARAMETERS, dtype=torch.bool)
    fitter._create_hierarchical_masks = lambda exclude_expression: stages

    def solve(target, stage):
        fitter._solved_parameters = (
            fitter._solved_parameters + stage.active_parameter_mask.double()
        )

    fitter._optimize_one_stage = solve
    return fitter


def _stage(info, constant_parameter_mask=None):
    return pymomentum_fitting.PyMomentumOptimizationStage(
        active_parameter_mask=torch.ones(_NUM_PARAMETERS, dtype=torch.bool),
        vertex_cons_weights=torch.ones(1),
        info=info,
        constant_parameter_mask=constant_parameter_mask,
    )


def test_fit_keeps_parameters_the_caller_holds_constant():
    # The same layout as the real stages: a stage that holds its own
    # parameters constant (the face, in the body stages) between free stages.
    stage_constant = _mask(4, 5)
    fitter = _fitter(
        [_stage("first"), _stage("holds face", stage_constant), _stage("last")]
    )
    # The caller holds an identity constant, as single-identity conversion does.
    held = _mask(1, 2)
    held_values = torch.tensor([0.5, -0.25], dtype=torch.float64)
    fitter.set_constant_parameters(held, held_values)

    fitter.fit(torch.zeros(1, 3))

    solved = fitter._solved_parameters
    assert torch.equal(solved[held], held_values)
    # The stage's own constants change only in the stage before them.
    assert torch.equal(solved[stage_constant], torch.ones(2, dtype=torch.float64))
    free = ~(held | stage_constant)
    assert torch.equal(solved[free], torch.full((4,), 3.0, dtype=torch.float64))
