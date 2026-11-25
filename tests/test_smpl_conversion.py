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

"""Default (not fast) fitting paths of the SMPL-MHR conversion tool.

The tool tests use the small SMPL stand-in of test_smpl_fast_fitting.py. The
MHR tracking test runs when MHR_TEST_ASSETS names the release asset folder.
"""

import os
import types

import numpy as np
import pytest
import torch

from tests.test_smpl_fast_fitting import (
    _fitter,
    _load_tool_module,
    _NUM_VERTICES,
    _ToyBodyModel,
    conversion,
)

ASSETS = os.environ.get("MHR_TEST_ASSETS")
_NUM_VERTICES_SMPL = 6890

utils = _load_tool_module("utils")


def _converter(smpl_model) -> object:
    converter = object.__new__(conversion.Conversion)
    converter.__dict__.update(
        _smpl_model=smpl_model,
        smpl_model_type="smpl",
        _DEVICE="cpu",
        _batch_size=256,
    )
    return converter


@pytest.mark.skipif(ASSETS is None, reason="set MHR_TEST_ASSETS to the asset folder")
def test_mhr_tracking_fits_the_head_of_every_frame(monkeypatch):
    from mhr.mhr import MHR

    model = MHR.from_files(ASSETS, device=torch.device("cpu"), lod=1)
    smpl_model = types.SimpleNamespace(
        v_template=torch.zeros(_NUM_VERTICES_SMPL, 3), to=lambda device: smpl_model
    )
    converter = conversion.Conversion(model, smpl_model, method="pytorch", batch_size=4)
    converter._s2m_load_masks()

    # One iteration per stage keeps the test fast.
    optimize_mhr = conversion.PyTorchMHRFitting._optimize_mhr

    def optimize_mhr_once(self, *args, **kwargs):
        kwargs["stage_iterations"] = [[1, 1], [1, 1, 1], [1]]
        return optimize_mhr(self, *args, **kwargs)

    monkeypatch.setattr(
        conversion.PyTorchMHRFitting, "_optimize_mhr", optimize_mhr_once
    )
    # More frames than the batch size, so the frames are tracked.
    num_frames = 9
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        vertices, _ = model(
            torch.randn(num_frames, 45, generator=generator),
            0.05 * torch.randn(num_frames, 204, generator=generator),
            torch.rand(num_frames, 72, generator=generator),
        )

    parameters = converter._s2m_fit_mhr_using_pytorch(
        vertices, single_identity=False, is_tracking=True
    )

    face_expression = parameters["face_expr_coeffs"]
    head_identity = parameters["identity_coeffs"][:, 20:40]
    assert face_expression.shape[0] == num_frames
    assert (face_expression.abs().amax(dim=1) > 0).all()
    assert (head_identity.abs().amax(dim=1) > 0).all()


def test_one_betas_row_is_shared_by_every_frame(monkeypatch):
    num_frames = 4

    def smpl_model(transl, **unused):
        # Each output frame carries the id of the input frame it came from.
        return types.SimpleNamespace(
            vertices=transl[:, None, :].expand(-1, _NUM_VERTICES_SMPL, 3)
        )

    smpl_model.faces = np.zeros((1, 3), dtype=np.int64)
    parameters = {
        "betas": torch.zeros(1, 10),
        "body_pose": torch.zeros(num_frames, 69),
        "global_orient": torch.zeros(num_frames, 3),
        "transl": torch.arange(num_frames, dtype=torch.float32)[:, None].expand(-1, 3),
    }
    converter = _converter(smpl_model)
    converter._method = conversion.FittingMethod.PYMOMENTUM
    recorded = {}

    def fit(target_vertices, *args, **kwargs):
        recorded["target_vertices"] = target_vertices
        return {}

    def reprocess(fitting_parameter_results, *args, smpl_parameters, **kwargs):
        recorded["smpl_parameters"] = smpl_parameters
        return fitting_parameter_results, np.zeros(num_frames)

    monkeypatch.setattr(converter, "_compute_target_vertices", lambda v, d: v)
    monkeypatch.setattr(converter, "_s2m_fit_mhr_using_pymomentum", fit)
    monkeypatch.setattr(converter, "_s2m_reprocess_failure_cases", reprocess)

    converter.convert_smpl2mhr(smpl_parameters=parameters)

    assert recorded["target_vertices"][:, 0, 0].tolist() == list(range(num_frames))
    assert utils._get_num_frames(parameters, "smpl") == num_frames
    # Failure cases are reprocessed with the parameters of their own frame.
    for name, value in recorded["smpl_parameters"].items():
        assert value.shape[0] == num_frames, name


def test_smpl_body_shape_is_weighted_by_the_errors_of_the_selected_frames(
    monkeypatch,
):
    model = _ToyBodyModel()
    fitter = _fitter(model)
    # The frames whose scale is closest to the template's are selected first.
    scales = torch.tensor([1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1.25, 1.35])
    target = scales[:, None, None] * model.v_template

    def optimize_smpl(target_vertices, single_identity, known_parameters=None):
        rows = target_vertices.shape[0]
        betas = (known_parameters or {}).get("betas")
        if betas is None:
            betas = 0.01 * torch.arange(rows * model.num_betas).reshape(rows, -1)
        return {
            "betas": betas.float(),
            "body_pose": torch.zeros(rows, 69),
            "global_orient": torch.zeros(rows, 3),
            "transl": torch.zeros(rows, 3),
        }

    selected = []
    scored = []
    select_frames = fitter._select_frames_for_identity_estimation

    def record_selection(target_vertices):
        selected.append(select_frames(target_vertices))
        return selected[-1]

    def record_scored_target(results, target_vertices):
        scored.append(target_vertices.clone())
        return np.linspace(1.0, 2.0, len(results["betas"]))

    monkeypatch.setattr(fitter, "_optimize_smpl", optimize_smpl)
    monkeypatch.setattr(
        fitter, "_select_frames_for_identity_estimation", record_selection
    )
    monkeypatch.setattr(fitter, "_evaluate_conversion_error", record_scored_target)

    fitter.fit(target.clone(), single_identity=True, is_tracking=True)

    assert selected[0].tolist() != list(range(len(scales)))
    centers = 0.5 * (target.min(1)[0] + target.max(1)[0])
    centered = target - centers[:, None, :]
    torch.testing.assert_close(scored[0], centered[selected[0]])


@pytest.mark.parametrize("tool", ["pytorch_fitting", "conversion"])
def test_frame_selection_returns_frame_indices_of_long_sequences(tool):
    model = _ToyBodyModel()
    template = model.v_template
    # Above 256 * 16 frames, only every 16th frame is scored.
    num_frames = 4097
    generator = torch.Generator().manual_seed(0)
    target = template + torch.rand(num_frames, _NUM_VERTICES, 3, generator=generator)
    best_frame = 4096
    target[best_frame] = template

    if tool == "pytorch_fitting":
        selected = _fitter(model)._select_frames_for_identity_estimation(target)
    else:
        selected = _converter(model)._select_frames_for_identity_estimation(
            target, template.numpy(), model.faces
        )

    assert selected[0] == best_frame
    assert np.all(selected % 16 == 0)
    assert np.all(selected < num_frames)
