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

"""Fast MHR-to-SMPL fitting (fast=True) of the SMPL-MHR conversion tool.

The tests use a small differentiable stand-in for the SMPL model, so they need
neither the licensed SMPL model files nor the smplx package.
"""

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

# The tool needs pymomentum and trimesh, which are not dependencies of the mhr
# package.
pytest.importorskip("pymomentum.solver")
pytest.importorskip("trimesh")

_TOOL_DIR = Path(__file__).resolve().parents[1] / "tools" / "mhr_smpl_conversion"


def _load_tool_module(name: str):
    """Import one module of the tool.

    smplx and tqdm are only needed for annotations and progress bars on the
    code paths tested here; stand-ins are used while importing if they are not
    installed, and removed again afterwards.
    """
    stubs = {}
    if importlib.util.find_spec("smplx") is None:
        smplx = types.ModuleType("smplx")
        smplx.SMPL = smplx.SMPLX = smplx.SMPLH = object
        stubs["smplx"] = smplx
    if importlib.util.find_spec("tqdm") is None:
        tqdm = types.ModuleType("tqdm")
        tqdm.tqdm = lambda iterable=None, *args, **kwargs: iterable
        stubs["tqdm"] = tqdm
    tool_modules = (
        "conversion",
        "pytorch_fitting",
        "pymomentum_fitting",
        "utils",
        "file_assets",
    )
    saved = {name: sys.modules.get(name) for name in (*stubs, *tool_modules)}
    sys.modules.update(stubs)
    sys.path.insert(0, str(_TOOL_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            f"mhr_smpl_{name}", _TOOL_DIR / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_TOOL_DIR))
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


pytorch_fitting = _load_tool_module("pytorch_fitting")
conversion = _load_tool_module("conversion")

_NUM_VERTICES = 60
_NUM_BETAS = 4
_BODY_POSE_DIM = 69  # The SMPL body pose size the fitting code expects.


def _rodrigues(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrices [B, 3, 3] from axis-angle vectors [B, 3]."""
    angle = torch.sqrt(axis_angle.square().sum(-1, keepdim=True) + 1e-12)
    k = axis_angle / angle
    zero = torch.zeros_like(k[:, 0])
    skew = torch.stack(
        [zero, -k[:, 2], k[:, 1], k[:, 2], zero, -k[:, 0], -k[:, 1], k[:, 0], zero],
        dim=1,
    ).reshape(-1, 3, 3)
    eye = torch.eye(3, dtype=axis_angle.dtype).expand_as(skew)
    sin, cos = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    return eye + sin * skew + (1 - cos) * skew @ skew


class _ToyBodyModel(torch.nn.Module):
    """A smooth, nonlinear stand-in with the SMPL model's call interface.

    Shape and pose offsets are linear in betas and body_pose; the result is
    rotated by global_orient and moved by transl.
    """

    num_betas = _NUM_BETAS
    num_expression_coeffs = 0

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.v_template = torch.randn(_NUM_VERTICES, 3, generator=generator)
        self.shapedirs = 0.1 * torch.randn(
            _NUM_VERTICES, 3, _NUM_BETAS, generator=generator
        )
        # Each pose value moves a few vertices along a fixed direction, through
        # a sine so that the model is not linear in the pose.
        self.posedirs = 0.2 * torch.randn(
            _BODY_POSE_DIM, _NUM_VERTICES, 3, generator=generator
        )
        self.faces = np.stack(
            [
                np.arange(_NUM_VERTICES),
                np.roll(np.arange(_NUM_VERTICES), -1),
                np.roll(np.arange(_NUM_VERTICES), -2),
            ],
            axis=1,
        )

    def forward(self, betas, body_pose, global_orient, transl, **unused):
        shaped = self.v_template + torch.einsum("vck,bk->bvc", self.shapedirs, betas)
        posed = shaped + torch.einsum(
            "pvc,bp->bvc", self.posedirs, torch.sin(body_pose)
        )
        rotated = posed @ _rodrigues(global_orient).transpose(1, 2)
        return types.SimpleNamespace(vertices=rotated + transl[:, None, :])


def _fitter(model: _ToyBodyModel):
    edges = torch.from_numpy(
        np.stack([np.arange(_NUM_VERTICES), np.roll(np.arange(_NUM_VERTICES), -1)], 1)
    )
    return pytorch_fitting.PyTorchSMPLFitting(
        smpl_model=model,
        smpl_edges=edges,
        smpl_model_type="smpl",
        hand_pose_dim=0,
        device="cpu",
        batch_size=256,
    )


def _random_parameters(num_frames: int, seed: int, scale: float = 0.3):
    generator = torch.Generator().manual_seed(seed)
    return {
        "betas": torch.randn(num_frames, _NUM_BETAS, generator=generator),
        "body_pose": scale
        * torch.randn(num_frames, _BODY_POSE_DIM, generator=generator),
        "global_orient": scale * torch.randn(num_frames, 3, generator=generator),
        "transl": torch.randn(num_frames, 3, generator=generator),
    }


def _vertices(model, parameters):
    with torch.no_grad():
        return model(**parameters).vertices


def _mean_error(model, parameters, target):
    return (_vertices(model, parameters) - target).norm(dim=-1).mean(1)


def _nudge(parameters, amount: float):
    return {name: value + amount for name, value in parameters.items()}


def test_levenberg_marquardt_never_increases_the_error():
    model = _ToyBodyModel()
    fitter = _fitter(model)
    truth = _random_parameters(3, seed=1)
    target = _vertices(model, truth)
    start = _nudge(truth, 0.2)

    before = _mean_error(model, start, target)
    refined = fitter._refine_with_levenberg_marquardt(target, start, iterations=1)
    after_one = _mean_error(model, refined, target)
    refined = fitter._refine_with_levenberg_marquardt(target, start, iterations=4)
    after_four = _mean_error(model, refined, target)

    assert torch.all(after_one < before)
    assert torch.all(after_four <= after_one + 1e-7)
    assert torch.all(after_four < 1e-3 * before)


def test_frames_are_refined_independently_of_each_other():
    model = _ToyBodyModel()
    fitter = _fitter(model)
    truth = _random_parameters(3, seed=2)
    target = _vertices(model, truth)
    start = _nudge(truth, 0.1)

    together = fitter._refine_with_levenberg_marquardt(target, start, iterations=2)
    for frame in range(3):
        alone = fitter._refine_with_levenberg_marquardt(
            target[frame : frame + 1],
            {name: value[frame : frame + 1] for name, value in start.items()},
            iterations=2,
        )
        for name in start:
            torch.testing.assert_close(
                together[name][frame : frame + 1], alone[name], atol=1e-4, rtol=1e-4
            )


def test_fast_fit_from_scratch_matches_the_targets():
    model = _ToyBodyModel()
    fitter = _fitter(model)
    truth = _random_parameters(2, seed=3, scale=0.1)
    target = _vertices(model, truth)

    result = fitter.fit(
        target.clone(), single_identity=False, is_tracking=False, fast=True
    )

    assert set(result) == {"betas", "body_pose", "global_orient", "transl"}
    assert result["body_pose"].shape == (2, _BODY_POSE_DIM)
    assert torch.all(_mean_error(model, result, target) < 0.01)


def test_fast_fit_runs_the_default_initial_pose_stages(monkeypatch):
    model = _ToyBodyModel()
    fitter = _fitter(model)
    target = _vertices(model, _random_parameters(2, seed=3, scale=0.1))
    default_iterations = inspect.signature(fitter._optimize_smpl).parameters[
        "iterations"
    ].default
    recorded = []
    optimize_smpl = fitter._optimize_smpl

    def recording_optimize_smpl(*args, **kwargs):
        recorded.append(kwargs.get("iterations"))
        return optimize_smpl(*args, **kwargs)

    monkeypatch.setattr(fitter, "_optimize_smpl", recording_optimize_smpl)

    fitter.fit(target.clone(), single_identity=False, is_tracking=False, fast=True)

    # Fewer initial pose steps leave some large poses in a wrong pose.
    assert recorded == [(default_iterations[0], 0)]


def test_fast_fit_from_scratch_reaches_large_poses(monkeypatch):
    model = _ToyBodyModel()
    targets = [
        _vertices(model, _random_parameters(16, seed=seed, scale=1.0))
        for seed in range(8)
    ]

    def mean_error_of_fast_fits():
        errors = []
        for target in targets:
            result = _fitter(model).fit(
                target.clone(), single_identity=False, is_tracking=False, fast=True
            )
            errors.append(_mean_error(model, result, target))
        return torch.cat(errors).mean()

    error = mean_error_of_fast_fits()
    # Half of the initial pose steps leave more of these large poses in a
    # wrong pose: about 2.7 times the error on this stand-in model.
    monkeypatch.setattr(
        pytorch_fitting.OptimizationConstants,
        "SMPL_FAST_COARSE_ITERATIONS",
        (20, 40, 20),
    )
    assert error < 0.6 * mean_error_of_fast_fits()


def test_fast_fit_starts_from_the_initial_parameters():
    model = _ToyBodyModel()
    fitter = _fitter(model)
    previous = _random_parameters(2, seed=4)
    truth = _nudge(previous, 0.05)  # The next video frame moved a little.
    target = _vertices(model, truth)

    result = fitter.fit(
        target.clone(),
        single_identity=False,
        is_tracking=False,
        fast=True,
        initial_parameters={k: v.numpy() for k, v in previous.items()},
    )

    assert torch.all(_mean_error(model, result, target) < 1e-3)


def test_fast_tracking_starts_each_frame_from_the_previous_one():
    model = _ToyBodyModel()
    fitter = _fitter(model)
    first = _random_parameters(1, seed=5)
    # A short sequence that moves and turns a little from frame to frame.
    steps = torch.arange(5, dtype=torch.float32)[:, None]
    truth = {name: value + 0.04 * steps for name, value in first.items()}
    truth["transl"] = first["transl"] + 0.5 * steps
    target = _vertices(model, truth)

    result = fitter.fit(
        target.clone(),
        single_identity=False,
        is_tracking=True,
        fast=True,
        initial_parameters=first,
    )

    assert torch.all(_mean_error(model, result, target) < 1e-3)


def test_a_poor_starting_point_is_fitted_again(monkeypatch):
    model = _ToyBodyModel()
    fitter = _fitter(model)
    truth = _random_parameters(2, seed=6, scale=0.1)
    target = _vertices(model, truth)
    poor_start = _random_parameters(2, seed=7, scale=1.5)
    # One LM step from a far start leaves a large error.
    monkeypatch.setattr(
        pytorch_fitting.OptimizationConstants, "SMPL_FAST_WARM_START_LM_ITERATIONS", 1
    )
    refits = []
    fit_fast_batch = fitter._fit_fast_batch

    def recording_fit_fast_batch(target_vertices, center, initialization):
        if initialization is None:
            refits.append(target_vertices.shape[0])
        return fit_fast_batch(target_vertices, center, initialization)

    monkeypatch.setattr(fitter, "_fit_fast_batch", recording_fit_fast_batch)

    result = fitter.fit(
        target.clone(),
        single_identity=False,
        is_tracking=False,
        fast=True,
        initial_parameters=poor_start,
    )

    assert refits, "the poor start was not fitted again"
    assert torch.all(_mean_error(model, result, target) < 0.01)


def _one_person(num_frames: int, seed: int, scale: float = 0.3):
    """Parameters of one person (one shape) in num_frames different poses."""
    parameters = _random_parameters(num_frames, seed=seed, scale=scale)
    parameters["betas"] = parameters["betas"][:1].expand(num_frames, -1).clone()
    return parameters


def test_fast_fit_with_a_single_identity_fits_one_shape():
    model = _ToyBodyModel()
    fitter = _fitter(model)
    truth = _one_person(6, seed=8)
    target = _vertices(model, truth)

    result = fitter.fit(
        target.clone(), single_identity=True, is_tracking=False, fast=True
    )

    assert torch.equal(result["betas"], result["betas"][:1].expand(6, -1))
    assert torch.allclose(result["betas"][0], truth["betas"][0], atol=1e-2)
    assert torch.all(_mean_error(model, result, target) < 1e-3)


def test_fast_tracking_with_a_single_identity_fits_one_shape():
    model = _ToyBodyModel()
    fitter = _fitter(model)
    first = _one_person(1, seed=9)
    steps = torch.arange(5, dtype=torch.float32)[:, None]
    truth = {name: value + 0.04 * steps for name, value in first.items()}
    truth["betas"] = first["betas"].expand(5, -1)
    truth["transl"] = first["transl"] + 0.5 * steps
    target = _vertices(model, truth)

    result = fitter.fit(
        target.clone(), single_identity=True, is_tracking=True, fast=True
    )

    assert torch.equal(result["betas"], result["betas"][:1].expand(5, -1))
    assert torch.all(_mean_error(model, result, target) < 1e-3)


def test_shared_identity_refinement_lowers_the_error_of_the_mean_shape(
    monkeypatch,
):
    model = _ToyBodyModel()
    fitter = _fitter(model)
    truth = _one_person(6, seed=10)
    target = _vertices(model, truth)
    # Per-frame fits, each with its own shape, as the fast path makes them.
    per_frame = fitter.fit(
        target.clone(), single_identity=False, is_tracking=False, fast=True
    )
    per_frame = {name: value + 0.02 for name, value in per_frame.items()}
    per_frame["transl"] = per_frame["transl"] - 0.5 * (
        target.min(1)[0] + target.max(1)[0]
    )
    centered = target - 0.5 * (target.min(1)[0] + target.max(1)[0])[:, None, :]

    def total_error(parameters):
        return _mean_error(model, parameters, centered).sum()

    monkeypatch.setattr(
        pytorch_fitting.OptimizationConstants,
        "SMPL_FAST_SHARED_IDENTITY_LM_ITERATIONS",
        0,
    )
    mean_shape = fitter._refine_shared_identity(centered, per_frame)
    monkeypatch.undo()
    refined = fitter._refine_shared_identity(centered, per_frame)

    assert torch.equal(refined["betas"], refined["betas"][:1].expand(6, -1))
    assert total_error(refined) < 0.5 * total_error(mean_shape)


def test_fast_fit_rejects_unsupported_options():
    fitter = _fitter(_ToyBodyModel())
    target = torch.zeros(2, _NUM_VERTICES, 3)

    with pytest.raises(ValueError, match="requires fast=True"):
        fitter.fit(
            target.clone(),
            single_identity=False,
            is_tracking=False,
            initial_parameters=_random_parameters(2, seed=0),
        )
    with pytest.raises(ValueError, match="expected 1 or 2"):
        fitter.fit(
            target.clone(),
            single_identity=False,
            is_tracking=False,
            fast=True,
            initial_parameters=_random_parameters(3, seed=0),
        )


def test_conversion_passes_the_fast_options_to_the_fitter(monkeypatch):
    recorded = {}

    class RecordingFitter:
        def __init__(self, **kwargs):
            pass

        def fit(self, **kwargs):
            recorded.update(kwargs)
            return {}

    monkeypatch.setattr(conversion, "PyTorchSMPLFitting", RecordingFitter)
    converter = object.__new__(conversion.Conversion)
    converter.__dict__.update(
        _smpl_model=None,
        _smpl_edges=None,
        smpl_model_type="smpl",
        _hand_pose_dim=0,
        _DEVICE="cpu",
        _batch_size=256,
    )
    start = _random_parameters(1, seed=0)

    converter._m2s_fit_smpl_using_pytorch(
        torch.zeros(1, _NUM_VERTICES, 3),
        single_identity=False,
        is_tracking=True,
        fast_fitting=True,
        initial_smpl_parameters=start,
    )

    assert recorded["fast"] is True
    assert recorded["initial_parameters"] is start
    assert recorded["is_tracking"] is True


def test_sam3d_conversion_passes_the_fast_options_on(monkeypatch):
    converter = object.__new__(conversion.Conversion)
    converter._DEVICE = "cpu"
    recorded = {}
    monkeypatch.setattr(
        converter, "convert_mhr2smpl", lambda **kwargs: recorded.update(kwargs)
    )
    person = {
        "mhr_model_params": np.zeros(204, np.float32),
        "shape_params": np.zeros(45, np.float32),
        "expr_params": np.zeros(72, np.float32),
        "pred_vertices": np.zeros((5, 3), np.float32),
        "pred_cam_t": np.zeros(3, np.float32),
    }
    start = _random_parameters(2, seed=0)

    converter.convert_sam3d_output_to_smpl(
        [person, person], fast_fitting=True, initial_smpl_parameters=start
    )

    assert recorded["fast_fitting"] is True
    assert recorded["initial_smpl_parameters"] is start
    assert recorded["mhr_vertices"].shape == (2, 5, 3)


def test_sam3d_vertices_in_meters_are_fitted_in_centimeters(monkeypatch):
    # SAM 3D Body returns meters and a camera translation; the SMPL fit runs in
    # centimeters on the camera-space mesh.
    converter = object.__new__(conversion.Conversion)
    converter._DEVICE = "cpu"
    recorded = {}
    monkeypatch.setattr(
        converter, "convert_mhr2smpl", lambda **kwargs: recorded.update(kwargs)
    )
    vertices = np.array([[0.1, 1.5, -0.2], [-0.3, 0.0, 0.05]], np.float32)
    cam_t = np.array([0.01, -0.02, 3.0], np.float32)

    converter.convert_sam3d_output_to_smpl(
        [
            {
                "mhr_model_params": np.zeros(204, np.float32),
                "shape_params": np.zeros(45, np.float32),
                "expr_params": np.zeros(72, np.float32),
                "pred_vertices": vertices,
                "pred_cam_t": cam_t,
            }
        ]
    )

    np.testing.assert_allclose(
        recorded["mhr_vertices"].numpy(), 100.0 * (vertices + cam_t)[None], rtol=1e-6
    )

