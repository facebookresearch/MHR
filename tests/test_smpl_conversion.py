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

The tool tests use the small SMPL stand-in of test_smpl_fast_fitting.py and an
SMPL-X-sized stand-in, so no licensed SMPL(X) model file is needed. The tests
on the MHR model run when MHR_TEST_ASSETS names the release asset folder.
"""

import os
import sys
import types

import numpy as np
import pytest
import torch

from tests.test_smpl_fast_fitting import (
    _fitter,
    _load_tool_module,
    _NUM_VERTICES,
    _one_person,
    _random_parameters,
    _rodrigues,
    _TOOL_DIR,
    _ToyBodyModel,
    _vertices,
    conversion,
)

ASSETS = os.environ.get("MHR_TEST_ASSETS")
_NUM_VERTICES_SMPL = 6890
_NUM_VERTICES_SMPLX = 10475

utils = _load_tool_module("utils")


def _converter(smpl_model) -> object:
    converter = object.__new__(conversion.Conversion)
    converter.__dict__.update(
        _smpl_model=smpl_model,
        smpl_model_type="smpl",
        _hand_pose_dim=0,
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

    monkeypatch.setattr(converter, "_compute_target_vertices", lambda v, d, **k: v)
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


def _mhr_rest_vertices(model) -> torch.Tensor:
    with torch.no_grad():
        vertices, _ = model(torch.zeros(1, 45), torch.zeros(1, 204), torch.zeros(1, 72))
    return vertices


@pytest.mark.skipif(ASSETS is None, reason="set MHR_TEST_ASSETS to the asset folder")
@pytest.mark.parametrize("sequence", ["tracking", "reprocessing"])
def test_pymomentum_tracking_fits_the_face_of_every_frame(sequence):
    from mhr.mhr import MHR

    model = MHR.from_files(ASSETS, device=torch.device("cpu"), lod=1)
    smpl_model = types.SimpleNamespace(
        v_template=torch.zeros(_NUM_VERTICES_SMPL, 3), to=lambda device: smpl_model
    )
    converter = conversion.Conversion(model, smpl_model, method="pymomentum")
    generator = torch.Generator().manual_seed(0)
    identity = 0.5 * torch.randn(1, 45, generator=generator)
    if sequence == "tracking":
        # Frame 0 has a neutral face, the later frames their own expression.
        weights = torch.ones(3, 1)
        expression = torch.rand(3, 72, generator=generator)
        expression[0] = 0
    else:
        # Failure cases are reprocessed by tracking an interpolation from zero
        # parameters to the parameters of the failed frame.
        weights = torch.linspace(0, 1, 5)[:, None]
        expression = weights * torch.rand(1, 72, generator=generator)
    with torch.no_grad():
        target, _ = model(
            weights * identity, torch.zeros(len(weights), 204), expression
        )

    tracked = converter._s2m_fit_mhr_using_pymomentum(
        target, single_identity=False, is_tracking=True
    )
    per_frame = converter._s2m_fit_mhr_using_pymomentum(
        target, single_identity=False, is_tracking=False
    )

    face_expression = tracked["face_expr_coeffs"]
    head_identity = tracked["identity_coeffs"][:, 20:40]
    for frame in range(1, len(weights)):
        assert not torch.equal(face_expression[frame], face_expression[0]), frame
        assert not torch.equal(head_identity[frame], head_identity[0]), frame
    head = torch.from_numpy(
        np.load(_TOOL_DIR / "assets" / "head_hand_mask.npz")["head_mask"] > 0
    )

    def head_error(parameters):
        with torch.no_grad():
            vertices, _ = model(
                parameters["identity_coeffs"],
                parameters["lbs_model_params"],
                parameters["face_expr_coeffs"],
            )
        return (vertices - target).norm(dim=-1)[:, head].mean(dim=1)

    # Tracking only initializes each frame from the previous one, so it fits
    # the head about as well as fitting every frame on its own (cm).
    tracked_error = head_error(tracked)[1:]
    per_frame_error = head_error(per_frame)[1:]
    assert (tracked_error <= 2 * per_frame_error + 0.05).all(), (
        tracked_error,
        per_frame_error,
    )


class _ToySMPLXModel(torch.nn.Module):
    """An SMPL-X-sized stand-in with the hand pose and expression sizes of
    smplx.SMPLX.

    Like smplx, it maps hand poses through the first num_pca_comps PCA
    components when use_pca is set and reshapes them to 15 joints otherwise,
    and appends the expression to the betas before the shape blendshapes. Zero
    parameters give v_template.
    """

    num_betas = 10

    def __init__(
        self,
        v_template: np.ndarray,
        use_pca: bool = True,
        num_pca_comps: int = 6,
        num_expression_coeffs: int = 10,
    ) -> None:
        super().__init__()
        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        self.num_expression_coeffs = num_expression_coeffs
        generator = torch.Generator().manual_seed(0)
        self.v_template = torch.as_tensor(v_template, dtype=torch.float32)
        self.shapedirs = 1e-3 * torch.randn(
            _NUM_VERTICES_SMPLX,
            3,
            self.num_betas + num_expression_coeffs,
            generator=generator,
        )
        self.hand_components = torch.eye(45)[:num_pca_comps]
        self.posedirs = 1e-3 * torch.randn(
            63 + 2 * 45, _NUM_VERTICES_SMPLX, 3, generator=generator
        )
        # The SMPL-X to MHR mapping indexes up to 20897 triangles.
        index = np.arange(20908)
        self.faces = np.stack([index, index + 1, index + 2], 1) % _NUM_VERTICES_SMPLX

    def forward(
        self,
        betas,
        body_pose,
        global_orient,
        transl,
        left_hand_pose,
        right_hand_pose,
        expression,
        **unused,
    ):
        shape = torch.cat([betas, expression], dim=1)
        hand_poses = []
        for hand_pose in (left_hand_pose, right_hand_pose):
            if self.use_pca:
                hand_pose = torch.einsum("bi,ij->bj", hand_pose, self.hand_components)
            hand_poses.append(hand_pose.reshape(-1, 15, 3).flatten(1))
        pose = torch.cat([body_pose.reshape(-1, 63), *hand_poses], dim=1)
        shaped = self.v_template + torch.einsum("vck,bk->bvc", self.shapedirs, shape)
        posed = shaped + torch.einsum("pvc,bp->bvc", self.posedirs, torch.sin(pose))
        rotated = posed @ _rodrigues(global_orient.reshape(-1, 3)).transpose(1, 2)
        return types.SimpleNamespace(vertices=rotated + transl[:, None, :])


@pytest.fixture(scope="module")
def mhr_rest_body_on_smplx():
    """The MHR LOD 1 model and its rest body mapped onto the SMPL-X vertices
    (meters), the exact MHR-to-SMPL-X answer for the MHR rest body."""

    from mhr.mhr import MHR

    model = MHR.from_files(ASSETS, device=torch.device("cpu"), lod=1)
    mapping = np.load(_TOOL_DIR / "assets" / "mhr2smplx_mapping.npz")
    faces = np.asarray(model.character.mesh.faces)
    corners = _mhr_rest_vertices(model)[0].numpy()[faces[mapping["triangle_ids"]]]
    v_template = (corners * mapping["baryc_coords"][..., None]).sum(1) * 0.01
    return model, v_template


@pytest.mark.skipif(ASSETS is None, reason="set MHR_TEST_ASSETS to the asset folder")
@pytest.mark.parametrize(
    "smplx_options",
    [
        {},
        {"use_pca": False},
        {"num_pca_comps": 12},
        {"num_expression_coeffs": 50},
    ],
    ids=["defaults", "no_pca", "12_pca_components", "50_expressions"],
)
def test_smplx_conversion_follows_the_hand_pose_and_expression_sizes(
    mhr_rest_body_on_smplx, smplx_options
):
    model, v_template = mhr_rest_body_on_smplx
    smplx_model = _ToySMPLXModel(v_template, **smplx_options)

    # Without hand poses and expression, the conversion fills them in.
    parameters = {
        "betas": torch.zeros(1, 10),
        "body_pose": torch.zeros(1, 63),
        "global_orient": torch.zeros(1, 3),
        "transl": torch.zeros(1, 3),
    }
    result = conversion.Conversion(
        model, smplx_model, method="pymomentum"
    ).convert_smpl2mhr(smpl_parameters=parameters, return_mhr_vertices=True)
    assert result.result_vertices.shape == (1, 18439, 3)
    assert np.isfinite(result.result_errors).all()

    result = conversion.Conversion(
        model, smplx_model, method="pytorch"
    ).convert_mhr2smpl(
        mhr_vertices=_mhr_rest_vertices(model),
        return_smpl_vertices=True,
        fast_fitting=True,
    )
    deviation_cm = 100 * np.linalg.norm(result.result_vertices[0] - v_template, axis=-1)
    assert deviation_cm.max() < 1.0

    # Fit results without hand poses are scored with zero hand poses.
    converter = conversion.Conversion(model, smplx_model, method="pytorch")
    fitter = conversion.PyTorchSMPLFitting(
        smpl_model=smplx_model,
        smpl_edges=converter._smpl_edges,
        smpl_model_type="smplx",
        hand_pose_dim=converter._hand_pose_dim,
        device="cpu",
    )
    body_parameters = {
        "betas": torch.zeros(1, 10),
        "body_pose": torch.zeros(1, 63),
        "global_orient": torch.zeros(1, 3),
        "transl": torch.zeros(1, 3),
    }
    errors = fitter._evaluate_conversion_error(
        body_parameters, torch.as_tensor(v_template, dtype=torch.float32)[None]
    )
    assert errors.max() < 1e-4


@pytest.mark.skipif(ASSETS is None, reason="set MHR_TEST_ASSETS to the asset folder")
@pytest.mark.parametrize("lod", [0, 1, 2, 6])
def test_mhr_to_smplx_returns_the_rest_body_or_refuses_the_lod(
    mhr_rest_body_on_smplx, lod
):
    from mhr.mhr import MHR

    lod1_model, v_template = mhr_rest_body_on_smplx
    model = MHR.from_files(ASSETS, device=torch.device("cpu"), lod=lod)
    # Both a converter of this LOD and MHR vertices of this LOD given to an
    # LOD 1 converter.
    for converter_model in [model, lod1_model]:
        try:
            converter = conversion.Conversion(
                converter_model, _ToySMPLXModel(v_template), method="pytorch"
            )
            result = converter.convert_mhr2smpl(
                mhr_vertices=_mhr_rest_vertices(model),
                return_smpl_vertices=True,
                fast_fitting=True,
            )
        except ValueError as error:
            # The surface mappings are made for one LOD; a clear refusal is fine.
            assert lod != 1 and "LOD 1" in str(error)
            continue
        deviation_cm = 100 * np.linalg.norm(
            result.result_vertices[0] - v_template, axis=-1
        )
        assert deviation_cm.max() < 1.0


def _mhr_converter(monkeypatch):
    """A PyTorch converter of the LOD 1 release model with one step per stage."""
    from mhr.mhr import MHR

    model = MHR.from_files(ASSETS, device=torch.device("cpu"), lod=1)
    smpl_model = types.SimpleNamespace(
        v_template=torch.zeros(_NUM_VERTICES_SMPL, 3), to=lambda device: smpl_model
    )
    converter = conversion.Conversion(model, smpl_model, method="pytorch")
    converter._s2m_load_masks()
    optimize_mhr = conversion.PyTorchMHRFitting._optimize_mhr

    def optimize_mhr_briefly(self, *args, **kwargs):
        kwargs["stage_iterations"] = [[1, 1], [1, 1, 1], [5]]
        return optimize_mhr(self, *args, **kwargs)

    monkeypatch.setattr(
        conversion.PyTorchMHRFitting, "_optimize_mhr", optimize_mhr_briefly
    )
    return converter, model


def _mhr_vertices(model, num_frames):
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        vertices, _ = model(
            torch.randn(num_frames, 45, generator=generator),
            0.05 * torch.randn(num_frames, 204, generator=generator),
            torch.rand(num_frames, 72, generator=generator),
        )
    return vertices


@pytest.mark.parametrize(
    "target_model",
    [
        "smpl",
        pytest.param(
            "mhr",
            marks=pytest.mark.skipif(
                ASSETS is None, reason="set MHR_TEST_ASSETS to the asset folder"
            ),
        ),
    ],
)
@pytest.mark.parametrize("single_identity", [False, True])
def test_the_fit_does_not_depend_on_the_batch_size(
    target_model, single_identity, monkeypatch
):
    # Every stage steps on the loss of all frames, added up batch by batch, so
    # every batch size must give the fit of one batch, also for one body shape
    # shared by all frames. The targets are two people, so that a batch can
    # hold one person or both.
    fits = {}
    if target_model == "smpl":
        # In float64 the batch sums match the sum of one batch to about 1e-13.
        dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            model = _ToyBodyModel()
            people = [_one_person(6, seed=seed, scale=0.1) for seed in (21, 22)]
            target = torch.cat([_vertices(model, person) for person in people])
            for batch_size in (256, 1, 4, 8):
                fitter = _fitter(model)
                fitter._batch_size = batch_size
                fits[batch_size] = fitter.fit(
                    target.clone(),
                    single_identity=single_identity,
                    is_tracking=False,
                )
        finally:
            torch.set_default_dtype(dtype)
        atol = 1e-9
    else:
        converter, model = _mhr_converter(monkeypatch)
        generator = torch.Generator().manual_seed(0)
        identity = torch.randn(2, 45, generator=generator).repeat_interleave(2, 0)
        with torch.no_grad():
            target, _ = model(
                identity,
                0.05 * torch.randn(4, 204, generator=generator),
                torch.zeros(4, 72),
            )
        # In float32, one thread keeps the float sums the same in every run.
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for batch_size in (256, 1, 2, 3):
                fitter = conversion.PyTorchMHRFitting(
                    mhr_model=model,
                    mhr_edges=converter._mhr_edges,
                    mhr_vertex_mask=converter._mhr_vertex_mask,
                    mhr_param_masks=converter._mhr_param_masks,
                    device="cpu",
                    batch_size=batch_size,
                )
                fits[batch_size] = fitter.fit(
                    target.clone(), single_identity=single_identity, is_tracking=False
                )
        finally:
            torch.set_num_threads(threads)
        # float32 sums in another order differ by up to about 2e-4 here.
        atol = 5e-4

    single_batch = fits.pop(256)
    for batch_size, fit in fits.items():
        for name, value in single_batch.items():
            torch.testing.assert_close(
                fit[name].detach(),
                value.detach(),
                rtol=0,
                atol=atol,
                msg=lambda message: f"batch_size={batch_size}, {name}: {message}",
            )


@pytest.mark.skipif(ASSETS is None, reason="set MHR_TEST_ASSETS to the asset folder")
def test_pytorch_mhr_fit_returns_parameters_without_gradients(monkeypatch):
    converter, model = _mhr_converter(monkeypatch)

    parameters = converter._s2m_fit_mhr_using_pytorch(
        _mhr_vertices(model, 2), single_identity=False
    )

    for name, value in parameters.items():
        assert not value.requires_grad, name
        value.numpy()


@pytest.mark.parametrize(
    "single_identity,is_tracking,num_frames,batch_size",
    [
        (False, False, 2, 256),
        (True, False, 2, 256),
        (True, True, 2, 256),
        (False, True, 17, 1),  # More than 16 x batch_size frames are tracked.
    ],
)
def test_default_smpl_fit_returns_parameters_without_gradients(
    single_identity, is_tracking, num_frames, batch_size
):
    model = _ToyBodyModel()
    fitter = _fitter(model)
    fitter._batch_size = batch_size
    target = _vertices(model, _random_parameters(num_frames, seed=12, scale=0.1))

    parameters = fitter.fit(
        target.clone(), single_identity=single_identity, is_tracking=is_tracking
    )

    for name, value in parameters.items():
        assert not value.requires_grad, name
        value.numpy()


@pytest.mark.parametrize(
    "method", ["convert_mhr2smpl", "convert_sam3d_output_to_smpl", "convert_smpl2mhr"]
)
def test_a_batch_size_given_to_a_call_applies_to_that_call_only(method, monkeypatch):
    # A stand-in for the LOD 1 model, with its 18439 mesh vertices.
    mesh = types.SimpleNamespace(vertices=np.zeros((18439, 3)))
    mhr_model = types.SimpleNamespace(
        character=types.SimpleNamespace(mesh=mesh), to=lambda device: mhr_model
    )
    smpl_model = types.SimpleNamespace(
        v_template=torch.zeros(_NUM_VERTICES_SMPL, 3), to=lambda device: smpl_model
    )
    converter = conversion.Conversion(
        mhr_model, smpl_model, method="pytorch", batch_size=4
    )
    used_batch_sizes = {"meshes": [], "fit": []}

    def record(step, result):
        def run(*args, batch_size=None, **kwargs):
            used_batch_sizes[step].append(
                converter._batch_size if batch_size is None else batch_size
            )
            return result

        return run

    mhr_vertices = np.zeros((1, 18439, 3), np.float32)
    monkeypatch.setattr(
        converter, "_mhr_para2mesh", record("meshes", (None, mhr_vertices))
    )
    monkeypatch.setattr(converter, "_m2s_fit_smpl_using_pytorch", record("fit", {}))
    monkeypatch.setattr(converter, "_s2m_fit_mhr_using_pytorch", record("fit", {}))
    monkeypatch.setattr(converter, "_compute_target_vertices", lambda v, *a, **k: v)
    monkeypatch.setattr(converter, "_s2m_load_masks", lambda: None)
    monkeypatch.setattr(
        converter, "_m2s_evaluate_conversion_error", lambda *a, **k: None
    )
    monkeypatch.setattr(
        converter, "_s2m_evaluate_conversion_error", lambda *a, **k: None
    )
    # Without pred_vertices, the MHR meshes are made from the parameters.
    person = {
        "mhr_model_params": np.zeros(204, np.float32),
        "shape_params": np.zeros(45, np.float32),
        "expr_params": np.zeros(72, np.float32),
        "pred_cam_t": np.zeros(3, np.float32),
    }
    inputs = {
        "convert_mhr2smpl": {"mhr_vertices": mhr_vertices},
        "convert_sam3d_output_to_smpl": {"sam3d_outputs": [person]},
        "convert_smpl2mhr": {
            "smpl_vertices": np.zeros((1, _NUM_VERTICES_SMPL, 3), np.float32)
        },
    }[method]
    convert = getattr(converter, method)

    convert(**inputs)
    convert(**inputs, batch_size=2)
    convert(**inputs)

    assert used_batch_sizes["fit"] == [4, 2, 4]
    if method == "convert_sam3d_output_to_smpl":
        assert used_batch_sizes["meshes"] == [4, 2, 4]


@pytest.mark.parametrize(
    "people", ["nobody", "without pred_vertices", "only pred_vertices"]
)
def test_sam3d_conversion_of_people_with_some_outputs_missing(people, monkeypatch):
    converter = object.__new__(conversion.Conversion)
    converter._DEVICE = "cpu"
    recorded = {}
    monkeypatch.setattr(
        converter, "convert_mhr2smpl", lambda **kwargs: recorded.update(kwargs)
    )
    monkeypatch.setattr(
        converter,
        "_mhr_para2mesh",
        lambda parameters, return_mesh, **kwargs: (
            None,
            np.ones((1, 5, 3), np.float32),
        ),
    )
    person = {
        "mhr_model_params": np.zeros(204, np.float32),
        "shape_params": np.zeros(45, np.float32),
        "expr_params": np.zeros(72, np.float32),
        "pred_vertices": np.zeros((5, 3), np.float32),
        "pred_cam_t": np.zeros(3, np.float32),
    }
    if people == "without pred_vertices":
        del person["pred_vertices"]
    elif people == "only pred_vertices":
        for name in ("mhr_model_params", "shape_params", "expr_params"):
            del person[name]

    result = converter.convert_sam3d_output_to_smpl(
        [] if people == "nobody" else [person]
    )

    if people == "nobody":
        # SAM 3D Body returns no outputs for an image without a person.
        assert result == conversion.ConversionResult()
        assert recorded == {}
    else:
        assert recorded["mhr_vertices"].shape == (1, 5, 3)


@pytest.mark.parametrize("given", ["--smpl", "--smplx"])
def test_example_skips_the_model_that_is_not_given(given, monkeypatch):
    example = _load_tool_module("example")
    monkeypatch.setattr(sys, "argv", ["example.py", given, "model.npz"])

    arguments = example._parse_arguments()

    # run_examples skips the conversions of a model whose file is None.
    assert (arguments.smpl, arguments.smplx) == (
        ("model.npz", None) if given == "--smpl" else (None, "model.npz")
    )
