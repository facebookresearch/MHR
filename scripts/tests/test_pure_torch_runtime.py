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

"""`from mhr import MHR` on the real assets, without PyMomentum.

Set MHR_TEST_ASSETS to the folder of the release assets. The test converts
LOD 1 and compares the pure-PyTorch model, run in a fresh interpreter in which
PyMomentum cannot be imported, with the TorchScript model of the release and
with mhr.mhr.MHR.
"""

import os
import subprocess
import sys

from pathlib import Path

import pytest
import torch

pytest.importorskip("pymomentum.geometry")

from mhr.mhr import MHR as PyMomentumMHR
from scripts.convert_assets import convert_assets

ASSETS = os.environ.get("MHR_TEST_ASSETS")

_RUN_WITHOUT_PYMOMENTUM = """
import sys
sys.modules["pymomentum"] = None
import torch
from mhr import MHR

converted, inputs, output = sys.argv[1:]
model = MHR.from_files(converted, device="cpu", lod=1)
with torch.no_grad():
    vertices, skeleton = model(*torch.load(inputs))
torch.save((vertices, skeleton, sorted(model.state_dict())), output)
"""


@pytest.mark.skipif(ASSETS is None, reason="set MHR_TEST_ASSETS to the asset folder")
def test_pure_pytorch_model_matches_the_release_models(tmp_path: Path) -> None:
    assets = Path(ASSETS).resolve()
    convert_assets(assets, tmp_path, [1], "test")

    generator = torch.Generator().manual_seed(0)
    inputs = (
        0.8 * torch.randn(16, 45, generator=generator),
        0.2 * (torch.rand(16, 204, generator=generator) - 0.5),
        0.3 * torch.randn(16, 72, generator=generator),
    )
    torch.save(inputs, tmp_path / "inputs.pt")
    subprocess.run(
        [
            sys.executable,
            "-c",
            _RUN_WITHOUT_PYMOMENTUM,
            str(tmp_path),
            str(tmp_path / "inputs.pt"),
            str(tmp_path / "output.pt"),
        ],
        check=True,
        cwd=tmp_path,
    )
    vertices, skeleton, state_keys = torch.load(tmp_path / "output.pt")

    reference = PyMomentumMHR.from_files(assets, device=torch.device("cpu"), lod=1)
    scripted = torch.jit.load(str(assets / "mhr_model.pt"))
    with torch.no_grad():
        reference_vertices, reference_skeleton = reference(*inputs)
        scripted_vertices, _ = scripted(*inputs)

    # The same tolerance as the PyTorch rig parity test, in cm.
    torch.testing.assert_close(vertices, reference_vertices, atol=1e-3, rtol=0)
    torch.testing.assert_close(skeleton, reference_skeleton, atol=1e-3, rtol=0)
    torch.testing.assert_close(vertices, scripted_vertices, atol=1e-3, rtol=0)
    # Checkpoints of either model load into the other.
    assert state_keys == sorted(reference.state_dict())
