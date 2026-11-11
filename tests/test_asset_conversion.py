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

import builtins
import gc
import os

from pathlib import Path

import pytest
import torch

from mhr.mhr import MHR
from scripts.convert_assets import convert_assets


ASSETS = os.environ.get("MHR_TEST_ASSETS")


@pytest.mark.skipif(ASSETS is None, reason="set MHR_TEST_ASSETS for parity test")
def test_converted_assets_match_legacy(tmp_path: Path) -> None:
    pytest.importorskip("pymomentum.geometry")
    assets = Path(ASSETS)
    convert_assets(assets, tmp_path, range(7), "test")

    original_import = builtins.__import__

    def reject_pymomentum(name, *args, **kwargs):
        if name == "pymomentum" or name.startswith("pymomentum."):
            raise AssertionError("converted assets imported PyMomentum")
        return original_import(name, *args, **kwargs)

    for lod in range(7):
        with pytest.warns(DeprecationWarning, match="PyMomentum Character"):
            legacy = MHR.from_files(assets, device="cpu", lod=lod)
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(builtins, "__import__", reject_pymomentum)
            converted = MHR.from_files(tmp_path, device="cpu", lod=lod)

            torch.manual_seed(1234 + lod)
            values = (
                0.1 * torch.randn(1, 45),
                0.05 * torch.randn(1, 204),
                0.1 * torch.randn(1, 72),
            )
            legacy_inputs = [value.clone().requires_grad_() for value in values]
            converted_inputs = [value.clone().requires_grad_() for value in values]
            legacy_vertices, legacy_skeleton = legacy(*legacy_inputs)
            converted_vertices, converted_skeleton = converted(*converted_inputs)

        torch.testing.assert_close(
            converted_vertices, legacy_vertices, atol=1e-4, rtol=1e-5
        )
        torch.testing.assert_close(
            converted_skeleton, legacy_skeleton, atol=1e-4, rtol=1e-5
        )

        weights = torch.randn_like(legacy_vertices)
        (legacy_vertices * weights).sum().backward()
        (converted_vertices * weights).sum().backward()
        for converted_input, legacy_input in zip(converted_inputs, legacy_inputs):
            torch.testing.assert_close(
                converted_input.grad, legacy_input.grad, atol=2e-3, rtol=1e-5
            )

        assert set(converted.state_dict()) == set(legacy.state_dict())
        del legacy, converted, legacy_vertices, converted_vertices
        gc.collect()
