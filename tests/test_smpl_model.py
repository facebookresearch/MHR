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


import numpy as np
import pytest

from mhr.smpl import (
    _normalize_smpl_model_data,
    _reshape_blendshape_basis,
)


def test_reshape_component_first_blendshape_basis() -> None:
    basis = np.arange(24).reshape(2, 12)

    reshaped = _reshape_blendshape_basis("posedirs", basis, num_vertices=4)

    assert reshaped.shape == (4, 3, 2)
    np.testing.assert_array_equal(reshaped.reshape(12, 2), basis.T)


def test_normalize_standard_smpl_model_data() -> None:
    num_vertices = 6890
    model_data = {
        "v_template": np.zeros((num_vertices, 3), dtype=np.float32),
        "shapedirs": np.zeros((10, num_vertices * 3), dtype=np.float32),
        "posedirs": np.zeros((207, num_vertices * 3), dtype=np.float32),
        "J_regressor": np.zeros((24, num_vertices), dtype=np.float32),
        "weights": np.zeros((num_vertices, 24), dtype=np.float32),
        "kintree_table": np.zeros((2, 24), dtype=np.int64),
        "f": np.zeros((1, 3), dtype=np.int64),
    }

    normalized = _normalize_smpl_model_data(model_data)

    assert normalized["shapedirs"].shape == (num_vertices, 3, 10)
    assert normalized["posedirs"].shape == (num_vertices, 3, 207)


def test_reject_nonstandard_smpl_topology() -> None:
    model_data = {
        key: np.empty(0)
        for key in (
            "J_regressor",
            "f",
            "kintree_table",
            "posedirs",
            "shapedirs",
            "weights",
        )
    }
    model_data["v_template"] = np.zeros((10, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="6890 vertices"):
        _normalize_smpl_model_data(model_data)
