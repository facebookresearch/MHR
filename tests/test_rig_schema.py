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

from pathlib import Path

import numpy as np
import pytest

from mhr._rig_schema import validate_rig_assets
from tests import rig_assets


def _validate(rig, lod, expected_lod=1):
    validate_rig_assets(rig, lod, Path("rig.npz"), Path("lod1.npz"), expected_lod)


def test_valid_assets_pass():
    _validate(rig_assets.rig_arrays(), rig_assets.lod_arrays())


def test_unsupported_schema_is_rejected():
    with pytest.raises(ValueError, match="unsupported MHR asset schema in rig.npz"):
        _validate(rig_assets.rig_arrays(schema_version=99), rig_assets.lod_arrays())
    with pytest.raises(ValueError, match="unsupported MHR asset schema in lod1.npz"):
        _validate(rig_assets.rig_arrays(), rig_assets.lod_arrays(schema_version=99))


def test_wrong_lod_is_rejected():
    with pytest.raises(ValueError, match="wrong LOD"):
        _validate(rig_assets.rig_arrays(), rig_assets.lod_arrays(lod=2))


@pytest.mark.parametrize("name", ["joint_offsets", "fk_sizes", "limit_min"])
def test_missing_rig_array_is_rejected(name):
    rig = rig_assets.rig_arrays()
    del rig[name]
    with pytest.raises(ValueError, match=f"rig.npz is missing required arrays: {name}"):
        _validate(rig, rig_assets.lod_arrays())


@pytest.mark.parametrize("name", ["base_shape", "skin_weights"])
def test_missing_lod_array_is_rejected(name):
    lod = rig_assets.lod_arrays()
    del lod[name]
    with pytest.raises(
        ValueError, match=f"lod1.npz is missing required arrays: {name}"
    ):
        _validate(rig_assets.rig_arrays(), lod)


def test_bad_shapes_are_rejected():
    rig = rig_assets.rig_arrays()
    rig["parameter_transform"] = rig["parameter_transform"][:-1]
    with pytest.raises(ValueError, match="invalid parameter transform shape"):
        _validate(rig, rig_assets.lod_arrays())

    lod = rig_assets.lod_arrays()
    lod["shape_vectors"] = lod["shape_vectors"][:, :1]
    with pytest.raises(ValueError, match="invalid blend shape dimensions"):
        _validate(rig_assets.rig_arrays(), lod)

    rig = rig_assets.rig_arrays()
    rig["fk_sizes"] = np.asarray([2], dtype=np.int64)
    with pytest.raises(ValueError, match="invalid FK schedule"):
        _validate(rig, rig_assets.lod_arrays())


def test_skin_index_out_of_range_is_rejected():
    lod = rig_assets.lod_arrays()
    lod["skin_indices"][0, 0] = 2
    with pytest.raises(ValueError, match="skinning joint index out of range"):
        _validate(rig_assets.rig_arrays(), lod)
