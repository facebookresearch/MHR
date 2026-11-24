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

"""Tests for the body-part segmentation example (tools/mhr_create_segmentation).

The test runs when MHR_TEST_ASSETS names the release asset folder.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

ASSETS = os.environ.get("MHR_TEST_ASSETS")
_EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "mhr_create_segmentation"
    / "example.py"
)


@pytest.mark.skipif(ASSETS is None, reason="set MHR_TEST_ASSETS to the asset folder")
def test_every_vertex_gets_one_body_part():
    pytest.importorskip("pymomentum.geometry")
    pytest.importorskip("trimesh")
    import torch

    from mhr.mhr import MHR

    spec = importlib.util.spec_from_file_location("mhr_segmentation", _EXAMPLE)
    segmentation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(segmentation)

    character = MHR.from_files(
        ASSETS, device=torch.device("cpu"), lod=1, wants_pose_correctives=False
    ).character
    names = list(character.skeleton.joint_names)
    skin = character.skin_weights
    vertices = np.asarray(character.mesh.vertices)

    groups = segmentation.build_joint_groups(names)
    soft, _ = segmentation.compute_part_weights(groups, skin, hard_assignment=False)
    hard, parts = segmentation.compute_part_weights(
        groups, skin, hard_assignment=True
    )

    # Every vertex is fully covered, and the hard segmentation is a partition.
    np.testing.assert_allclose(soft.sum(axis=0), 1.0, atol=1e-4)
    np.testing.assert_array_equal(hard.sum(axis=0), 1.0)
    # hard_assignment uses the first influence, which must be the largest one.
    np.testing.assert_array_equal(skin.weight[:, 0], skin.weight.max(axis=1))
    # Joints grouped as zero-weight carry no skin weight.
    zero_weight = groups["zero_weight"]
    assert not np.isin(skin.index[skin.weight > 0], zero_weight).any()
    # The hands are on their own side (+X = character left).
    for side, sign in (("l_", 1), ("r_", -1)):
        hand = vertices[parts[f"{side}hand"]]
        assert len(hand) > 500
        assert (sign * hand[:, 0] > 30).all()
    assert len(parts["c_head"]) > 1000
