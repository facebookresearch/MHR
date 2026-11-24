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

"""The documented MHR axes match the released model.

+X points to the character's left, +Y up and +Z to the character's front
(right-handed). The real-asset test runs when MHR_TEST_ASSETS names the
release asset folder.
"""

import json
import os
from pathlib import Path

import pytest
import torch

ASSETS = os.environ.get("MHR_TEST_ASSETS")
_NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "mhr_visualization"
    / "MHR visualization.ipynb"
)


def test_notebook_states_the_axes_of_the_model():
    cells = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    text = "".join("".join(cell["source"]) for cell in cells)

    assert (
        "right-handed coordinate system "
        "(+X=character left, +Y=up, +Z=character forward)" in text
    )
    assert "X=right" not in text


@pytest.mark.skipif(ASSETS is None, reason="set MHR_TEST_ASSETS to the asset folder")
def test_rest_pose_matches_the_documented_axes():
    pytest.importorskip("pymomentum.geometry")
    from mhr.mhr import MHR

    model = MHR.from_files(
        ASSETS, device=torch.device("cpu"), lod=1, wants_pose_correctives=False
    )
    with torch.no_grad():
        vertices, skeleton = model(
            torch.zeros(1, model.get_num_identity_blendshapes()),
            torch.zeros(1, 204),
            torch.zeros(1, model.get_num_face_expression_blendshapes()),
        )
    names = model.character.skeleton.joint_names
    joint = {name: skeleton[0, index, :3] for index, name in enumerate(names)}
    vertices = vertices[0]

    assert joint["l_wrist"][0] > 40 and joint["r_wrist"][0] < -40  # +X = left
    assert joint["c_head"][1] > joint["l_upleg"][1] + 50  # +Y = up
    # +Z = front: the most forward vertex of the head is the nose tip.
    head = vertices[vertices[:, 1] > joint["c_head"][1]]
    nose = head[torch.argmax(head[:, 2])]
    assert nose[2] > joint["c_head"][2] + 8 and abs(nose[0]) < 1
