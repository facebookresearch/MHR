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


from mhr import FACE_EXPRESSION_NAMES
from mhr.mhr import NUM_FACE_EXPRESSION_BLENDSHAPES


def test_face_expression_names() -> None:
    assert len(FACE_EXPRESSION_NAMES) == NUM_FACE_EXPRESSION_BLENDSHAPES == 72
    assert len(set(FACE_EXPRESSION_NAMES)) == len(FACE_EXPRESSION_NAMES)
    assert FACE_EXPRESSION_NAMES[:4] == (
        "browLowerer_L",
        "browLowerer_R",
        "cheekPuff_L",
        "cheekPuff_R",
    )
    assert FACE_EXPRESSION_NAMES[24] == "jawDrop"
    assert FACE_EXPRESSION_NAMES[40:54] == (
        "lipPucker_L",
        "lipPucker_R",
        "lipStretcher_L",
        "lipStretcher_R",
        "lipSuck_LB",
        "lipSuck_LT",
        "lipSuck_RB",
        "lipSuck_RT",
        "lipTightener_L",
        "lipTightener_R",
        "lipsToward_LB",
        "lipsToward_LT",
        "lipsToward_RB",
        "lipsToward_RT",
    )
    assert FACE_EXPRESSION_NAMES[-2:] == (
        "upperLipRaiser_L",
        "upperLipRaiser_R",
    )
