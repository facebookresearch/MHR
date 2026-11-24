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

"""Checks on README.md."""

import re
from pathlib import Path

_README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_links_the_paper_on_arxiv():
    text = _README.read_text(encoding="utf-8")

    assert "arxiv.org/abs/2511.15586" in text
    # No placeholder such as https://arxiv.org/abs/your-arxiv-id.
    for arxiv_id in re.findall(r"arxiv\.org/abs/([^\s)\]>\"'},]+)", text):
        assert re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", arxiv_id), arxiv_id
