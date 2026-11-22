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


"""PyMomentum helpers that moved between releases."""

import unittest

try:
    from pymomentum.geometry_test_utils import create_test_character
except ModuleNotFoundError as error:
    if error.name != "pymomentum.geometry_test_utils":
        raise
    import pymomentum.geometry as _geometry

    if hasattr(_geometry, "create_test_character"):  # older PyMomentum releases
        create_test_character = _geometry.create_test_character
    else:
        # The PyPI wheels of recent releases (pymomentum-cpu, -gpu and -core
        # 0.1.114) ship neither helper. Only the tests that need a test
        # character skip; the pip CI job checks the real model against the
        # TorchScript model instead.
        def create_test_character(*args, **kwargs):
            raise unittest.SkipTest(
                "this PyMomentum build has no create_test_character"
            )


__all__ = ["create_test_character"]
