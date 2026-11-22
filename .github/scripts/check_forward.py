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

"""Check an installed MHR against the TorchScript model from the same assets.

Usage: python check_forward.py

Run it from the directory where mhr-download-assets extracted ./assets, so
that MHR.from_files() finds the assets the way it does for users.
"""

import torch
from mhr.io import get_default_asset_folder
from mhr.mhr import MHR

# A wrong PyMomentum build gives offsets of centimeters; matching builds stay
# below 1e-3 cm.
MAX_OFFSET_CM = 1e-2


def main() -> None:
    torch.manual_seed(0)
    batch_size = 16
    identity = 0.8 * torch.randn(batch_size, 45)
    pose = 0.2 * (torch.rand(batch_size, 204) - 0.5)
    expression = 0.3 * torch.randn(batch_size, 72)

    model = MHR.from_files(device=torch.device("cpu"), lod=1)
    assets = get_default_asset_folder()
    print(f"assets {assets}")
    reference = torch.jit.load(str(assets / "mhr_model.pt"))
    with torch.no_grad():
        verts, _ = model(identity, pose, expression)
        verts_ref, _ = reference(identity, pose, expression)

    offset = (verts - verts_ref).abs()
    print(f"vertices {tuple(verts.shape)}")
    print(
        f"offset from TorchScript: mean {offset.mean():.3g} cm, "
        f"max {offset.max():.3g} cm"
    )
    if offset.max() >= MAX_OFFSET_CM:
        raise SystemExit("MHR does not match the TorchScript model")


if __name__ == "__main__":
    main()
