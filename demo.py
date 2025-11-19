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


import torch
from mhr.mhr import MHR
import trimesh

torch.manual_seed(0)

def _prepare_input_data(batch_size: int) -> torch.Tensor:
    identity_coeffs = 0.8 * torch.randn(batch_size, 45).cpu()
    model_parameters = 0.2 * (torch.rand(batch_size, 204) - 0.5).cpu()
    face_expr_coeffs = 0.3 * torch.randn(batch_size, 72).cpu()
    return identity_coeffs, model_parameters, face_expr_coeffs

def run():
    mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)
    batch_size = 2
    identity_coeffs, model_parameters, face_expr_coeffs = _prepare_input_data(batch_size)

    with torch.no_grad():
        verts, skel_state = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)

    mesh = trimesh.Trimesh(vertices=verts[0].numpy(), faces=mhr_model.character.mesh.faces, process=False)
    output_mesh_path = "./test.ply"
    mesh.export(output_mesh_path)
    print(f"Saved example MHR mesh to {output_mesh_path}")

def compare_with_torchscript_model():
    print("Comparing MHR model with TorchScripted model.")
    scripted_model = torch.jit.load("./assets/mhr_model.pt")
    mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)

    batch_size = 128
    identity_coeffs, model_parameters, face_expr_coeffs = _prepare_input_data(batch_size)

    with torch.no_grad():
        verts, _ = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)
        verts_ts, _ = scripted_model(identity_coeffs, model_parameters, face_expr_coeffs)
        print(f"Averge per-vertex offsets {torch.abs(verts - verts_ts).mean()} cm.")
        print(f"Max per-vertex offsets {torch.abs(verts - verts_ts).max()} cm.")

if __name__ == "__main__":
    run()
    compare_with_torchscript_model()
