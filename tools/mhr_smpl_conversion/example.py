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

"""
pixi run python example.py --smpl path/to/smpl/model.pkl --smplx path/to/smplx/model.pkl -o output_dir
"""


import argparse
import os

import numpy as np
import smplx

import torch
import trimesh
from mhr.mhr import MHR
from conversion import Conversion

_INPUT_FILE = "./data/example_smplx_poses.npy"  # Directory to store input data
_SAM3D_BODY_OUTPUT_DIR = (
    "./data/sam3d_body_outputs"  # Directory to store SAM3D body output
)
_OUTPUT_DIR = "./tmp_results"  # Directory to store conversion results


class DEMO:
    def __init__(self):
        self.smpl_model = None
        self.smplx_model = None
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = torch.device("cpu")

    def run_examples(
        self,
        input_smplx_pose_file: str,
        output_dir: str,
        smpl_model_file: str | None,
        smplx_model_file: str | None,
    ):
        if smpl_model_file is not None:
            # Unfortunately, SMPL model .pkl file may come with Chumpy, which is not compatible with
            # latest Python versions. Although the latest official SMPL model in .npz format is chumpy
            # free, the default smplx package does not support .npz file as SMPL model file.
            # So please provide either a chumpy-free SMPL model .pkl file or the official .npz file,
            # from which, a chumpy-free SMPL model .pkl file can be created.
            try:
                self.smpl_model = smplx.SMPL(
                    model_path=smpl_model_file,
                )
            except:
                print(
                    "If the provided SMPL model file is a .pkl file, please make sure it is chumpy free."
                )
                if smpl_model_file.endswith(".npz"):
                    converted_smpl_model_file = smpl_model_file.replace(
                        ".npz", "_generate_from_npz.pkl"
                    )
                    if not os.path.exists(converted_smpl_model_file):
                        smpl_model_data = dict(np.load(smpl_model_file))
                        import pickle

                        with open(converted_smpl_model_file, "wb") as f:
                            pickle.dump(smpl_model_data, f)
                    self.smpl_model = smplx.SMPL(
                        model_path=converted_smpl_model_file,
                    )
        if smplx_model_file is not None:
            self.smplx_model = smplx.SMPLX(
                model_path=smplx_model_file,
                gender="neutral",
                use_pca=False,
                flat_hand_mean=True,
            ).to(self._device)
        smpl_paras, smplx_paras = self._get_parameter_data(input_smplx_pose_file)
        os.makedirs(output_dir, exist_ok=True)

        # Example 1: Convert the SMPL parameters to MHR with PyMomentum and multiple identities
        if self.smpl_model is not None:
            print(
                "\nConverting SMPL parameters to MHR with PyMomentum and multiple identities."
            )
            example_output_dir = output_dir + "/smpl_para2mhr_pymomentum"
            os.makedirs(example_output_dir, exist_ok=True)
            mhr_vertices = (
                self.example_smpl_parameters_to_mhr_with_pymomentum_multiple_identity(
                    smpl_paras, example_output_dir
                )
            )

        # Example 2: Convert the SMPLX meshes to MHR with PyTorch and single identity
        if self.smplx_model is not None:
            print("\nConverting SMPLX meshes to MHR with PyTorch and single identity.")
            example_output_dir = output_dir + "/smplx_mesh2mhr_pytorch_single_identity"
            os.makedirs(example_output_dir, exist_ok=True)
            mhr_parameters = (
                self.example_smplx_meshes_to_mhr_with_pytorch_single_identity(
                    smplx_paras, example_output_dir
                )
            )

        # Example 3: Convert the MHR parameters to SMPLX with a single identity
        if self.smplx_model is not None:
            print("\nConverting MHR parameters to PyTorch with a single identity.")
            example_output_dir = output_dir + "/mhr_para2smplx_pytorch_single_identity"
            os.makedirs(example_output_dir, exist_ok=True)
            self.example_mhr_parameters_to_smplx_pytorch_single_identity(
                mhr_parameters, example_output_dir
            )

        # Example 4: Convert the SAM3D output to SMPLX
        if self.smplx_model is not None:
            print("\nConverting SAM3D output to SMPLX.")
            example_output_dir = output_dir + "/sam3d_output_to_smplx"
            os.makedirs(example_output_dir, exist_ok=True)
            self.example_sam3d_output_to_smplx(example_output_dir)

    def example_sam3d_output_to_smplx(self, output_dir: str):
        """Convert SAM3D output to SMPLX."""
        mhr_model = MHR.from_files(lod=1, device=self._device)

        # Load the SAM3D outputs
        subject_files = [
            f for f in os.listdir(_SAM3D_BODY_OUTPUT_DIR) if f.endswith(".npz")
        ]
        sam3d_outputs = []

        for subject_file in subject_files:
            subject_data = np.load(os.path.join(_SAM3D_BODY_OUTPUT_DIR, subject_file))
            sam3d_outputs.append(subject_data)
            mesh = trimesh.Trimesh(
                subject_data["pred_vertices"] + subject_data["pred_cam_t"][None, ...],
                mhr_model.character.mesh.faces,
                process=False,
            )
            mesh.export(f"{output_dir}/{subject_file[:-4]}_sam3d.ply")

        # ***** Core conversion code *****
        converter = Conversion(
            mhr_model=mhr_model, smpl_model=self.smplx_model, method="pytorch"
        )

        conversion_results = converter.convert_sam3d_output_to_smpl(
            sam3d_outputs=sam3d_outputs,
            return_smpl_meshes=True,
            return_smpl_parameters=False,
            return_smpl_vertices=False,
            return_fitting_errors=True,
        )

        print("Conversion errors:")
        print(conversion_results.result_errors)

        # Save the results
        print(f"Exporting the results to {output_dir}...")
        for i, mesh in enumerate(conversion_results.result_meshes):
            mesh.export(f"{output_dir}/{subject_files[i][:-4]}_result_smplx.ply")

    def example_smpl_parameters_to_mhr_with_pymomentum_multiple_identity(
        self, smpl_paras: dict[str, torch.Tensor], output_dir: str
    ):
        """Convert SMPL parameters to MHR."""
        # Compute the SMPLX meshes with multiple identities and export them.
        num_frames = smpl_paras["body_pose"].shape[0]
        for i in range(num_frames):
            smplx_vertices = (
                self.smpl_model(
                    global_orient=smpl_paras["global_orient"][i : i + 1],
                    body_pose=smpl_paras["body_pose"][i : i + 1],
                    betas=smpl_paras["betas"][i : i + 1],
                )
                .vertices.detach()
                .cpu()
                .numpy()[0]
            )
            smplx_mesh = trimesh.Trimesh(
                smplx_vertices, self.smpl_model.faces, process=False
            )
            smplx_mesh.export(f"{output_dir}/{i:03d}_smpl.ply")

        # ***** Core conversion code *****
        mhr_model = MHR.from_files(lod=1, device=self._device)
        converter = Conversion(
            mhr_model=mhr_model, smpl_model=self.smpl_model, method="pymomentum"
        )
        conversion_results = converter.convert_smpl2mhr(
            smpl_vertices=None,
            smpl_parameters=smpl_paras,
            single_identity=False,
            return_mhr_meshes=True,
            return_mhr_vertices=True,
            return_mhr_parameters=False,
            return_fitting_errors=True,
        )
        # ********************************

        print("Conversion errors:")
        print(conversion_results.result_errors)

        # Save the results
        for i, mesh in enumerate(conversion_results.result_meshes):
            mesh.vertices /= 100.0
            mesh.export(f"{output_dir}/{i:03d}_result_mhr.ply")

        return conversion_results.result_vertices

    def example_smplx_meshes_to_mhr_with_pytorch_single_identity(
        self, smplx_paras: dict, output_dir: str
    ):
        """Convert SMPL(X) meshes to MHR format."""
        # Compute the SMPLX meshes with single identity and export them.
        input_smpl_vertices = []
        for i in range(smplx_paras["body_pose"].shape[0]):
            smplx_vertices = self.smplx_model(
                global_orient=smplx_paras["global_orient"][i : i + 1],
                body_pose=smplx_paras["body_pose"][i : i + 1],
                betas=smplx_paras["betas"][0:1],
                expression=smplx_paras["expression"][i : i + 1],
                left_hand_pose=smplx_paras["left_hand_pose"][i : i + 1],
                right_hand_pose=smplx_paras["right_hand_pose"][i : i + 1],
            ).vertices.detach()
            input_smpl_vertices.append(smplx_vertices)
            smplx_mesh = trimesh.Trimesh(
                smplx_vertices.cpu().numpy()[0], self.smplx_model.faces, process=False
            )
            smplx_mesh.export(f"{output_dir}/{i:03d}_smplx.ply")
        input_smpl_vertices = torch.stack(input_smpl_vertices, dim=0)

        # ***** Core conversion code *****
        mhr_model = MHR.from_files(lod=1, device=self._device)
        converter = Conversion(
            mhr_model=mhr_model, smpl_model=self.smplx_model, method="pytorch"
        )
        conversion_results = converter.convert_smpl2mhr(
            smpl_vertices=input_smpl_vertices,
            smpl_parameters=None,
            single_identity=True,
            return_mhr_meshes=True,
            return_mhr_vertices=False,
            return_mhr_parameters=True,
            return_fitting_errors=True,
        )
        # ********************************

        print("Conversion errors:")
        print(conversion_results.result_errors)

        # Save the results
        for i, mesh in enumerate(conversion_results.result_meshes):
            mesh.vertices /= 100.0
            mesh.export(f"{output_dir}/{i:03d}_result_mhr.ply")

        return conversion_results.result_parameters

    def example_mhr_parameters_to_smplx_pytorch_single_identity(
        self, mhr_parameters: dict, output_dir: str
    ):
        """Convert MHR parameters to PyTorch."""
        mhr_model = MHR.from_files(lod=1, device=self._device)
        converter = Conversion(
            mhr_model=mhr_model, smpl_model=self.smplx_model, method="pytorch"
        )

        # Compute MHR meshes and export them.
        mhr_meshes, _ = converter._mhr_para2mesh(mhr_parameters, return_mesh=True)
        for i, mesh in enumerate(mhr_meshes):
            mesh.vertices /= 100.0
            mesh.export(f"{output_dir}/{i:03d}_mhr.ply")

        conversion_results = converter.convert_mhr2smpl(
            mhr_vertices=None,
            mhr_parameters=mhr_parameters,
            single_identity=True,
            return_smpl_meshes=True,
            return_smpl_parameters=False,
            return_smpl_vertices=False,
            return_fitting_errors=True,
        )

        print("Conversion errors:")
        print(conversion_results.result_errors)

        # Save the results
        for i, mesh in enumerate(conversion_results.result_meshes):
            mesh.export(f"{output_dir}/{i:03d}_result_smplx.ply")

    def _get_parameter_data(self, input_smplx_poses_file: str):
        """Get SMPL(X) data for examples."""
        # Load the SMPL(X) parameters
        smplx_full_poses = np.load(input_smplx_poses_file)
        num_frames = smplx_full_poses.shape[0]

        smplx_parameters, smpl_parameters = None, None
        # Generate the SMPL(X) parameters by adding random betas and expression coefficients.
        if self.smplx_model is not None:
            smplx_parameters = {
                "global_orient": np.zeros((num_frames, 3)),
                "body_pose": smplx_full_poses[:, 3:66],
                "left_hand_pose": smplx_full_poses[:, -90:-45],
                "right_hand_pose": smplx_full_poses[:, -45:],
                "betas": np.random.randn(1, self.smplx_model.num_betas).repeat(
                    num_frames, axis=0
                ),
                "expression": np.random.randn(
                    1, self.smplx_model.num_expression_coeffs
                ).repeat(num_frames, axis=0),
            }
            for k, v in smplx_parameters.items():
                smplx_parameters[k] = (
                    torch.from_numpy(v).to(torch.float32).to(self._device)
                )

        if self.smpl_model is not None:
            smpl_parameters = {
                "global_orient": np.zeros((num_frames, 3)),
                "body_pose": np.concatenate(
                    [smplx_full_poses[:, 3:66], np.zeros_like(smplx_full_poses[:, :6])],
                    axis=-1,
                ),
                "betas": np.random.randn(num_frames, 10),
            }
            for k, v in smpl_parameters.items():
                smpl_parameters[k] = (
                    torch.from_numpy(v).to(torch.float32).to(self._device)
                )

        return smpl_parameters, smplx_parameters


def _parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert SMPL(X) parameters/meshes into MHR format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--smpl",
        type=str,
        required=False,
        default="./data/SMPL_NEUTRAL.pkl",
        help="Path to the SMPL model file. If not specified, SMPL related conversion will be skipped.",
    )

    parser.add_argument(
        "--smplx",
        type=str,
        required=False,
        default="./data/SMPLX_NEUTRAL.npz",
        help="Path to the SMPL-X model file. If not specified, SMPL-X related conversion will be skipped.",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=False,
        default=_INPUT_FILE,
        help="Path to the input .npy file containing SMPLX pose parameters.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default=_OUTPUT_DIR,
        help="Path to the output directory where results will be saved.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()
    demo = DEMO()
    demo.run_examples(args.input, args.output, args.smpl, args.smplx)
