# MHR - Momentum Human Rig

A minimal Python package for the Momentum Human Rig - a parametric 3D human body model with identity, pose, and facial expression parameterization.

[![arXiv](https://img.shields.io/badge/arXiv-2511.15586-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2511.15586)

## Overview

![MHR teaser](images/teaser.jpg?raw=true)

MHR (Momentum Human Rig) is a high-fidelity 3D human body model that provides:

- **Identity Parameterization**: 45 shape parameters controlling body identity
- **Pose Parameterization**: 204 model parameters for full-body articulation
- **Facial Expression**: 72 expression parameters for detailed face animation
- **Multiple LOD Levels**: 7 levels of detail (LOD 0-6) for different performance requirements
- **Non-linear Pose Correctives**: Neural network-based pose-dependent deformations
- **PyTorch Integration**: Differentiable inference and optimization on CPU or GPU
- **[PyMomentum](https://facebookresearch.github.io/momentum/) Integration**: Compatible with fast CPU solver

## Installation

### Option 1. Using Pixi (Recommended)

[Pixi](https://pixi.sh/) is a conda-based package manager. This is the **recommended** installation method as it provides the most reliable environment setup.

```bash
# Clone the repository
git clone git@github.com:facebookresearch/MHR.git
cd MHR

# Install dependencies with Pixi
pixi install

# Download and unzip the model assets
pixi run download-assets

# Activate the environment
pixi shell
```

### Option 2. Using the TorchScript model

```bash
# Install MHR
pip install mhr

# Download the torchscript model
mhr-download-assets --member assets/mhr_model.pt --output mhr_model.pt

# Start using the torchscript model
```
New to TorchScript model? In short it's a Graph mode of pytorch models. More details [here](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#id3). You can take ./demo.py as a reference to start using the torchscript model.

- Advantage: no codebase or model assets are required.
- Disadvantage: Currently only support for LOD 1; limited access to model properties.

> **Python 3.14:** loading the TorchScript model works there with PyTorch 2.14 and gives the same output as on Python 3.12, but PyTorch marks TorchScript as not guaranteed to work on Python 3.14 ([PyTorch 2.10 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.10.0)). On Python 3.14, prefer `MHR.from_files` (Option 1).

### Option 3. Using pip (⚠️ Experimental)

> **Note:** pip installation stays **experimental**, mainly because of the Linux crash below, which needs a PyMomentum release on PyPI to fix. CI installs MHR with pip and checks the model on Ubuntu 22.04 and 24.04, macOS 15 (Apple silicon), and Windows with Python 3.12 and 3.13, and on Rocky Linux 9 with Python 3.12 ([pip install workflow](https://github.com/facebookresearch/MHR/actions/workflows/pip.yml)). If you encounter issues, please use the recommended Pixi installation above.
>
> PyMomentum's PyPI wheels cover Python 3.12 and 3.13 only. On Python 3.14, use Pixi or conda-forge `pymomentum` until PyMomentum publishes 3.14 wheels.

```bash
# Install MHR with PyMomentum (pymomentum-core, which works with any recent PyTorch)
pip install "mhr[momentum]"

# Download and unzip the model assets into ./assets
mhr-download-assets
```

> **Known issue on Linux:** the PyPI builds of PyMomentum ship a private copy of `libstdc++`. Depending on which PyTorch build loads the system copy first, the two copies can mix, and then `MHR.from_files` crashes with a segmentation fault ([#44](https://github.com/facebookresearch/MHR/issues/44)). No PyMomentum release on PyPI fixes this yet. It happens with `pymomentum-cpu`, which requires PyTorch 2.8, and with the CPU-only PyTorch wheels from download.pytorch.org, on every Linux distribution that CI tests. `pymomentum-core` with the default PyTorch from PyPI does not crash there, and macOS and Windows are not affected. If you hit the crash, use the Pixi environment or conda-forge `pymomentum`, or start Python with PyMomentum's copy preloaded:
>
> ```bash
> LD_PRELOAD=$(python -c "import glob, sysconfig; print(glob.glob(sysconfig.get_paths()['purelib'] + '/pymomentum_*.libs/libstdc++*.so*')[0])") python demo.py
> ```



### Option 4. Using pip without PyMomentum (from v1.2.0)

From release v1.2.0 on, MHR also runs with PyTorch alone, from converted assets that each release publishes per LOD.

```bash
# Install MHR
pip install mhr

# Download the converted assets for LOD 1 into ./assets (about 25 MB)
mhr-download-assets --lod 1
```

```python
import torch
from mhr import MHR  # needs no PyMomentum

mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)
```

`mhr.MHR` is the model for new code (PyTorch only). `mhr.mhr.MHR` is the PyMomentum-backed API that the tools use; it subclasses `mhr.MHR`. Both give the same vertices, skeleton states and gradients, and each loads the other's saved state. `mhr.mhr` and the tools that use PyMomentum directly (SMPL conversion, glTF export) still need PyMomentum.

### Dependencies

- Python >= 3.12 (CI tests 3.12, 3.13 and 3.14 with Pixi; pip needs Python 3.12 or 3.13, see Option 3)
- PyTorch
- pymomentum >= 0.1.89, for `mhr.mhr` and the tools (CI tests 0.1.114)
- trimesh >= 4.8.3 (for additional tools only: mhr-smpl conversion, LOD conversion and segmentation)

## Quick Start

### Run the Demo

```bash
python demo.py
```

This will generate a test MHR mesh and compare outputs with the TorchScript model.

The demo saves the mesh with trimesh. The Pixi environment includes it; with a pip install, run `pip install "trimesh>=4.8.3,<6"` first.

### Visualization Demo

![Visualization Notebook](images/visualization_notebook.png?raw=true)

[Interactive Jupyter notebook](https://colab.research.google.com/github/facebookresearch/MHR/blob/main/tools/mhr_visualization/MHR%20visualization.ipynb) for MHR visualization. See [`tools/mhr_visualization/README.md`](tools/mhr_visualization/README.md).


### SMPL/SMPL-X Conversion

Conversion between MHR and SMPL/SMPL-X. See [`tools/mhr_smpl_conversion/README.md`](tools/mhr_smpl_conversion/README.md).

### glTF Export

Export MHR poses and animations to glTF (`.glb`) for Blender and other tools. See [`tools/mhr_export/README.md`](tools/mhr_export/README.md).

### Basic Usage

```python
import torch
from mhr.mhr import MHR

# Load MHR model (LOD 1, on CPU)
mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)

# Define parameters
batch_size = 2
identity_coeffs = 0.8 * torch.randn(batch_size, 45)      # Identity
model_parameters = 0.2 * (torch.rand(batch_size, 204) - 0.5)  # Pose
face_expr_coeffs = 0.3 * torch.randn(batch_size, 72)     # Facial expression

# Generate mesh vertices and skeleton information (joint orientation and positions).
vertices, skeleton_state = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)
```

`MHR.from_files()` uses the first of these folders that contains the assets: `assets` in a source checkout, `$MHR_ASSETS_DEST/assets`, then `./assets`, which is where `mhr-download-assets` extracts them. To use another folder, pass `folder="path/to/assets"`.

Without PyMomentum, use `from mhr import MHR` with the converted assets instead (see Option 4); it takes the same arguments.

### Computing gradients

Both the Python and TorchScript models support standard PyTorch autograd.
Enable gradients only on the inputs you want to optimize before the forward
pass, and keep gradient mode enabled; do not wrap the call in `torch.no_grad()`
or `torch.inference_mode()`:

```python
identity_coeffs.requires_grad_()
model_parameters.requires_grad_()
face_expr_coeffs.requires_grad_()

vertices, skeleton_state = mhr_model(
    identity_coeffs,
    model_parameters,
    face_expr_coeffs,
)
loss = vertices.square().mean()
loss.backward()
```

The calls in `demo.py` use `torch.no_grad()` because that script performs
inference only.

## Model Parameters

### Identity Parameters (`identity_coeffs`)
- **Shape**: `[batch_size, 45]`
- **Description**: The first 20 control body shape identity, second 20 control head, and the last 5 for hands.
- **Typical Range**: -3 to +3 (zero-mean, unit variance)

### Model Parameters (`model_parameters`)
- **Shape**: `[batch_size, 204]`
- **Description**: Joint angles and scalings

### Expression Parameters (`face_expr_coeffs`)
- **Shape**: `[batch_size, 72]`
- **Description**: Artist-sculpted, sparse semantic blendshape weights that follow FACS. These are not PCA components or a one-to-one list of FACS Action Units.
- **Typical Range**: -1 to +1
- **Names and indices**: See the [facial expression mapping](docs/face-expressions.md), or import `FACE_EXPRESSION_NAMES` from `mhr`.

## Project Structure

```
MHR/
├── assets                              # Assets (downloaded and unzipped from release)
│   ├── compact_v6_1.model              # Model parameterization
│   ├── corrective_activation.npz       # Pose corrective MLP sparse activations
│   ├── corrective_blendshapes_lod?.npz # Pose corrective blendshapes
│   ├── lod?.fbx                        # Rig with identity and expression blendshapes
│   └── mhr_model.pt                    # Torchscript model
├── demo.py                             # Basic demo script
├── mhr                                 # Main package
│   ├── _model.py                       # MHR model on PyTorch alone (from mhr import MHR)
│   ├── io.py                           # Asset loading utilities
│   ├── mhr.py                          # MHR model loaded with PyMomentum
│   └── utils.py                        # Helper functions
├── pyproject.toml                      # Pixi project configuration
├── tests                               # Unit tests
└── tools                               # Additional tools
    ├── mhr_export                      # glTF export
    ├── mhr_visualization               # Jupyter visualization
    └── mhr_smpl_conversion             # Conversion between MHR and SMPL/SMPL-X
```

## Testing

Run the test suite:

```bash
# Run all tests: tests/ tests the mhr package, scripts/tests/ the maintainer scripts
pixi run test

# Run specific test
pixi run pytest tests/test_mhr.py
```

## Inferring MHR parameters from images

If you want to do Human Motion Recovery with MHR, head to [Sam3D](https://github.com/facebookresearch/sam-3d-body).

## Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## Code of Conduct

Please read our [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) before contributing.

## Citation

If you use MHR in your research, please cite:

```bibtex
@misc{MHR:2025,
      title={MHR: Momentum Human Rig},
      author={Aaron Ferguson and Ahmed A. A. Osman and Berta Bescos and Carsten Stoll and Chris Twigg and Christoph Lassner and David Otte and Eric Vignola and Fabian Prada and Federica Bogo and Igor Santesteban and Javier Romero and Jenna Zarate and Jeongseok Lee and Jinhyung Park and Jinlong Yang and John Doublestein and Kishore Venkateshan and Kris Kitani and Ladislav Kavan and Marco Dal Farra and Matthew Hu and Matthew Cioffi and Michael Fabris and Michael Ranieri and Mohammad Modarres and Petr Kadlecek and Rawal Khirodkar and Rinat Abdrashitov and Romain Prévost and Roman Rajbhandari and Ronald Mallet and Russell Pearsall and Sandy Kao and Sanjeev Kumar and Scott Parrish and Shoou-I Yu and Shunsuke Saito and Takaaki Shiratori and Te-Li Wang and Tony Tung and Yichen Xu and Yuan Dong and Yuhua Chen and Yuanlu Xu and Yuting Ye and Zhongshi Jiang},
      year={2025},
      eprint={2511.15586},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2511.15586},
}
```

## License

MHR is licensed under the Apache Software License 2.0, as found in the [LICENSE](LICENSE) file.
