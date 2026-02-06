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
- **PyTorch Integration**: GPU-accelerated inference for real-time applications
- **[PyMomentum](https://facebookresearch.github.io/momentum/) Integration**: Compatible with fast CPU solver

## Installation

### Option 1. Using Pixi (Recommended)

[Pixi](https://pixi.sh/) is a conda-based package manager. This is the **recommended** installation method as it provides the most reliable environment setup.

```bash
# Clone the repository
git clone git@github.com:facebookresearch/MHR.git
cd MHR

# Download and unzip the model assets
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
unzip assets.zip

# Install dependencies with Pixi
pixi install

# Activate the environment
pixi shell
```

### Option 2. Using the TorchScript model

```bash
# Download the torchscript model
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip

# Unzip torchscript
unzip -p assets.zip assets/mhr_model.pt  > mhr_model.pt

# Start using the torchscript model
```
New to TorchScript model? In short it's a Graph mode of pytorch models. More details [here](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#id3). You can take ./demo.py as a reference to start using the torchscript model.

- Advantage: no codebase or model assets are required.
- Disadvantage: Currently only support for LOD 1; limited access to model properties.

### Option 3. Using pip (⚠️ Experimental)

> **Note:** pip installation is **experimental**. Some dependencies (e.g., `pymomentum`) may not resolve correctly on all platforms. If you encounter issues, please use the recommended Pixi installation above.

```bash
# Install PyMomentum (CPU or GPU)
pip install pymomentum-cpu  # or pymomentum-gpu

# Install MHR
pip install mhr

# Download and unzip the model assets
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
unzip assets.zip
```



### Dependencies

- Python >= 3.11
- PyTorch
- pymomentum >= 0.1.90
- trimesh >= 4.8.3 (Only for demo.py)

## Quick Start

### Run the Demo

```bash
python demo.py
```

This will generate a test MHR mesh and compare outputs with the TorchScript model.

### Visualization Demo

![Visualization Notebook](images/visualization_notebook.png?raw=true)

Interactive Jupyter notebook for MHR visualization. See [`tools/mhr_visualization/README.md`](tools/mhr_visualization/README.md).


### SMPL/SMPL-X Conversion

Conversion between MHR and SMPL/SMPL-X. See [`tools/mhr_smpl_conversion/README.md`](tools/mhr_smpl_conversion/README.md).

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
- **Description**: Facial expression blendshape weights
- **Typical Range**: -1 to +1

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
│   ├── io.py                           # Asset loading utilities
│   ├── mhr.py                          # MHR model implementation
│   └── utils.py                        # Helper functions
├── pyproject.toml                      # Pixi project configuration
├── tests                               # Unit tests
└── tools                               # Additional tools
    ├── mhr_visualization               # Jupyter visualization
    └── mhr_smpl_conversion             # Conversion between MHR and SMPL/SMPL-X
```

## Testing

Run the test suite:

```bash
# Run all tests
pixi run pytest tests/

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
