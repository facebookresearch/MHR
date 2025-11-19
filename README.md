# MHR - Momentum Human Rig

A minimal Python package for the Momentum Human Rig - a parametric 3D human body model with identity, pose, and facial expression parameterization.

## Overview

MHR (Momentum Human Rig) is a high-fidelity 3D human body model that provides:

- **Identity Parameterization**: 45 shape parameters controlling body identity
- **Pose Parameterization**: 204 model parameters for full-body articulation
- **Facial Expression**: 72 expression parameters for detailed face animation
- **Multiple LOD Levels**: 7 levels of detail (LOD 0-6) for different performance requirements
- **Non-linear Pose Correctives**: Neural network-based pose-dependent deformations
- **PyTorch Integration**: GPU-accelerated inference for real-time applications
- **PyMomentum Integration**: Compatible to fast CPU solver

## Installation

### Option 1. Using the torchscript model (Recommended)
<span style="color: red;">To be completed.</span>
```bash
# Download the torchscipt model
**TODO**

# Start using the torchscript model
```
New to TorchScript model? In short it's a Graph mode of pytorch models. More details [here](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#id3). You can take ./demo.py as a reference to start using th torchscript model.

- Advantage: no codebase or model assets are required.
- Disadvantage: Currently only support for LOD 1; limited access to model properties.

### Option 2. Using Pixi (Recommended)
<span style="color: red;">To be completed.</span>
```bash
# After cloning the repository
cd MHR

# Download the model assets
**TODO**

# Install dependencies with Pixi
pixi install

# Activate the environment
pixi shell
```


### Option 3. Using pip
<span style="color: red;">To be completed.</span>
```bash
# Pip install
pip install mhr .

# Download the model assets
**TODO**
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

## Tools

### Visualization

Interactive Jupyter notebook for MHR visualization. See [`tools/mhr_visualization/README.md`](tools/mhr_visualization/README.md).


## Project Structure

```
MHR/
├── assets/                      # Model assets
│   ├── rig_lod*.fbx            # Rig files for each LOD
│   ├── corrective_blendshapes_lod*.npz  # Blendshapes
│   ├── corrective_activation.npz        # None-linear pose correctives
│   └── model_definition.model           # Model parameterization
├── mhr/                         # Main package
│   ├── __init__.py
│   ├── mhr.py                  # MHR model implementation
│   ├── io.py                   # Asset loading utilities
│   └── utils.py                # Helper functions
├── tools/                       # Additional tools
│   ├── mhr_visualization/      # Jupyter visualization
├── tests/                       # Unit tests
├── demo.py                      # Basic demo script
└── pyproject.toml              # Project configuration
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_mhr.py

# Run with coverage
pytest --cov=mhr tests/
```

## Inferring MHR parameters from images

If you want to do Human Motion Recovery with MHR, head to [Sam3D][https://github.com/facebookresearch/sam-3d-body].

## Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## Code of Conduct

Please read our [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) before contributing.

## Citation

If you use MHR in your research, please cite:
<span style="color: red;">To be completed, add the tech report.</span>
```bibtex
}
@inproceedings{MHR:2025,
	author    = {Ferguson, Aaron and Osman, Ahmed A. A. and Bescos, Berta and Stoll, Carsten and Twigg, Chris and Lassner, Christoph and Otte, David and Vignola, Eric and Bogo, Federica and Santesteban, Igor and Romero, Javier and Zarate, Jenna and Lee, Jeongseok and Park, Jinhyung and Yang, Jinlong and Doublestein, John and Venkateshan, Kishore and Kitani, Kris and Kavan, Ladislav and Dal Farra, Marco and Hu, Matthew and Cioffi, Matthew and Fabris, Michael and Ranieri, Michael and Modarres, Mohammad and Kadlecek, Petr and Khirodkar, Rawal and Abdrashitov, Rinat and Prévost, Romain and Rajbhandari, Roman and Mallet, Ronald and Pearsall, Russel and Kao, Sandy and Kumar, Sanjeev and Parrish, Scott and Saito, Shunsuke and Wang, Te-Li and Tung, Tony and Dong, Yuan and Chen, Yuhua and Xu, Yuanlu and Ye, Yuting and Jiang, Zhongshi},
	title     = {MHR: Momentum Human Rig},
	booktitle = {Tech Report},
	year      = {2025},
	url       = {https://arxiv.org/abs/your-arxiv-id}
}
```

## License

MHR is licensed under the Apache Software License 2.0, as found in the [LICENSE](LICENSE) file.
