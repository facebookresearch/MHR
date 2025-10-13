# SMPL-PROTO Conversion Tool

A Python toolkit for converting between SMPL/SMPLX and PROTO body model representations. This tool provides bidirectional conversion capabilities with multiple optimization backends and identity handling options.

## Overview

The SMPL-PROTO Conversion Tool enables seamless conversion between different 3D human body model formats:

- **SMPL → PROTO**: Convert SMPL model parameters/meshes to PROTO format
- **SMPLX → PROTO**: Convert SMPLX model parameters/meshes to PROTO format
- **PROTO → SMPL/SMPLX**: Convert PROTO parameters back to SMPL/SMPLX format

The tool uses barycentric interpolation for topology mapping and offers multiple optimization backends for parameter fitting.

## Features

### 🔄 **Bidirectional Conversion**
- SMPL/SMPLX parameters → PROTO parameters
- SMPL/SMPLX vertices → PROTO vertices
- PROTO parameters → SMPL/SMPLX parameters
- PROTO vertices → SMPL/SMPLX vertices

### ⚙️ **Multiple Optimization Methods**
- **PyMomentum**: CPU-based hierarchical optimization with robust fitting
- **PyTorch**: GPU-accelerated optimization with edge and vertex loss

### 🎭 **Identity Handling**
- **Single Identity**: Use consistent shape parameters across all frames
- **Multiple Identities**: Unique shape parameters for each frame

### 📊 **Output Options**
- Export meshes in PLY format
- Return parameter dictionaries
- Generate vertex arrays
- Compute fitting error metrics

## Installation

### Prerequisites

```bash
# On top of PROTO
pixi add --pypi trimesh scikit-learn tqdm smplx
```

### SMPL/SMPLX Model Files

You'll need the official SMPL/SMPLX model files:

1. **SMPL**: Download from [SMPL website](https://smpl.is.tue.mpg.de/)
2. **SMPLX**: Download from [SMPLX website](https://smpl-x.is.tue.mpg.de/)

**Note**: For SMPL models, ensure you have a chumpy-free `.pkl` file or the current official `.npz` file. The tool can automatically convert `.npz` to compatible `.pkl` format.

## Quick Start

### Run Examples

```bash
pixi run python example.py --smpl path/to/smpl/model.pkl --smplx path/to/smplx/model.pkl -o output_dir
```

### Programmatic Usage

```python
import torch
from proto.proto import PROTO
from smpl_proto import Conversion
import smplx

# Initialize models
proto_model = PROTO.from_files(lod=1, device="cuda")
smplx_model = smplx.SMPLX(model_path="path/to/smplx", gender="neutral")

# Create converter
converter = Conversion(
    proto_model=proto_model,
    smpl_model=smplx_model,
    method="pytorch"  # or "pymomentum"
)

# Convert SMPLX to PROTO
results = converter.convert_smpl2proto(
    smpl_parameters=smplx_params,
    single_identity=True,
    return_proto_meshes=True,
    return_proto_parameters=True
)

# Convert PROTO back to SMPLX
smplx_results = converter.convert_proto2smpl(
    proto_parameters=results.result_parameters,
    return_smpl_meshes=True
)
```

## API Reference

### `Conversion` Class

The main conversion class that handles all transformations between SMPL/SMPLX and PROTO formats.

#### Constructor

```python
Conversion(proto_model, smpl_model, method="pytorch")
```

**Parameters:**
- `proto_model` (PROTO): PROTO body model instance
- `smpl_model` (smplx.SMPLX | smplx.SMPL): SMPL or SMPLX model instance
- `method` (str): Optimization method ("pytorch" or "pymomentum")

#### `convert_smpl2proto()`

Convert SMPL/SMPLX data to PROTO format.

**Parameters:**
- `smpl_vertices` (torch.Tensor, optional): SMPL vertex positions [B, V, 3]
- `smpl_parameters` (dict, optional): SMPL parameter dictionary
- `single_identity` (bool): Use single identity across frames (default: True)
- `is_tracking` (bool): Use temporal tracking for optimization (default: False)
- `return_proto_meshes` (bool): Return mesh objects (default: False)
- `return_proto_parameters` (bool): Return parameter dictionary (default: False)
- `return_proto_vertices` (bool): Return vertex arrays (default: False)
- `return_fitting_errors` (bool): Return fitting error metrics (default: True)

**Returns:**
- `ConversionResult`: Object containing requested outputs

#### `convert_proto2smpl()`

Convert PROTO data to SMPL/SMPLX format.

**Parameters:**
- `proto_vertices` (torch.Tensor, optional): PROTO vertex positions [B, V, 3]
- `proto_parameters` (dict, optional): PROTO parameter dictionary
- `single_identity` (bool): Use single identity across frames (default: True)
- `is_tracking` (bool): Use temporal tracking for optimization, only for SMPL(X)-> PROTO with PyMomentum (default: False)
- `return_smpl_meshes` (bool): Return mesh objects (default: False)
- `return_smpl_parameters` (bool): Return parameter dictionary (default: False)
- `return_smpl_vertices` (bool): Return vertex arrays (default: False)

**Returns:**
- `ConversionResult`: Object containing requested outputs

### `ConversionResult` Class

Container for conversion results.

**Attributes:**
- `result_meshes` (list[trimesh.Trimesh]): Generated mesh objects
- `result_vertices` (np.ndarray): Vertex positions [B, V, 3]
- `result_parameters` (dict): Model parameter dictionary
- `result_errors` (np.ndarray): Per-frame fitting errors


## Optimization Methods

### PyMomentum Backend

- **Pros**: Robust hierarchical optimization, make use of temporal consistency (is_tracking=True)
- **Cons**: CPU-only, may be slower for large batch of temporally inconsistent data. The identity is the average identity across the first sequential processing, not optimized across the whole sequence.

**Features:**
- Hierarchical optimization stages
- Automatic failure case reprocessing
- Temporal tracking support

### PyTorch Backend

- **Pros**: GPU acceleration, faster processing
- **Cons**: Currently process each frame independently, no temporal consistency is leveraged.
- **Best for**: Large-scale convertion of independent poses.

**Features:**
- Edge + vertex loss combination
- Batch processing
