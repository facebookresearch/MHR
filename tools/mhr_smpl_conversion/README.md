# SMPL-MHR Conversion Tool

A Python toolkit for converting between SMPL/SMPLX and MHR body model representations. This tool provides bidirectional conversion capabilities with multiple optimization backends and identity handling options.

## Overview

The SMPL-MHR Conversion Tool enables seamless conversion between different 3D human body model formats:

- **SMPL/SMPLX → MHR**: Convert SMPL/SMPLX model parameters/meshes to MHR format
- **MHR → SMPL/SMPLX**: Convert MHR parameters back to SMPL/SMPLX format
- **SAM3D Outputs (MHR) → SMPL/SMPLX**: Convert SAM3D output to SMPL/SMPLX model parameters

The tool uses barycentric interpolation for topology mapping and offers multiple optimization backends for parameter fitting.

## Features

### 🔄 **Bidirectional Conversion**
- SMPL/SMPLX parameters or vertices → MHR parameters
- MHR parameters or vertices → SMPL/SMPLX parameters

### ⚙️ **Multiple Optimization Methods**
- **PyMomentum**: CPU-based Gauss-Newton hierarchical optimization with robust fitting
- **PyTorch**: GPU-accelerated optimization with edge and vertex loss

### 🎭 **Identity Handling**
- **Single Identity**: Use consistent shape parameters across all frames
- **Multiple Identities**: Unique shape parameters for each frame

### 📊 **Output Options**
- Return meshes in trimesh.Trimesh format
- Return parameter dictionaries
- Return vertex as numpy arrays
- Return fitting errors

## Installation

### Prerequisites

```bash
# On top of MHR
pixi add pymomentum scikit-learn tqdm "trimesh>=4.8.3,<5"
pixi add --pypi smplx==0.1.28
```

The conversion tool runs from an MHR source checkout. It is not part of the
`mhr` PyPI or conda-forge packages.

### SMPL/SMPLX Model Files

You'll need the official SMPL/SMPLX model files:

1. **SMPL**: Download from the [SMPL website](https://smpl.is.tue.mpg.de/).
   Models with the standard 6890-vertex SMPL mesh are supported.
2. **SMPL-X**: Download the `.npz` model from the
   [SMPL-X website](https://smpl-x.is.tue.mpg.de/).

`example.py` loads the SMPL model with `smpl_model.load_smpl_model`, which
accepts:

- `.npz` files with `posedirs` and `shapedirs` shaped `(6890, 3, components)`.
- `.npz` files with these arrays flattened to `(components, 20670)` or
  `(20670, components)`. The 20670 values can be ordered vertex by vertex
  (`x0, y0, z0, x1, ...`) or coordinate by coordinate (`x0, x1, ..., y0, ...`).
  The loader detects the order from how smoothly the offsets vary over the
  mesh. If the data do not decide it, loading stops with an error; state the
  order with `--smpl-blendshape-layout vertex_major` or
  `--smpl-blendshape-layout coordinate_major`.
- Chumpy-free `.pkl` files, which are passed to `smplx` unchanged.

`.pkl` files that contain Chumpy objects cannot be loaded in the supported
Python 3.12/3.13 environments; use an `.npz` file instead.

## Quick Start

### Run Examples

```bash
cd tools/mhr_smpl_conversion
pixi run python example.py \
  --smpl path/to/smpl/model.npz \
  --smplx path/to/smplx/model.npz \
  -o output_dir
```
Three conversions will be conducted and the results will be exported in three folders under the output_dir. The results folders are named as {source_model}\_{input_format}2{target_model}\_{method}(_{if_single_identity_sequence}):

```
mhr_smpl_conversion/
├── output_dir
│   ├── smpl_para2mhr_pymomentum
│   ├── mhr_para2smplx_pytorch_single_identity
│   ├── smplx_mesh2mhr_pytorch_single_identity
│   ├── sam3d_output_to_smplx
└──...
```

Meshes of the source and the target model can be found under each conversion results folder.

### Programmatic Usage

```python
import torch
from mhr.mhr import MHR
from smpl_mhr import Conversion
import smplx

# Initialize models
mhr_model = MHR.from_files(lod=1, device="cuda")
smplx_model = smplx.SMPLX(model_path="path/to/smplx", gender="neutral")

# Create converter
converter = Conversion(
    mhr_model=mhr_model,
    smpl_model=smplx_model,
    method="pytorch"  # or "pymomentum"
)

# Convert SAM3D output to SMPL(X)
results = converter.convert_sam3d_output_to_smplx(
    sam3d_outputs=sam3d_outputs,
    return_smpl_meshes=True,
    return_smpl_parameters=True,
    return_fitting_errors=True
)

# Convert SMPLX to MHR
results = converter.convert_smpl2mhr(
    smpl_parameters=smplx_params,
    single_identity=True,
    return_mhr_meshes=True,
    return_mhr_parameters=True
)

# Convert MHR back to SMPLX
smplx_results = converter.convert_mhr2smpl(
    mhr_parameters=results.result_parameters,
    return_smpl_meshes=True
)
```

## API Reference

### `Conversion` Class

The main conversion class that handles all transformations between SMPL/SMPLX and MHR formats.

#### Constructor

```python
Conversion(mhr_model, smpl_model, method="pytorch")
```

**Parameters:**
- `mhr_model` (MHR): MHR body model instance
- `smpl_model` (smplx.SMPLX | smplx.SMPL): SMPL or SMPLX model instance
- `method` (str): Optimization method ("pytorch" or "pymomentum")

#### `convert_smpl2mhr()`

Convert SMPL/SMPLX data to MHR format.

**Parameters:**
- `smpl_vertices` (torch.Tensor, optional): SMPL vertex positions [B, V, 3]
- `smpl_parameters` (dict, optional): SMPL parameter dictionary
- `single_identity` (bool): Use single identity across frames (default: True)
- `is_tracking` (bool): Use temporal tracking for optimization (default: False)
- `return_mhr_meshes` (bool): Return mesh objects (default: False)
- `return_mhr_parameters` (bool): Return parameter dictionary (default: False)
- `return_mhr_vertices` (bool): Return vertex arrays (default: False)
- `return_fitting_errors` (bool): Return fitting error metrics (default: True)

**Returns:**
- `ConversionResult`: Object containing requested outputs

#### `convert_mhr2smpl()`

Convert MHR data to SMPL/SMPLX format.

**Parameters:**
- `mhr_vertices` (torch.Tensor, optional): MHR vertex positions [B, V, 3]
- `mhr_parameters` (dict, optional): MHR parameter dictionary
- `single_identity` (bool): Use single identity across frames (default: True)
- `is_tracking` (bool): Use temporal tracking for optimization, only for SMPL(X)-> MHR with PyMomentum (default: False)
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
- **Best for**: Large-scale conversion of independent poses.

**Features:**
- Edge + vertex loss combination
- Batch processing
