# MHR glTF Export

Export MHR parameters to a glTF binary (`.glb`) file that Blender and other
glTF viewers can open.

## Installation

The tool needs [PyMomentum](https://facebookresearch.github.io/momentum/). The
repository's pixi environment includes it. With pip, install it on top of MHR,
and see the known Linux issue under "Option 3. Using pip" in the
[main README](../../README.md):

```bash
pip install pymomentum-core
```

Download the MHR assets first (see the main README).

## Usage

```bash
# Mean shape in the rest pose
python tools/mhr_export/export_gltf.py --output mhr.glb

# An animation, one frame per row of the input parameters
python tools/mhr_export/export_gltf.py --input motion.npz --output motion.glb --fps 30 --lod 1
```

`motion.npz` holds the inputs of `MHR.forward`:

| Key | Shape | Required |
|-----|-------|----------|
| `model_parameters` | `[frames, 204]` | yes |
| `identity_coeffs` | `[45]`, `[1, 45]` (same shape in every frame) or `[frames, 45]` | yes |
| `face_expr_coeffs` | `[frames, 72]` | no (zeros) |

```python
import numpy as np

np.savez(
    "motion.npz",
    model_parameters=model_parameters.cpu().numpy(),
    identity_coeffs=identity_coeffs.cpu().numpy(),
    face_expr_coeffs=face_expr_coeffs.cpu().numpy(),
)
```

From Python, `load_character`, `build_motion` and `save_gltf` do the same
steps:

```python
import sys

sys.path.append("tools/mhr_export")  # from the repository root
from export_gltf import build_motion, load_character, save_gltf

character = load_character(assets_folder, lod=1)
motion = build_motion(character, model_parameters, identity_coeffs, face_expr_coeffs)
save_gltf(character, "motion.glb", motion, fps=30)
```

## What is exported

- The skinned mesh of the chosen LOD and the 127-joint skeleton.
- The 45 identity and 72 face expression blendshapes as morph targets.
- One animation keyframe per input frame: joint transforms and morph target
  weights.
- Meters, Y-up, as glTF defines.

## What is not exported

- **Pose correctives.** glTF has no equivalent of the MHR pose corrective
  network, so the exported mesh matches `MHR.forward(...,
  apply_correctives=False)`. On random poses, the correctives move vertices
  by 1-4 mm on average and by up to 1-5 cm near strongly bent joints.
- **FBX.** pymomentum writes FBX only when built with the Autodesk FBX SDK,
  which the pip and conda-forge packages do not include.

## Blender

Import with **File > Import > glTF 2.0**. The mesh has 117 shape keys, one per
blendshape. Blender's importer sets each shape key's slider range from the
first keyframe only and clamps later weights outside that range (for example
negative face expression weights). Widen the ranges after the import:

```python
import bpy

for obj in bpy.context.selected_objects:
    if obj.type == "MESH" and obj.data.shape_keys:
        for key in obj.data.shape_keys.key_blocks:
            key.slider_min, key.slider_max = -10.0, 10.0
```

Blender places glTF keyframes by time, so set the scene frame rate
(**Output Properties > Frame Rate**) to the export's `--fps` before the import;
each exported frame then lands on a whole Blender frame. The import leaves the
scene's frame range at Blender's default (1-250), which starts after the first
exported frame (frame 0) and holds the last pose long after the animation
ends. Set the range to the animation's frames after the import:

```python
import math

import bpy

scene = bpy.context.scene
for obj in bpy.context.selected_objects:
    if obj.type == "ARMATURE" and obj.animation_data and obj.animation_data.action:
        start, end = obj.animation_data.action.frame_range
        scene.frame_start, scene.frame_end = math.floor(start), math.ceil(end)
```
