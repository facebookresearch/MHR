# MHR LOD Conversion

Convert MHR mesh vertices between LOD levels using precomputed barycentric mappings.

## Usage

```python
import numpy as np
from example import LODConverter

converter = LODConverter(source_lod=1)

# Provide your own source vertices (shape: (num_verts, 3))
src_verts = np.load("my_vertices.npy")
mesh = converter.convert(src_verts, target_lod=6)
mesh.export("converted_lod6.ply")
```

## Mapping Files

The included `.npz` files contain precomputed barycentric coordinates for converting from LOD 1 to other LOD levels:

| File | Conversion |
|------|------------|
| `lod1_to_lod0_mapping.npz` | LOD 1 → LOD 0 |
| `lod1_to_lod2_mapping.npz` | LOD 1 → LOD 2 |
| `lod1_to_lod3_mapping.npz` | LOD 1 → LOD 3 |
| `lod1_to_lod4_mapping.npz` | LOD 1 → LOD 4 |
| `lod1_to_lod5_mapping.npz` | LOD 1 → LOD 5 |
| `lod1_to_lod6_mapping.npz` | LOD 1 → LOD 6 |

Each mapping file contains:
- `triangle_ids`: Triangle indices on the source mesh for each target vertex.
- `baryc_coords`: Barycentric coordinates within those triangles.
