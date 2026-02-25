from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh

from mhr.mhr import MHR, LOD


class LODConverter:
    """Converts an MHR mesh from one LOD to another using precomputed barycentric mappings."""

    def __init__(
        self,
        source_lod: LOD = 1,
        mapping_dir: Optional[Path] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self._source_lod = source_lod
        self._device = device
        self._mapping_dir = Path(mapping_dir or Path(__file__).parent)
        self._models: dict[LOD, MHR] = {}
        self._mappings: dict[LOD, dict[str, np.ndarray]] = {}

        self._models[source_lod] = MHR.from_files(lod=source_lod, device=device)

    def _ensure_model(self, target_lod: LOD) -> MHR:
        if target_lod not in self._models:
            self._models[target_lod] = MHR.from_files(lod=target_lod, device=self._device)
        return self._models[target_lod]

    def _ensure_mapping(self, target_lod: LOD) -> dict[str, np.ndarray]:
        if target_lod not in self._mappings:
            path = self._mapping_dir / f"lod{self._source_lod}_to_lod{target_lod}_mapping.npz"
            self._mappings[target_lod] = dict(np.load(path))
        return self._mappings[target_lod]

    def convert(self, src_verts: np.ndarray, target_lod: LOD) -> trimesh.Trimesh:
        """Map vertices from the source LOD onto the target LOD topology.

        Args:
            src_verts: Source mesh vertices, shape ``(num_source_verts, 3)``.
            target_lod: The target LOD to convert to.
        """
        source = self._models[self._source_lod]
        expected_n = len(source.character.mesh.vertices)

        if src_verts.ndim != 2 or src_verts.shape[1] != 3:
            raise ValueError(
                f"src_verts must have shape (N, 3), got {src_verts.shape}"
            )
        if src_verts.shape[0] != expected_n:
            raise ValueError(
                f"src_verts has {src_verts.shape[0]} vertices, "
                f"expected {expected_n} for LOD {self._source_lod}"
            )

        target = self._ensure_model(target_lod)
        mapping = self._ensure_mapping(target_lod)

        src_faces = source.character.mesh.faces
        tri_verts = src_verts[src_faces[mapping["triangle_ids"]]]
        new_verts = np.einsum("ijk,ij->ik", tri_verts, mapping["baryc_coords"])

        return trimesh.Trimesh(new_verts, target.character.mesh.faces, process=False)


if __name__ == "__main__":
    converter = LODConverter(source_lod=1)
    src_verts = converter._models[1].character.mesh.vertices
    mesh = converter.convert(src_verts, target_lod=6)
    mesh.export("/tmp/converted_lod6.ply")
