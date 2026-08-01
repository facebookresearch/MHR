"""Build compact LOD face-index data for the MHR web viewer.

MHR's official barycentric conversion files output FBX control points. General
FBX renderers duplicate those points at UV and normal seams, so this utility
reads the binary FBX control-point topology directly and triangulates it while
preserving the original vertex indices.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO
import zlib

import numpy as np

VIEWER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VIEWER_ROOT.parent
DEFAULT_OUTPUT = VIEWER_ROOT / "data" / "lod_topology.npz"
MAPPING_ROOT = REPO_ROOT / "tools" / "mhr_LOD_conversion"
FBX_HEADER = b"Kaydara FBX Binary  \x00\x1a\x00"


@dataclass
class FbxNode:
    name: str
    properties: list[object]
    children: list["FbxNode"]


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise EOFError("Unexpected end of FBX file")
    return value


def _read_property(stream: BinaryIO) -> object:
    kind = _read_exact(stream, 1)
    scalars: dict[bytes, tuple[str, int]] = {
        b"Y": ("<h", 2),
        b"C": ("<?", 1),
        b"I": ("<i", 4),
        b"F": ("<f", 4),
        b"D": ("<d", 8),
        b"L": ("<q", 8),
    }
    if kind in scalars:
        spec, size = scalars[kind]
        return struct.unpack(spec, _read_exact(stream, size))[0]
    if kind in (b"S", b"R"):
        size = struct.unpack("<I", _read_exact(stream, 4))[0]
        value = _read_exact(stream, size)
        return value.decode("utf-8", errors="replace") if kind == b"S" else value

    arrays: dict[bytes, tuple[str, int]] = {
        b"f": ("<f4", 4),
        b"d": ("<f8", 8),
        b"i": ("<i4", 4),
        b"l": ("<i8", 8),
        b"b": ("u1", 1),
        b"c": ("u1", 1),
    }
    if kind not in arrays:
        raise ValueError(f"Unsupported FBX property type {kind!r}")
    dtype, item_size = arrays[kind]
    length, encoding, payload_size = struct.unpack("<III", _read_exact(stream, 12))
    payload = _read_exact(stream, payload_size)
    if encoding == 1:
        payload = zlib.decompress(payload)
    elif encoding != 0:
        raise ValueError(f"Unsupported FBX array encoding {encoding}")
    expected_size = length * item_size
    if len(payload) != expected_size:
        raise ValueError(
            f"FBX array has {len(payload)} bytes; expected {expected_size}"
        )
    return np.frombuffer(payload, dtype=dtype, count=length).copy()


def _read_node(stream: BinaryIO, version: int) -> FbxNode | None:
    if version >= 7500:
        end_offset, property_count, _, name_length = struct.unpack(
            "<QQQB", _read_exact(stream, 25)
        )
        null_record_size = 25
    else:
        end_offset, property_count, _, name_length = struct.unpack(
            "<IIIB", _read_exact(stream, 13)
        )
        null_record_size = 13
    if end_offset == 0:
        return None

    name = _read_exact(stream, name_length).decode("utf-8", errors="replace")
    properties = [_read_property(stream) for _ in range(property_count)]
    children: list[FbxNode] = []
    child_boundary = end_offset - null_record_size
    while stream.tell() < child_boundary:
        child = _read_node(stream, version)
        if child is None:
            break
        children.append(child)
    stream.seek(end_offset)
    return FbxNode(name, properties, children)


def _walk(nodes: list[FbxNode]):
    for node in nodes:
        yield node
        yield from _walk(node.children)


def _child(node: FbxNode, name: str) -> FbxNode | None:
    return next((child for child in node.children if child.name == name), None)


def _triangulate(polygon_indices: np.ndarray) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    polygon: list[int] = []
    for encoded in polygon_indices:
        value = int(encoded)
        is_last = value < 0
        polygon.append(-value - 1 if is_last else value)
        if is_last:
            if len(polygon) < 3:
                raise ValueError("FBX polygon contains fewer than three vertices")
            faces.extend(
                (polygon[0], polygon[index], polygon[index + 1])
                for index in range(1, len(polygon) - 1)
            )
            polygon.clear()
    if polygon:
        raise ValueError("FBX polygon index array ended mid-polygon")
    return np.asarray(faces, dtype=np.uint32)


def read_control_point_faces(path: Path, expected_vertices: int) -> np.ndarray:
    with path.open("rb") as stream:
        if _read_exact(stream, len(FBX_HEADER)) != FBX_HEADER:
            raise ValueError(f"{path} is not a binary FBX file")
        version = struct.unpack("<I", _read_exact(stream, 4))[0]
        nodes: list[FbxNode] = []
        while True:
            node = _read_node(stream, version)
            if node is None:
                break
            nodes.append(node)

    available: list[int] = []
    for geometry in (node for node in _walk(nodes) if node.name == "Geometry"):
        vertices_node = _child(geometry, "Vertices")
        indices_node = _child(geometry, "PolygonVertexIndex")
        if not vertices_node or not indices_node:
            continue
        vertex_count = len(vertices_node.properties[0]) // 3
        available.append(vertex_count)
        if vertex_count == expected_vertices:
            return _triangulate(indices_node.properties[0])
    raise ValueError(
        f"Could not find {expected_vertices}-vertex geometry in {path}; "
        f"found {available}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output: dict[str, np.ndarray] = {}
    for lod in (0, 2, 3, 4, 5, 6):
        with np.load(MAPPING_ROOT / f"lod1_to_lod{lod}_mapping.npz") as mapping:
            vertex_count = len(mapping["triangle_ids"])
        faces = read_control_point_faces(
            REPO_ROOT / "assets" / f"lod{lod}.fbx", vertex_count
        )
        output[f"lod{lod}_faces"] = faces
        print(f"LOD {lod}: {vertex_count} vertices, {len(faces)} faces")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
