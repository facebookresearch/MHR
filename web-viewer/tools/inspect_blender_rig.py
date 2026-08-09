"""Inspect a GLB or USD rig after importing it into Blender."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import bpy


def _arguments() -> tuple[bool, list[str]]:
    if "--" not in sys.argv:
        raise SystemExit("Pass one or more model paths after --.")
    values = sys.argv[sys.argv.index("--") + 1 :]
    disable_glb_bone_shapes = "--disable-glb-bone-shapes" in values
    paths = [value for value in values if value != "--disable-glb-bone-shapes"]
    return disable_glb_bone_shapes, paths


def _vector(values: object) -> list[float]:
    return [float(value) for value in values]


def inspect_file(
    path: Path,
    *,
    disable_glb_bone_shapes: bool = False,
) -> dict[str, object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if path.suffix.lower() == ".glb":
        bpy.ops.import_scene.gltf(
            filepath=str(path),
            disable_bone_shape=disable_glb_bone_shapes,
        )
    elif path.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}:
        bpy.ops.wm.usd_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported file: {path}")

    armatures = []
    for armature_object in (
        item for item in bpy.data.objects if item.type == "ARMATURE"
    ):
        bones = []
        for bone in armature_object.data.bones:
            bones.append(
                {
                    "name": bone.name,
                    "parent": bone.parent.name if bone.parent else None,
                    "head": _vector(bone.head_local),
                    "tail": _vector(bone.tail_local),
                    "length": float(bone.length),
                }
            )
        bones.sort(key=lambda item: item["length"], reverse=True)
        custom_shapes = [
            pose_bone.custom_shape.name
            for pose_bone in armature_object.pose.bones
            if pose_bone.custom_shape is not None
        ]
        armatures.append(
            {
                "name": armature_object.name,
                "displayType": armature_object.data.display_type,
                "boneCount": len(bones),
                "customShapeCount": len(custom_shapes),
                "customShapeObjectNames": sorted(set(custom_shapes)),
                "longestBones": bones[:12],
            }
        )

    meshes = []
    for mesh_object in (item for item in bpy.data.objects if item.type == "MESH"):
        meshes.append(
            {
                "name": mesh_object.name,
                "vertexCount": len(mesh_object.data.vertices),
                "modifiers": [
                    {
                        "name": modifier.name,
                        "type": modifier.type,
                        "object": (
                            modifier.object.name
                            if getattr(modifier, "object", None)
                            else None
                        ),
                    }
                    for modifier in mesh_object.modifiers
                ],
                "vertexGroupCount": len(mesh_object.vertex_groups),
            }
        )

    empties = [
        {
            "name": item.name,
            "displayType": item.empty_display_type,
            "displaySize": float(item.empty_display_size),
        }
        for item in bpy.data.objects
        if item.type == "EMPTY"
    ]
    return {
        "file": str(path),
        "disableGlbBoneShapes": disable_glb_bone_shapes,
        "objects": len(bpy.data.objects),
        "armatures": armatures,
        "meshes": meshes,
        "emptyCount": len(empties),
        "empties": empties[:12],
    }


def main() -> None:
    disable_glb_bone_shapes, paths = _arguments()
    reports = [
        inspect_file(
            Path(value).resolve(),
            disable_glb_bone_shapes=disable_glb_bone_shapes,
        )
        for value in paths
    ]
    print("MHR_RIG_INSPECTION=" + json.dumps(reports, separators=(",", ":")))


if __name__ == "__main__":
    main()
