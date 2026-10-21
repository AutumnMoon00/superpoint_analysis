from pathlib import Path
import numpy as np

from geon.data.boundingbox import BoundingBoxData
from geon.data.cellcomplex import CellComplexData
from geon.data.document import Document
from geon.data.pointcloud import FieldType, PointCloudData


datapath = Path(__file__).resolve().parent.parent / "bridge_data_with_primitives" / "BW-0000_SC-00.h5"

# Load the point cloud and the bridge bounding box layer.
doc = Document.load_hdf5(str(datapath))
bbd = None
ccd = None
pcd = None
for name, item in doc.scene_items.items():
    if isinstance(item, BoundingBoxData):
        bbd = item
        print("Bounding box data name:", name)
    elif isinstance(item, CellComplexData):
        ccd = item
        print("Cell complex data name:", name)
    elif isinstance(item, PointCloudData):
        pcd = item
        print("Point cloud data name:", name)

if bbd is None or ccd is None or pcd is None:
    raise ValueError("The document must contain BBD, CCD, and PCD layers.")

# These extents describe an axis-aligned envelope around the rotated box.
# They are useful to report, but not for clipping to the box itself.
bbox_extents = bbd.get_extents()
if bbox_extents is None:
    raise ValueError("The bounding box layer contains no valid boxes.")

pts = pcd.points
field_names = pcd.field_names



# Transform points into each box's local coordinate system. In geon, the
# bottom center is the origin; x/y span half the width/depth in each direction,
# while z spans from zero to the full height. Process in chunks to limit memory.
inside_mask = np.zeros(len(pts), dtype=bool)
chunk_size = 1_000_000
tolerance = 1e-6
for box in bbd.boxes:
    center = np.asarray(box.center_bottom_xyz, dtype=np.float64)
    rotation = box.rotation_matrix
    half_width = box.width / 2
    half_depth = box.depth / 2
    for start in range(0, len(pts), chunk_size):
        stop = min(start + chunk_size, len(pts))
        local_points = (pts[start:stop].astype(np.float64) - center) @ rotation
        inside_mask[start:stop] |= (
            (np.abs(local_points[:, 0]) <= half_width + tolerance)
            & (np.abs(local_points[:, 1]) <= half_depth + tolerance)
            & (local_points[:, 2] >= -tolerance)
            & (local_points[:, 2] <= box.height + tolerance)
        )
points_inside_bbox = pts[inside_mask]

print("Number of bounding boxes:", bbd.box_count)
print("Bounding box extents:", bbox_extents)
print("Original points:", len(pts))
print("Field names:", field_names)
print("Points inside the rotated bounding box:", len(points_inside_bbox))
print("Clipped points shape:", points_inside_bbox.shape)

print("Points outside box: ", len(pts) - len(points_inside_bbox))


print('\n\n')

# print all the fields and their datatypes
field_data = None
for field in field_names:
    field_data = pcd.get_fields(names=[field])
    print("field: ", field)
    print(field_data)
    print()


# Clip every point field with the same mask, preserving field types and schemas.
clipped_pcd = PointCloudData(points_inside_bbox)
clipped_pcd.id = pcd.id
for field in pcd.get_fields():
    if len(field.data) != len(pts):
        raise ValueError(f"Field {field.name!r} does not match the point count.")
    clipped_pcd.add_field(
        name=field.name,
        data=field.data[inside_mask],
        field_type=field.field_type,
        schema=field.schema if field.field_type == FieldType.SEMANTIC else None,
    )

# Save a geon document with the bridge box, cell complex, and clipped point cloud.
clipped_doc = Document(name=f"{doc.name}_clipped")
clipped_doc.add_data(bbd)
clipped_doc.add_data(ccd)
clipped_doc.add_data(clipped_pcd)
output_dir = datapath.parent / "bbox_clipped"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{datapath.stem}_clipped.h5"
temporary_path = output_path.with_name(f"{output_path.stem}.tmp.h5")
if temporary_path.exists():
    raise FileExistsError(f"Temporary output already exists: {temporary_path}")
clipped_doc.save_hdf5(temporary_path)
temporary_path.replace(output_path)

print("Saved clipped point cloud to:", output_path)
print("Saved points:", len(clipped_pcd.points))
print("Saved fields:", clipped_pcd.field_names)
