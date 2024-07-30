import os
import json
import re
from pxr import Usd, Gf, Tf, Sdf

def serialize_value(value):
    if isinstance(value, Gf.Matrix4d):
        return [list(value.GetRow(i)) for i in range(4)]
    elif isinstance(value, Gf.Vec3d):
        return [value[0], value[1], value[2]]
    elif isinstance(value, Gf.Quatd):
        return [value.GetReal(), value.GetImaginary()[0], value.GetImaginary()[1], value.GetImaginary()[2]]
    elif isinstance(value, Sdf.TokenListOp):
        return list(value.GetAddedOrExplicitItems())
    elif isinstance(value, Sdf.PathListOp):
        return list(value.GetAddedOrExplicitItems())
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    else:
        return str(value)  # Default case, convert to string for JSON serialization

def extract_prim_attributes(prim):
    attributes = {attr.GetName(): serialize_value(attr.Get()) for attr in prim.GetAttributes()}
    relationships = {rel.GetName(): [str(targetPath) for targetPath in rel.GetTargets()] for rel in prim.GetRelationships()}
    return {
        "name": prim.GetName(),
        "path": prim.GetPath().pathString,
        "type": prim.GetTypeName(),
        "attributes": attributes,
        "relationships": relationships
    }

def extract_fixed_joint_and_transforms(file_path):
    stage = Usd.Stage.Open(file_path)
    if not stage:
        print(f"Failed to open USDZ file: {file_path}")
        return None

    data = []
    site_xform_pattern = re.compile(r'^/World/SiteXform_\d+$')

    for prim in stage.Traverse():
        if site_xform_pattern.match(prim.GetPath().pathString):
            site_xform_info = extract_prim_attributes(prim)
            site_xform_info["children"] = [extract_prim_attributes(child) for child in prim.GetChildren()]
            data.append(site_xform_info)

    return data

def process_usdz_files(input_directory, output_file):
    all_data = {}
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".usdz"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                data = extract_fixed_joint_and_transforms(file_path)
                if data:
                    all_data[file_path] = data

    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)

    print(f"Data extraction complete. Output saved to {output_file}")

if __name__ == "__main__":
    input_directory = "/home/jacob/Downloads/usdz"  # Change to your directory containing USDZ files
    output_file = "extracted_data.json"
    process_usdz_files(input_directory, output_file)
