import re
from pxr import Usd

usd_path = "/home/jacob/Downloads/bowl2sink.usdz"

usd_stage = Usd.Stage.Open(usd_path)

default_prim = usd_stage.GetDefaultPrim()

# Regular expression to match SiteXform_*
site_xform_regex = re.compile(r'^/World/SiteXform_\d+$')

def print_prim_details(prim):
    print(f"Prim Name: {prim.GetName()}")
    print(f"Prim Path: {prim.GetPath()}")
    attributes = prim.GetAttributes()
    for attr in attributes:
        print(f"Attribute: {attr.GetName()} - Value: {attr.Get()}")
    children = prim.GetChildren()
    for child in children:
        print(f"  Child Prim Name: {child.GetName()}")
        print(f"  Child Prim Path: {child.GetPath()}")
        if 'fixed_joint' in child.GetName():
            print("  Fixed Joint found:")
            print_prim_details(child)

for prim in usd_stage.Traverse():
    if site_xform_regex.match(prim.GetPath().pathString):
        # Unsure what we should do here, but I just printed to test.
        print_prim_details(prim)