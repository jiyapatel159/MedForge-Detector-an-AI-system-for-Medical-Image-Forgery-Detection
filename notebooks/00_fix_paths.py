import os

def explore_deep(path, indent=0, max_depth=4):
    if not os.path.exists(path):
        print(f"{'  '*indent}❌ PATH NOT FOUND: {path}")
        return
    try:
        items = os.listdir(path)
    except:
        return
    for item in sorted(items)[:15]:
        full_path = os.path.join(path, item)
        prefix = "  " * indent + "├── "
        if os.path.isdir(full_path):
            count = len(os.listdir(full_path))
            print(f"{prefix}📁 {item}/  ({count} items)")
            explore_deep(full_path, indent+1, max_depth)
        else:
            size = os.path.getsize(full_path)
            ext = os.path.splitext(item)[1]
            print(f"{prefix}📄 {item}  ({size/1024:.1f} KB) [{ext}]")

print("="*60)
print("TAMPER DATASET:")
print("="*60)
explore_deep("data/raw/Tamper Dataset/")

print("\n" + "="*60)
print("BRAIN MRI:")
print("="*60)
explore_deep("data/raw/Brain MRI/")

print("\n" + "="*60)
print("RSNA (top level):")
print("="*60)
explore_deep("data/raw/rsna-pneumonia-detection-challenge/", max_depth=1)

print("\n" + "="*60)
print("CONVERTED PNG (sample check):")
print("="*60)
png_folder = "data/processed/real_xray_png/"
pngs = os.listdir(png_folder)
print(f"  Total PNGs: {len(pngs)}")
print(f"  Sample files: {pngs[:3]}")