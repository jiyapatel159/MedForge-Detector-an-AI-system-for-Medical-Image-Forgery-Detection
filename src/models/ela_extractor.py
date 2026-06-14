import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tqdm import tqdm

# ─────────────────────────────────────────
def generate_ela_image(image_path, quality=90):
    """
    Error Level Analysis:
    - Save image at lower JPEG quality
    - Measure pixel difference vs original
    - Tampered regions show HIGHER error (brighter in ELA)
    """
    try:
        original = Image.open(image_path).convert('RGB')

        # Save at reduced quality to temp file
        temp_path = "temp_ela_check.jpg"
        original.save(temp_path, 'JPEG', quality=quality)

        # Reload compressed version
        compressed = Image.open(temp_path)

        # Pixel-level difference
        ela_image = ImageChops.difference(original, compressed)

        # Amplify difference to make it visible
        extrema   = ela_image.getextrema()
        max_diff  = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1

        scale     = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        # Cleanup temp
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return ela_image

    except Exception as e:
        print(f"  ⚠️  ELA failed for {image_path}: {e}")
        return None


# ─────────────────────────────────────────
def batch_generate_ela(input_folder, output_folder, quality=90):
    """Generate ELA maps for all images in a folder"""

    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    success, failed = 0, 0

    for fname in tqdm(image_files, desc=f"ELA → {os.path.basename(output_folder)}"):
        src_path = os.path.join(input_folder, fname)
        # Save ELA as PNG always
        dst_name = os.path.splitext(fname)[0] + "_ela.png"
        dst_path = os.path.join(output_folder, dst_name)

        # Skip if already done
        if os.path.exists(dst_path):
            success += 1
            continue

        ela = generate_ela_image(src_path, quality=quality)
        if ela is not None:
            ela.save(dst_path)
            success += 1
        else:
            failed += 1

    print(f"  ✅ Done → success: {success} | failed: {failed}")
    return success, failed


# ─────────────────────────────────────────
if __name__ == "__main__":

    # All 6 folders that need ELA maps
    folders = [
        # (input,                              output)
        ("data/processed/train/real",   "data/processed/train_ela/real"),
        ("data/processed/train/forged", "data/processed/train_ela/forged"),
        ("data/processed/val/real",     "data/processed/val_ela/real"),
        ("data/processed/val/forged",   "data/processed/val_ela/forged"),
        ("data/processed/test/real",    "data/processed/test_ela/real"),
        ("data/processed/test/forged",  "data/processed/test_ela/forged"),
    ]

    total_success = 0
    total_failed  = 0

    print("="*55)
    print("GENERATING ELA MAPS FOR ALL SPLITS")
    print("="*55)

    for inp, out in folders:
        print(f"\n📁 Processing: {inp}")
        s, f = batch_generate_ela(inp, out)
        total_success += s
        total_failed  += f

    print("\n" + "="*55)
    print("ELA GENERATION COMPLETE")
    print("="*55)
    print(f"  Total success : {total_success:,}")
    print(f"  Total failed  : {total_failed:,}")
    print(f"  ELA maps saved to: data/processed/[split]_ela/")
    print("\n✅ Ready for model training!") 