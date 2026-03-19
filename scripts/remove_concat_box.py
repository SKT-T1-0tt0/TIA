#!/usr/bin/env python3
"""Remove the concatenation box from the MCFL diagram (Conditioning column, top)."""
import sys
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Need: pip install Pillow numpy")
    sys.exit(1)

def main():
    base = Path(__file__).resolve().parent.parent
    default_path = base / ".cursor/projects/root-autodl-tmp-TIA2V/assets/c__Users___AppData_Roaming_Cursor_User_workspaceStorage_aac6adad80161847fcf3b54062fe86a1_images_Generated_image-f48459e5-4103-4350-9771-85896cc1885a.png"
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    if not src.exists():
        print(f"Image not found: {src}")
        sys.exit(1)

    img = Image.open(src).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Conditioning column = right third; concat box = top of that column
    # Fill (x0:x1, y0:y1) with median background color from image edges
    x0, x1 = int(0.60 * w), w
    y0, y1 = 0, int(0.32 * h)

    # Sample background from dark corners/edges
    samples = [
        arr[10, 10], arr[h - 10, 10], arr[10, w - 10],
        arr[h // 2, x0 - 30],
    ]
    fill = np.median(samples, axis=0).astype(np.uint8)

    arr[y0:y1, x0:x1] = fill

    out = src.parent / (src.stem + "_no_concat_box" + src.suffix)
    Image.fromarray(arr).save(out, "PNG")
    print(f"Saved: {out}")
    return str(out)

if __name__ == "__main__":
    main()
