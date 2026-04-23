from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

ROOT = Path("data/spacenet/SN7_buildings")

SCAN_DIRS = [
    ("train", ROOT / "train" / "train"),
    ("val", ROOT / "val" / "val"),
    ("test_public", ROOT / "test_public"),
]

# train uses .../images/*.tif, test_public often uses .../images_masked/*.tif
IMAGE_DIR_CANDIDATES = ["images", "images_masked"]

PAT = re.compile(r"global_monthly_(\d{4})_(\d{2})_mosaic_(.+)\.tif$")

def main():
    rows = []

    for split, base in SCAN_DIRS:
        if not base.exists():
            continue

        tifs = []
        for img_dir in IMAGE_DIR_CANDIDATES:
            tifs.extend(list(base.rglob(f"{img_dir}/global_monthly_*_mosaic_*.tif")))

        if not tifs:
            print(f"⚠️ No mosaics found in {base}")
            continue

        by_aoi: dict[str, list[tuple[str, Path]]] = {}
        for fp in tifs:
            m = PAT.search(fp.name)
            if not m:
                continue
            year, month, aoi = m.group(1), m.group(2), m.group(3)
            yyyymm = f"{year}_{month}"
            by_aoi.setdefault(aoi, []).append((yyyymm, fp))

        for aoi, items in by_aoi.items():
            items = sorted(items, key=lambda x: x[0])
            for (t0, p0), (t1, p1) in zip(items[:-1], items[1:]):
                rows.append({
                    "split": split,
                    "scene_id": aoi,
                    "t0": t0,
                    "t1": t1,
                    "t0_image": str(p0),
                    "t1_image": str(p1),
                })

    if not rows:
        raise RuntimeError("No image pairs found. Check ROOT/SCAN_DIRS and directory names.")

    out = pd.DataFrame(rows).sort_values(["split", "scene_id", "t0"]).reset_index(drop=True)
    out_path = ROOT / "sn7_imagepair_manifest.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Wrote {len(out)} image pairs -> {out_path}")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()