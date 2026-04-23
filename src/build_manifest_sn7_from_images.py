from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path("/content/drive/MyDrive/deforestation_archetypes")
ROOT = PROJECT_ROOT / "data" / "spacenet" / "SN7_buildings"

PAT = re.compile(r"global_monthly_(\d{4})_(\d{2})_mosaic_(.+)\.tif$")

def collect_monthly_mosaics(split_dir: Path) -> list[Path]:
    """Prefer images_masked; fall back to images if needed."""
    masked = list(split_dir.rglob("images_masked/global_monthly_*_mosaic_*.tif"))
    if masked:
        return masked
    return list(split_dir.rglob("images/global_monthly_*_mosaic_*.tif"))

def main():
    splits = {
        "train": ROOT / "train",
        "test_public": ROOT / "test_public",
    }

    rows = []
    for split, base in splits.items():
        if not base.exists():
            print(f"⚠️ Missing split dir: {base}")
            continue

        tifs = collect_monthly_mosaics(base)
        print(f"{split}: found {len(tifs)} mosaics under {base}")
        if not tifs:
            continue

        by_aoi: dict[str, list[tuple[str, Path]]] = {}
        for fp in tifs:
            m = PAT.search(fp.name)
            if not m:
                continue
            year, month, aoi = m.group(1), m.group(2), m.group(3)
            by_aoi.setdefault(aoi, []).append((f"{year}_{month}", fp))

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
        raise RuntimeError("No mosaics found; check extraction paths.")

    out = pd.DataFrame(rows).sort_values(["split", "scene_id", "t0"]).reset_index(drop=True)
    out_path = ROOT / "sn7_imagepair_manifest.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Wrote {len(out)} pairs → {out_path}")
    print(out.groupby("split").size())
    print(out.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
