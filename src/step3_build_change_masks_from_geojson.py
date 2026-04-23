from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import fiona
from functools import lru_cache

PROJECT_ROOT = Path("/content/drive/MyDrive/deforestation_archetypes")

PATCH_INDEX = PROJECT_ROOT / "data" / "processed" / "patch_index.csv"
SN7_ROOT = PROJECT_ROOT / "data" / "spacenet" / "SN7_buildings"

OUT_MASK_ROOT = PROJECT_ROOT / "data" / "processed" / "change_masks"
OUT_INDEX = PROJECT_ROOT / "data" / "processed" / "patch_index_with_masks.csv"

PATCH_SIZE = 512
STRIDE = 512

LABEL_DIR_PREF = ["labels_match_pix", "labels_match", "labels"]  # prefer match_pix if present

def scene_dir_for(split: str, scene: str) -> Path:
    if split == "train":
        return SN7_ROOT / "train" / scene
    if split == "test_public":
        return SN7_ROOT / "test_public" / scene
    return SN7_ROOT / split / scene

def find_buildings_geojson(scene_dir: Path, yyyymm: str) -> Path | None:
    """
    Finds: global_monthly_YYYY_MM_mosaic_<AOI>_Buildings.geojson
    inside preferred label dirs.
    """
    for d in LABEL_DIR_PREF:
        p = scene_dir / d
        if not p.exists():
            continue
        hits = list(p.glob(f"*{yyyymm}*Buildings.geojson"))
        if hits:
            return hits[0]
    return None

@lru_cache(maxsize=256)
def load_geoms(geojson_path_str: str):
    """
    Load geometries from a GeoJSON file (cached).
    Returns list of geometry mappings.
    """
    geojson_path = Path(geojson_path_str)
    geoms = []
    with fiona.open(geojson_path, "r") as src:
        for feat in src:
            g = feat.get("geometry")
            if g:
                geoms.append(g)
    return geoms

def rasterize_window(geoms, window: Window, base_transform) -> np.ndarray:
    """
    Rasterize geometries into a PATCH_SIZE x PATCH_SIZE mask for the given window.
    """
    win_transform = rasterio.windows.transform(window, base_transform)
    if not geoms:
        return np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    mask = rasterize(
        ((g, 1) for g in geoms),
        out_shape=(PATCH_SIZE, PATCH_SIZE),
        transform=win_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,  # conservative; set True if you want thicker footprints
    )
    return mask

def main(max_rows: int | None = None, only_split: str | None = "train"):
    df = pd.read_csv(PATCH_INDEX)
    if only_split is not None:
        df = df[df["split"] == only_split].reset_index(drop=True)
    if max_rows is not None:
        df = df.head(max_rows).reset_index(drop=True)

    kept = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Change masks (GeoJSON→raster)"):
        split = r["split"]
        scene = r["scene_id"]
        t0 = r["t0"]
        t1 = r["t1"]
        rr = int(r["row"])
        cc = int(r["col"])
        p0_img = r["t0_image"] if "t0_image" in r else None

        # We'll use the t0_patch's parent month image path from manifest indirectly:
        # But patch_index.csv doesn't store the original t0_image path.
        # We can safely open the GeoTIFF by reconstructing from the patch paths:
        # However easiest is: read it from patch_index.csv? Not present.
        # So instead: use t0_patch path to infer scene/month and locate the matching mosaic in SN7 folders.
        # We will search for the mosaic under scene/images_masked or images.
        scene_dir = scene_dir_for(split, scene)
        if not scene_dir.exists():
            continue

        # locate mosaic for t0 (prefer images_masked)
        img0 = None
        for imgdir in ["images_masked", "images"]:
            cand = list((scene_dir / imgdir).glob(f"global_monthly_{t0}_mosaic_{scene}.tif"))
            if cand:
                img0 = cand[0]
                break
        if img0 is None:
            continue

        g0 = find_buildings_geojson(scene_dir, t0)
        g1 = find_buildings_geojson(scene_dir, t1)
        if g0 is None or g1 is None:
            # likely test_public or missing labels
            continue

        # Build window
        r0 = rr * STRIDE
        c0 = cc * STRIDE
        window = Window(c0, r0, PATCH_SIZE, PATCH_SIZE)

        # Rasterize t0/t1 building masks on the same grid as the image
        with rasterio.open(img0) as ds:
            base_transform = ds.transform

        m0 = rasterize_window(load_geoms(str(g0)), window, base_transform)
        m1 = rasterize_window(load_geoms(str(g1)), window, base_transform)

        change = ((m1 == 1) & (m0 == 0)).astype(np.uint8)

        out_dir = OUT_MASK_ROOT / split / scene / f"{t0}__{t1}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / f"{rr:04d}_{cc:04d}.npy"
        np.save(out_fp, change)

        rr_out = dict(r)
        rr_out["label_geojson_t0"] = str(g0)
        rr_out["label_geojson_t1"] = str(g1)
        rr_out["mosaic_for_transform"] = str(img0)
        rr_out["change_mask"] = str(out_fp)
        kept.append(rr_out)

    out = pd.DataFrame(kept)
    out.to_csv(OUT_INDEX, index=False)
    print(f"✅ wrote: {OUT_INDEX}")
    print("rows with masks:", len(out))
    if len(out):
        print(out.head(5).to_string(index=False))
        print("splits:", out.groupby("split").size().to_dict())

if __name__ == "__main__":
    main()
