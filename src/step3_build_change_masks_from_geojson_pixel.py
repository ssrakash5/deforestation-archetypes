from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import fiona
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.transform import Affine
from functools import lru_cache

PROJECT_ROOT = Path("/content/drive/MyDrive/deforestation_archetypes")
PATCH_INDEX = PROJECT_ROOT / "data" / "processed" / "patch_index.csv"
SN7_ROOT = PROJECT_ROOT / "data" / "spacenet" / "SN7_buildings"

OUT_MASK_ROOT = PROJECT_ROOT / "data" / "processed" / "change_masks_pixel"
OUT_INDEX = PROJECT_ROOT / "data" / "processed" / "patch_index_with_masks_pixel.csv"

PATCH_SIZE = 512
STRIDE = 512

LABEL_DIR_PREF = ["labels_match_pix", "labels_match", "labels"]

def scene_dir_for(split: str, scene: str) -> Path:
    if split == "train":
        return SN7_ROOT / "train" / scene
    if split == "test_public":
        return SN7_ROOT / "test_public" / scene
    return SN7_ROOT / split / scene

def find_buildings_geojson(scene_dir: Path, yyyymm: str) -> Path | None:
    for d in LABEL_DIR_PREF:
        p = scene_dir / d
        if not p.exists():
            continue
        hits = list(p.glob(f"*{yyyymm}*Buildings.geojson"))
        if hits:
            return hits[0]
    return None

def find_mosaic(scene_dir: Path, scene: str, yyyymm: str) -> Path | None:
    for imgdir in ["images_masked", "images"]:
        cand = list((scene_dir / imgdir).glob(f"global_monthly_{yyyymm}_mosaic_{scene}.tif"))
        if cand:
            return cand[0]
    return None

@lru_cache(maxsize=512)
def load_geoms(path_str: str):
    geoms = []
    with fiona.open(path_str, "r") as src:
        for feat in src:
            g = feat.get("geometry")
            if g:
                geoms.append(g)
    return geoms

@lru_cache(maxsize=512)
def rasterize_full_mask(geojson_path: str, width: int, height: int) -> np.ndarray:
    """
    Rasterize polygons assuming GeoJSON coords are in pixel units (x=col, y=row).
    We use transform = Affine.identity() so x,y map directly to pixel space.
    """
    geoms = load_geoms(geojson_path)
    if not geoms:
        return np.zeros((height, width), dtype=np.uint8)

    # Pixel-space transform: x=col, y=row
    transform = Affine.translation(0, 0) * Affine.scale(1, 1)

    mask = rasterize(
        ((g, 1) for g in geoms),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    return mask

def main(only_split: str = "train", max_rows: int | None = None):
    df = pd.read_csv(PATCH_INDEX)
    df = df[df["split"] == only_split].reset_index(drop=True)
    if max_rows is not None:
        df = df.head(max_rows).reset_index(drop=True)

    kept = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Change masks (pixel-geojson)"):
        split = r["split"]
        scene = r["scene_id"]
        t0 = r["t0"]
        t1 = r["t1"]
        rr = int(r["row"])
        cc = int(r["col"])

        scene_dir = scene_dir_for(split, scene)
        if not scene_dir.exists():
            continue

        g0 = find_buildings_geojson(scene_dir, t0)
        g1 = find_buildings_geojson(scene_dir, t1)
        if g0 is None or g1 is None:
            continue

        # get raster size from mosaic
        mosaic0 = find_mosaic(scene_dir, scene, t0)
        if mosaic0 is None:
            continue

        with rasterio.open(mosaic0) as ds:
            H, W = ds.height, ds.width

        # rasterize full masks in pixel coordinates, then window-slice
        m0_full = rasterize_full_mask(str(g0), W, H)
        m1_full = rasterize_full_mask(str(g1), W, H)

        r0 = rr * STRIDE
        c0 = cc * STRIDE
        m0 = m0_full[r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        m1 = m1_full[r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        if m0.shape != (PATCH_SIZE, PATCH_SIZE) or m1.shape != (PATCH_SIZE, PATCH_SIZE):
            continue

        # Start with NEW-ONLY (like deforestation analog)
        change = ((m1 == 1) & (m0 == 0)).astype(np.uint8)

        out_dir = OUT_MASK_ROOT / split / scene / f"{t0}__{t1}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / f"{rr:04d}_{cc:04d}.npy"
        np.save(out_fp, change)

        rr_out = dict(r)
        rr_out["label_geojson_t0"] = str(g0)
        rr_out["label_geojson_t1"] = str(g1)
        rr_out["mosaic_for_size"] = str(mosaic0)
        rr_out["change_mask"] = str(out_fp)
        kept.append(rr_out)

    out = pd.DataFrame(kept)
    out.to_csv(OUT_INDEX, index=False)
    print(f"✅ wrote: {OUT_INDEX}")
    print("rows with masks:", len(out))

if __name__ == "__main__":
    main()
