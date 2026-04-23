from __future__ import annotations
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import rasterio
from rasterio.windows import Window

PROJECT_ROOT = Path("/content/drive/MyDrive/deforestation_archetypes")

MANIFEST = PROJECT_ROOT / "data" / "spacenet" / "SN7_buildings" / "sn7_imagepair_manifest.csv"
OUT_PATCH_ROOT = PROJECT_ROOT / "data" / "processed" / "patches"
OUT_INDEX = PROJECT_ROOT / "data" / "processed" / "patch_index.csv"
OUT_STATS = PROJECT_ROOT / "data" / "processed" / "stats.json"

PATCH_SIZE = 512
STRIDE = 512

# How many bands to keep (SpaceNet mosaics are often 3-band RGB, sometimes more)
# We'll keep first 3 by default for simplicity.
KEEP_BANDS = 3

def _compute_global_stats(sample_paths: list[str], max_patches: int = 200) -> dict:
    """
    Compute simple mean/std on a small sample of patches (first KEEP_BANDS),
    ignoring NaNs. This is for normalization later.
    """
    rng = np.random.default_rng(1337)
    paths = sample_paths.copy()
    rng.shuffle(paths)

    sums = np.zeros(KEEP_BANDS, dtype=np.float64)
    sqs  = np.zeros(KEEP_BANDS, dtype=np.float64)
    n    = np.zeros(KEEP_BANDS, dtype=np.float64)

    used = 0
    for p in paths:
        if used >= max_patches:
            break
        with rasterio.open(p) as ds:
            h, w = ds.height, ds.width
            if h < PATCH_SIZE or w < PATCH_SIZE:
                continue
            # random window
            r = int(rng.integers(0, h - PATCH_SIZE + 1))
            c = int(rng.integers(0, w - PATCH_SIZE + 1))
            arr = ds.read(indexes=list(range(1, min(KEEP_BANDS, ds.count) + 1)),
                          window=Window(c, r, PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
            # arr: (C,H,W)
            for ch in range(arr.shape[0]):
                x = arr[ch].reshape(-1)
                x = x[np.isfinite(x)]
                if x.size == 0:
                    continue
                sums[ch] += float(x.sum())
                sqs[ch]  += float((x * x).sum())
                n[ch]    += float(x.size)
            used += 1

    mean = sums / np.maximum(n, 1.0)
    var = (sqs / np.maximum(n, 1.0)) - (mean * mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return {"mean": mean.tolist(), "std": std.tolist(), "patch_size": PATCH_SIZE, "stride": STRIDE, "keep_bands": KEEP_BANDS}

def _tile_one_image(image_path: str, out_dir: Path) -> list[tuple[int,int,Path]]:
    """
    Tiles one GeoTIFF into PATCH_SIZE windows with STRIDE.
    Saves each tile as .npy (C,H,W) float32.
    Returns list of (row, col, saved_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    with rasterio.open(image_path) as ds:
        H, W = ds.height, ds.width
        nrows = math.floor((H - PATCH_SIZE) / STRIDE) + 1
        ncols = math.floor((W - PATCH_SIZE) / STRIDE) + 1
        if nrows <= 0 or ncols <= 0:
            return saved

        band_count = min(KEEP_BANDS, ds.count)
        band_indexes = list(range(1, band_count + 1))

        for rr in range(nrows):
            r0 = rr * STRIDE
            for cc in range(ncols):
                c0 = cc * STRIDE
                window = Window(c0, r0, PATCH_SIZE, PATCH_SIZE)
                arr = ds.read(indexes=band_indexes, window=window).astype(np.float32)  # (C,H,W)
                # Save
                fp = out_dir / f"{rr:04d}_{cc:04d}.npy"
                np.save(fp, arr)
                saved.append((rr, cc, fp))

    return saved

def main(max_pairs: int | None = 50, splits: list[str] | None = None):
    df = pd.read_csv(MANIFEST)

    if splits:
        df = df[df["split"].isin(splits)].reset_index(drop=True)

    if max_pairs is not None:
        df = df.head(max_pairs).reset_index(drop=True)

    # compute stats from a small sample of images in this run
    sample_paths = df["t0_image"].tolist()[: min(len(df), 200)]
    stats = _compute_global_stats(sample_paths, max_patches=200)
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print("✅ wrote stats:", OUT_STATS)

    index_rows = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Tiling pairs"):
        split = row["split"]
        scene = row["scene_id"]
        t0 = row["t0"]
        t1 = row["t1"]
        p0 = row["t0_image"]
        p1 = row["t1_image"]

        out0 = OUT_PATCH_ROOT / split / scene / t0
        out1 = OUT_PATCH_ROOT / split / scene / t1

        tiles0 = _tile_one_image(p0, out0)
        tiles1 = _tile_one_image(p1, out1)

        # assume same tiling grid; index by rr,cc
        set1 = {(rr,cc): fp for rr,cc,fp in tiles1}
        for rr, cc, fp0 in tiles0:
            fp1 = set1.get((rr,cc))
            if fp1 is None:
                continue
            patch_id = f"{split}__{scene}__{t0}__{t1}__{rr:04d}_{cc:04d}"
            index_rows.append({
                "patch_id": patch_id,
                "split": split,
                "scene_id": scene,
                "t0": t0,
                "t1": t1,
                "row": rr,
                "col": cc,
                "t0_patch": str(fp0),
                "t1_patch": str(fp1),
            })

    out_index = pd.DataFrame(index_rows)
    out_index.to_csv(OUT_INDEX, index=False)
    print("✅ wrote patch index:", OUT_INDEX)
    print(out_index.head(5).to_string(index=False))
    print("Total patches:", len(out_index))

if __name__ == "__main__":
    # default: small test run
    main(max_pairs=20, splits=["train"])
