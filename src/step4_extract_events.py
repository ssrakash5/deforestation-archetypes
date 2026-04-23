from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label, regionprops

PROJECT_ROOT = Path("/content/drive/MyDrive/deforestation_archetypes")

INDEX_WITH_MASKS = PROJECT_ROOT / "data" / "processed" / "patch_index_with_masks.csv"
EVENT_DIR = PROJECT_ROOT / "data" / "events"
EVENTS_CSV = EVENT_DIR / "events.csv"

# event filtering + crop
MIN_AREA_PX = 30
CONNECTIVITY = 2        # 2 = 8-connectivity for 2D
PADDING = 16            # pixels around bbox

def crop_with_padding(arr: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int):
    H, W = arr.shape[-2], arr.shape[-1] if arr.ndim == 3 else arr.shape
    rr0 = max(r0 - pad, 0)
    cc0 = max(c0 - pad, 0)
    rr1 = min(r1 + pad, H)
    cc1 = min(c1 + pad, W)
    if arr.ndim == 3:
        return arr[:, rr0:rr1, cc0:cc1], rr0, cc0, rr1, cc1
    else:
        return arr[rr0:rr1, cc0:cc1], rr0, cc0, rr1, cc1

def main(max_patches: int | None = None):
    df = pd.read_csv(INDEX_WITH_MASKS)
    if max_patches is not None:
        df = df.head(max_patches).reset_index(drop=True)

    EVENT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    event_id = 0

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Extracting events"):
        patch_id = r["patch_id"]
        split = r["split"]
        scene = r["scene_id"]
        t0 = r["t0"]
        t1 = r["t1"]
        rr = int(r["row"])
        cc = int(r["col"])

        t1_patch_path = Path(r["t1_patch"])
        change_mask_path = Path(r["change_mask"])

        if not t1_patch_path.exists() or not change_mask_path.exists():
            continue

        img_t1 = np.load(t1_patch_path)         # (C,H,W)
        chg = np.load(change_mask_path)         # (H,W) uint8 {0,1}

        if chg.max() == 0:
            continue

        lab = label(chg.astype(bool), connectivity=CONNECTIVITY)
        props = regionprops(lab)

        for p in props:
            if p.area < MIN_AREA_PX:
                continue

            minr, minc, maxr, maxc = p.bbox  # [minr, minc, maxr, maxc)

            mask_crop, cr0, cc0, cr1, cc1 = crop_with_padding(chg, minr, minc, maxr, maxc, PADDING)
            img_crop, _, _, _, _ = crop_with_padding(img_t1, minr, minc, maxr, maxc, PADDING)

            # save event artifacts
            eid = f"EV_{event_id:07d}"
            out_img = EVENT_DIR / f"{eid}_img.npy"
            out_msk = EVENT_DIR / f"{eid}_mask.npy"
            np.save(out_img, img_crop.astype(np.float32))
            np.save(out_msk, mask_crop.astype(np.uint8))

            rows.append({
                "event_id": eid,
                "patch_id": patch_id,
                "split": split,
                "scene_id": scene,
                "t0": t0,
                "t1": t1,
                "row": rr,
                "col": cc,
                "area_px": int(p.area),
                "bbox_r0": int(cr0),
                "bbox_c0": int(cc0),
                "bbox_r1": int(cr1),
                "bbox_c1": int(cc1),
                "event_img": str(out_img),
                "event_mask": str(out_msk),
            })
            event_id += 1

    out = pd.DataFrame(rows)
    out.to_csv(EVENTS_CSV, index=False)
    print(f"✅ wrote {len(out)} events -> {EVENTS_CSV}")
    if len(out):
        print(out.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
