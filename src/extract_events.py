
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label, regionprops

MIN_AREA_PX = 30
CONNECTIVITY = 2
PADDING = 16

def crop_with_padding(arr, r0, c0, r1, c1, pad):
    # arr: (C,H,W) or (H,W)
    if arr.ndim == 3:
        H, W = arr.shape[1], arr.shape[2]
    else:
        H, W = arr.shape[0], arr.shape[1]

    rr0 = max(r0 - pad, 0)
    cc0 = max(c0 - pad, 0)
    rr1 = min(r1 + pad, H)
    cc1 = min(c1 + pad, W)

    if arr.ndim == 3:
        return arr[:, rr0:rr1, cc0:cc1], rr0, cc0, rr1, cc1
    else:
        return arr[rr0:rr1, cc0:cc1], rr0, cc0, rr1, cc1

def extract_events(index_csv, event_dir, min_area_px=MIN_AREA_PX):
    df = pd.read_csv(index_csv)
    event_dir = Path(event_dir)
    event_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    event_id = 0

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Extracting events"):
        t1_patch = Path(r["t1_patch"])
        change_mask = Path(r["change_mask"])

        if not t1_patch.exists() or not change_mask.exists():
            continue

        img_t1 = np.load(t1_patch)
        chg = np.load(change_mask)

        if chg.max() == 0:
            continue

        lab = label(chg.astype(bool), connectivity=CONNECTIVITY)
        for p in regionprops(lab):
            if p.area < min_area_px:
                continue

            minr, minc, maxr, maxc = p.bbox
            mask_crop, r0, c0, r1, c1 = crop_with_padding(
                chg, minr, minc, maxr, maxc, PADDING
            )
            img_crop, _, _, _, _ = crop_with_padding(
                img_t1, minr, minc, maxr, maxc, PADDING
            )

            eid = f"EV_{event_id:07d}"
            img_path = event_dir / f"{eid}_img.npy"
            msk_path = event_dir / f"{eid}_mask.npy"

            np.save(img_path, img_crop.astype(np.float32))
            np.save(msk_path, mask_crop.astype(np.uint8))

            rows.append({
                "event_id": eid,
                "scene_id": r["scene_id"],
                "area_px": int(p.area),
                "event_img": str(img_path),
                "event_mask": str(msk_path),
            })

            event_id += 1

    return pd.DataFrame(rows)
