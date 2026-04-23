from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label, regionprops

PROJECT_ROOT = Path("/content/drive/MyDrive/deforestation_archetypes")
INDEX_WITH_MASKS = PROJECT_ROOT / "data" / "processed" / "patch_index_with_masks_pixel.csv"
EVENT_DIR = PROJECT_ROOT / "data" / "events_pixel"
EVENTS_CSV = EVENT_DIR / "events.csv"

MIN_AREA_PX = 30
CONNECTIVITY = 2
PADDING = 16

def crop_pad(arr, r0,c0,r1,c1,pad):
    H = arr.shape[-2] if arr.ndim==3 else arr.shape[0]
    W = arr.shape[-1] if arr.ndim==3 else arr.shape[1]
    rr0=max(r0-pad,0); cc0=max(c0-pad,0)
    rr1=min(r1+pad,H); cc1=min(c1+pad,W)
    if arr.ndim==3:
        return arr[:, rr0:rr1, cc0:cc1], rr0,cc0,rr1,cc1
    return arr[rr0:rr1, cc0:cc1], rr0,cc0,rr1,cc1

def main():
    df = pd.read_csv(INDEX_WITH_MASKS)
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    rows=[]
    eidn=0

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Extracting events (pixel masks)"):
        img = np.load(r["t1_patch"])
        chg = np.load(r["change_mask"])
        if chg.max()==0:
            continue
        lab = label(chg.astype(bool), connectivity=CONNECTIVITY)
        for p in regionprops(lab):
            if p.area < MIN_AREA_PX:
                continue
            minr,minc,maxr,maxc = p.bbox
            msk_crop, cr0,cc0,cr1,cc1 = crop_pad(chg, minr,minc,maxr,maxc, PADDING)
            img_crop, _,_,_,_ = crop_pad(img, minr,minc,maxr,maxc, PADDING)

            eid=f"EV_{eidn:07d}"
            out_img = EVENT_DIR / f"{eid}_img.npy"
            out_msk = EVENT_DIR / f"{eid}_mask.npy"
            np.save(out_img, img_crop.astype(np.float32))
            np.save(out_msk, msk_crop.astype(np.uint8))

            rows.append({
                "event_id": eid,
                "patch_id": r["patch_id"],
                "split": r["split"],
                "scene_id": r["scene_id"],
                "t0": r["t0"],
                "t1": r["t1"],
                "row": int(r["row"]),
                "col": int(r["col"]),
                "area_px": int(p.area),
                "event_img": str(out_img),
                "event_mask": str(out_msk),
            })
            eidn += 1

    out = pd.DataFrame(rows)
    out.to_csv(EVENTS_CSV, index=False)
    print(f"✅ wrote {len(out)} events -> {EVENTS_CSV}")

if __name__ == "__main__":
    main()

