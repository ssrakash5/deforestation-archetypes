from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.measure import label, regionprops, perimeter as sk_perimeter
from skimage.feature import canny

import torch
import torchvision
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path("/content/drive/MyDrive/deforestation_archetypes")

EVENTS_CSV = PROJECT_ROOT / "data" / "events_pixel" / "events.csv"
OUT_DIR = PROJECT_ROOT / "data" / "features"
OUT_PARQUET = OUT_DIR / "event_features.parquet"
OUT_CSV = OUT_DIR / "event_features.csv"
OUT_EMB = OUT_DIR / "event_embeddings.npy"

# ---------- helpers ----------
def to_rgb_hwc(img: np.ndarray) -> np.ndarray:
    # input is (C,H,W) float32 typically
    if img.ndim == 3 and img.shape[0] in (3, 4):
        img = img.transpose(1, 2, 0)
    return img

def normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mx = float(x.max()) if x.size else 0.0
    if mx > 1.5:  # likely 0..255
        x = x / 255.0
    x = np.clip(x, 0.0, 1.0)
    return x

def largest_component_props(mask: np.ndarray):
    m = (mask > 0).astype(np.uint8)
    if m.max() == 0:
        return None, None
    lab = label(m, connectivity=2)
    props = regionprops(lab)
    if not props:
        return None, None
    p = max(props, key=lambda r: r.area)
    return p, lab

def compute_shape_features(mask: np.ndarray) -> dict:
    p, lab = largest_component_props(mask)
    if p is None:
        return {
            "area_px": 0,
            "perimeter_px": 0.0,
            "compactness": 0.0,
            "eccentricity": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "convex_area": 0,
            "bbox_area": 0,
            "aspect_ratio": 0.0,
            "n_components": 0,
        }

    m = (mask > 0).astype(np.uint8)
    per = float(sk_perimeter(m, neighborhood=8))
    area = int(p.area)
    comp = float((4.0 * math.pi * area) / (per * per + 1e-6)) if area > 0 else 0.0

    minr, minc, maxr, maxc = p.bbox
    bbox_h = maxr - minr
    bbox_w = maxc - minc
    bbox_area = int(bbox_h * bbox_w)
    ar = float(bbox_w / (bbox_h + 1e-6))

    n_comp = int(lab.max())

    return {
        "area_px": area,
        "perimeter_px": per,
        "compactness": comp,
        "eccentricity": float(getattr(p, "eccentricity", 0.0)),
        "solidity": float(getattr(p, "solidity", 0.0)),
        "extent": float(getattr(p, "extent", 0.0)),
        "convex_area": int(getattr(p, "convex_area", area)),
        "bbox_area": bbox_area,
        "aspect_ratio": ar,
        "n_components": n_comp,
    }

def compute_context_features(img_hwc: np.ndarray, mask: np.ndarray) -> dict:
    rgb = normalize_01(img_hwc[..., :3])
    gray = (0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]).astype(np.float32)

    # edge density (simple proxy for "roads/structures" context)
    edges = canny(gray, sigma=1.0)
    edge_density = float(edges.mean())

    # global intensity stats
    mean_r, mean_g, mean_b = [float(rgb[..., i].mean()) for i in range(3)]
    std_r, std_g, std_b = [float(rgb[..., i].std()) for i in range(3)]

    # masked intensity stats (inside event)
    m = (mask > 0)
    if m.any():
        in_mean = float(gray[m].mean())
        in_std = float(gray[m].std())
        frac = float(m.mean())
    else:
        in_mean, in_std, frac = 0.0, 0.0, 0.0

    return {
        "edge_density": edge_density,
        "mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b,
        "std_r": std_r, "std_g": std_g, "std_b": std_b,
        "masked_gray_mean": in_mean,
        "masked_gray_std": in_std,
        "mask_frac": frac,
        "crop_h": int(rgb.shape[0]),
        "crop_w": int(rgb.shape[1]),
    }

class ResNetEmbedder:
    def __init__(self, device: str):
        self.device = device
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        self.preprocess = weights.transforms()  # includes resize/crop/normalize to ImageNet
        model = torchvision.models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(device)

    @torch.no_grad()
    def embed_batch(self, imgs_hwc: list[np.ndarray]) -> np.ndarray:
        # imgs_hwc: list of HWC float arrays in [0,1] (we'll normalize/convert)
        tensors = []
        for img in imgs_hwc:
            rgb = normalize_01(img[..., :3])
            t = torch.from_numpy(rgb).permute(2, 0, 1)  # CHW
            t = self.preprocess(t)  # (3,224,224) normalized
            tensors.append(t)
        batch = torch.stack(tensors, dim=0).to(self.device)
        emb = self.model(batch)  # (B,512)
        return emb.detach().cpu().numpy().astype(np.float32)

def main(batch_size: int = 64, max_events: int | None = None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(EVENTS_CSV)
    if max_events is not None:
        df = df.head(max_events).reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = ResNetEmbedder(device=device)

    feature_rows = []
    embeddings = []

    batch_imgs = []
    batch_meta = []

    def flush_batch():
        nonlocal batch_imgs, batch_meta, embeddings, feature_rows
        if not batch_imgs:
            return
        emb = embedder.embed_batch(batch_imgs)
        embeddings.append(emb)
        feature_rows.extend(batch_meta)
        batch_imgs = []
        batch_meta = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Step5: features"):
        img = np.load(r["event_img"])
        mask = np.load(r["event_mask"])

        img_hwc = to_rgb_hwc(img)

        shape = compute_shape_features(mask)
        ctx = compute_context_features(img_hwc, mask)

        meta = {
            "event_id": r["event_id"],
            "patch_id": r["patch_id"],
            "split": r["split"],
            "scene_id": r["scene_id"],
            "t0": r["t0"],
            "t1": r["t1"],
            "row": int(r["row"]),
            "col": int(r["col"]),
        }
        meta.update(shape)
        meta.update(ctx)

        batch_imgs.append(img_hwc)
        batch_meta.append(meta)

        if len(batch_imgs) >= batch_size:
            flush_batch()

    flush_batch()

    feats = pd.DataFrame(feature_rows)
    emb_all = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 512), dtype=np.float32)

    # attach embedding columns
    for i in range(emb_all.shape[1]):
        feats[f"emb_{i:03d}"] = emb_all[:, i]

    feats.to_parquet(OUT_PARQUET, index=False)
    feats.to_csv(OUT_CSV, index=False)
    np.save(OUT_EMB, emb_all)

    print(f"✅ wrote: {OUT_PARQUET}")
    print(f"✅ wrote: {OUT_CSV}")
    print(f"✅ wrote: {OUT_EMB}  shape={emb_all.shape}")
    print("\nPreview:")
    print(feats.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
