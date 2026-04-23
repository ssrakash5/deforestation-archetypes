from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import umap
import hdbscan
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from skimage.transform import resize

PROJECT_ROOT = Path("/content/drive/MyDrive/deforestation_archetypes")
FEATS = PROJECT_ROOT / "data" / "features" / "event_features.parquet"
EVENTS = PROJECT_ROOT / "data" / "events_pixel" / "events.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "archetypes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LABELS = OUT_DIR / "event_clusters_shapeonly.csv"
OUT_STATS = OUT_DIR / "cluster_stats.csv"
OUT_README = OUT_DIR / "README.txt"

# ---- clustering config (shape-only) ----
SHAPE_COLS = [
    "area_px","perimeter_px","compactness","eccentricity","solidity",
    "extent","convex_area","bbox_area","aspect_ratio","n_components",
    "edge_density","mask_frac","crop_h","crop_w"
]

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.10
UMAP_N_COMPONENTS = 5
HDB_MIN_CLUSTER_SIZE = 30
HDB_MIN_SAMPLES = 5

# ---- viz config ----
SAMPLES_PER_CLUSTER = 25        # montage examples per cluster
MONTAGE_COLS = 5                # 5x5 grid
MEAN_MASK_SIZE = 64             # prototype size
RANDOM_SEED = 42

def to_rgb(img_chw_or_hwc: np.ndarray) -> np.ndarray:
    img = img_chw_or_hwc
    if img.ndim == 3 and img.shape[0] in (3,4):
        img = img.transpose(1,2,0)
    rgb = img[..., :3].astype(np.float32)
    mx = float(rgb.max()) if rgb.size else 0.0
    if mx > 1.5:  # likely 0..255
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb

def overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    # red overlay
    out = rgb.copy()
    m = (mask > 0)
    if m.any():
        out[m, 0] = np.clip(out[m, 0] * (1-alpha) + 1.0 * alpha, 0, 1)
        out[m, 1] = out[m, 1] * (1-alpha)
        out[m, 2] = out[m, 2] * (1-alpha)
    return out

def run_shape_only_clustering(df_feats: pd.DataFrame) -> np.ndarray:
    X = df_feats[SHAPE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
    Xs = StandardScaler().fit_transform(X)

    Z = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        random_state=RANDOM_SEED,
    ).fit_transform(Xs)

    labels = hdbscan.HDBSCAN(
        min_cluster_size=HDB_MIN_CLUSTER_SIZE,
        min_samples=HDB_MIN_SAMPLES,
    ).fit_predict(Z)

    return labels

def make_cluster_montage(rows: pd.DataFrame, out_png: Path, title: str):
    n = len(rows)
    if n == 0:
        return

    cols = MONTAGE_COLS
    rows_n = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(cols * 3.0, rows_n * 3.0))
    fig.suptitle(title, fontsize=14)

    for i, (_, r) in enumerate(rows.iterrows()):
        img = np.load(r["event_img"])
        msk = np.load(r["event_mask"])
        rgb = to_rgb(img)
        vis = overlay(rgb, msk)

        ax = plt.subplot(rows_n, cols, i+1)
        ax.imshow(vis)
        ax.axis("off")
        ax.set_title(f"{r['event_id']}\nA={int(r['area_px'])}", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def make_mean_mask(rows: pd.DataFrame, out_png: Path, title: str):
    if len(rows) == 0:
        return

    acc = np.zeros((MEAN_MASK_SIZE, MEAN_MASK_SIZE), dtype=np.float32)
    for _, r in rows.iterrows():
        m = np.load(r["event_mask"]).astype(np.float32)
        if m.max() <= 0:
            continue
        m = (m > 0).astype(np.float32)
        m_rs = resize(m, (MEAN_MASK_SIZE, MEAN_MASK_SIZE), order=0, preserve_range=True, anti_aliasing=False)
        acc += m_rs

    mean = acc / max(len(rows), 1)
    fig = plt.figure(figsize=(3.5, 3.5))
    plt.imshow(mean, cmap="viridis")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title, fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

def main():
    feats = pd.read_parquet(FEATS)
    events = pd.read_csv(EVENTS)

    # join to ensure we have event_img/event_mask paths + metadata
    df = feats.merge(events[["event_id","event_img","event_mask"]], on="event_id", how="left")

    # compute clustering labels (shape-only) to match your 14 cluster outcome
    labels = run_shape_only_clustering(df)
    df["cluster"] = labels

    # save labels
    df[["event_id","cluster","area_px","compactness","edge_density","mask_frac","scene_id","t0","t1"]].to_csv(OUT_LABELS, index=False)

    # stats table
    stats = (
        df.groupby("cluster")
          .agg(
              n=("event_id","count"),
              area_mean=("area_px","mean"),
              area_median=("area_px","median"),
              compact_mean=("compactness","mean"),
              ecc_mean=("eccentricity","mean"),
              aspect_mean=("aspect_ratio","mean"),
              edge_mean=("edge_density","mean"),
              maskfrac_mean=("mask_frac","mean"),
              crop_h_mean=("crop_h","mean"),
              crop_w_mean=("crop_w","mean"),
          )
          .reset_index()
          .sort_values(["cluster"])
    )
    stats.to_csv(OUT_STATS, index=False)

    # summary prints
    n_noise = int((df["cluster"] == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print("✅ wrote:", OUT_LABELS)
    print("✅ wrote:", OUT_STATS)
    print(f"events={len(df)} clusters={n_clusters} noise={n_noise} ({n_noise/len(df):.1%})")
    print("\nTop cluster sizes:")
    print(df["cluster"].value_counts().head(15))

    # generate per-cluster galleries
    rng = np.random.default_rng(RANDOM_SEED)
    clusters = sorted([c for c in df["cluster"].unique() if c != -1])

    for c in clusters + ([-1] if -1 in df["cluster"].unique() else []):
        sub = df[df["cluster"] == c].copy()
        if len(sub) == 0:
            continue

        # pick sample events (stable random)
        k = min(SAMPLES_PER_CLUSTER, len(sub))
        take_idx = rng.choice(len(sub), size=k, replace=False)
        sample = sub.iloc[take_idx].reset_index(drop=True)

        # filenames
        ctag = "noise" if c == -1 else f"{c:02d}"
        g_png = OUT_DIR / f"cluster_{ctag}_gallery.png"
        m_png = OUT_DIR / f"cluster_{ctag}_meanmask.png"

        title = f"Cluster {c} (n={len(sub)})" if c != -1 else f"Noise (n={len(sub)})"
        make_cluster_montage(sample, g_png, title=title)
        make_mean_mask(sub.head(300), m_png, title=f"{title} mean mask (first 300)")

    OUT_README.write_text(
        "Step 7 outputs:\n"
        f"- Labels: {OUT_LABELS}\n"
        f"- Stats:  {OUT_STATS}\n"
        "- cluster_XX_gallery.png: 25 sample events with mask overlays\n"
        "- cluster_XX_meanmask.png: mean mask prototype\n"
    )
    print("✅ galleries written to:", OUT_DIR)

if __name__ == "__main__":
    main()
