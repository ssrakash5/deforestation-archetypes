from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import umap
import hdbscan
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("/content/drive/MyDrive/deforestation_archetypes")
FEATS = PROJECT_ROOT / "data" / "features" / "event_features.parquet"
OUT = PROJECT_ROOT / "data" / "features" / "event_clusters.csv"

def main():
    df = pd.read_parquet(FEATS)

    # Choose feature columns: shape+context + embeddings
    shape_ctx = [
        "area_px","perimeter_px","compactness","eccentricity","solidity","extent",
        "convex_area","bbox_area","aspect_ratio","n_components",
        "edge_density","mean_r","mean_g","mean_b","std_r","std_g","std_b",
        "masked_gray_mean","masked_gray_std","mask_frac","crop_h","crop_w",
    ]
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    use_cols = shape_ctx + emb_cols

    X = df[use_cols].fillna(0.0).to_numpy(dtype=np.float32)

    # Scale (important for mixing handcrafted + deep features)
    Xs = StandardScaler().fit_transform(X)

    # UMAP to 20D for clustering stability
    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.05,
        n_components=20,
        metric="euclidean",
        random_state=42,
    )
    Z = reducer.fit_transform(Xs)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=25,
        min_samples=10,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(Z)
    probs = clusterer.probabilities_

    out = df[["event_id","scene_id","t0","t1","row","col","area_px","compactness","edge_density","mask_frac"]].copy()
    out["cluster"] = labels
    out["cluster_prob"] = probs

    out.to_csv(OUT, index=False)

    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"✅ wrote {OUT}")
    print("events:", len(out))
    print("clusters:", n_clusters)
    print("noise:", n_noise, f"({n_noise/len(out):.1%})")
    print("\nTop cluster sizes:")
    print(out[out.cluster!=-1].cluster.value_counts().head(15))

if __name__ == "__main__":
    main()
