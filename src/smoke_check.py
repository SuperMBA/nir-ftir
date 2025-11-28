from __future__ import annotations

import numpy as np
from umap import UMAP


def main() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 50))
    emb = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)
    print("UMAP ok, shape:", emb.shape)


if __name__ == "__main__":
    main()
