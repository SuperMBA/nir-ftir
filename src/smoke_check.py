import sys
mods = ["numpy","pandas","sklearn","umap","matplotlib","xgboost","shap","statsmodels"]
bad=[]
for m in mods:
    try:
        __import__(m if m!="sklearn" else "sklearn")
    except Exception as e:
        bad.append((m,str(e)))
if bad:
    print("НЕ ХВАТАЕТ:", bad); sys.exit(1)

# мини-проверка UMAP на синтетике
import numpy as np, umap
X = np.random.randn(200, 50)
emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42).fit_transform(X)
print("UMAP ok, shape:", emb.shape)
