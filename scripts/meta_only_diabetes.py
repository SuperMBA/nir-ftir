import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score

def spec_from_cm(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/processed/diabetes_saliva.parquet")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out", default="reports/meta_only_diabetes_holdout.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)

    # meta-only: ONLY age + gender (без glucose/hemoglobin!)
    X = pd.DataFrame({
        "age": pd.to_numeric(df["age"], errors="coerce"),
        # приведи gender к 0/1 (подстрой под свои значения)
        "gender_male": df["gender"].astype(str).str.upper().str.strip().isin(["MALE","M"]).astype(int),
    })
    y = df["target"].astype(int).to_numpy()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows = []
    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
        tr, te = next(sss.split(X, y))

        clf = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(X.iloc[tr], y[tr])

        p = clf.predict_proba(X.iloc[te])[:, 1]
        auc = roc_auc_score(y[te], p)

        pred = (p >= 0.5).astype(int)
        f1 = f1_score(y[te], pred)
        rec = recall_score(y[te], pred)
        prec = precision_score(y[te], pred, zero_division=0)
        spec = spec_from_cm(y[te], pred)

        rows.append({"seed": seed, "auc": auc, "f1@0.5": f1, "recall@0.5": rec, "prec@0.5": prec, "spec@0.5": spec})

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print("Saved:", args.out)
    print(out.describe())

if __name__ == "__main__":
    main()
