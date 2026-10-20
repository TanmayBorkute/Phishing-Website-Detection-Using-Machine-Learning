"""Training pipeline: Training_Dataset.arff (UCI #327) -> clean -> split -> scale -> 5 models -> evaluate
-> external validation on the legacy file -> best model -> save.  Also trains 'AI-mode' models on the feature subsets
that features.py can compute from a URL (URL-only set, and URL+page set)."""
import json, os, sys
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))  # silences joblib core-count warning on Windows
import warnings; warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from features import INFO, shift1
import difflib, re

OUT, SEED = "models", 42
MAIN = "data/Training_Dataset.arff"
LEGACY = next((p for p in ["data/old_dataset.arff", "data/_old.arff", "data/old.arff"] if os.path.exists(p)), "data/old_dataset.arff")
os.makedirs(OUT, exist_ok=True)

def canon_cols(df):
    """Rename dataset columns to the canonical names used by features.py (tolerates case, quotes, underscores and the
    dataset's own typos such as popUpWidnow/popUpWindow, Shortining/Shortening, registeration/registration)."""
    norm = lambda c: re.sub(r"[^a-z0-9]", "", str(c).lower())
    canon = {norm(k): k for k in INFO}
    ren = {}
    for c in df.columns:
        if c == "Result": continue
        n = norm(c)
        m = n if n in canon else (difflib.get_close_matches(n, list(canon), n=1, cutoff=0.85) or [None])[0]
        if m is None: sys.exit(f"Unknown column '{c}' in dataset")
        ren[c] = canon[m]
    return df.rename(columns=ren)

def read_arff(path):
    names, rows, data = [], [], False
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("%"): continue
        low = line.lower()
        if low.startswith("@attribute"): names.append(line.split()[1])
        elif low.startswith("@data"): data = True
        elif data: rows.append([int(v) for v in line.split(",")])
    return canon_cols(pd.DataFrame(rows, columns=[n.strip("'\"") for n in names]))

def load():
    if os.path.exists(MAIN): return read_arff(MAIN)
    try:  # fallback: download from UCI
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=327); df = d.data.features.copy(); df["Result"] = d.data.targets.iloc[:, 0].values; return canon_cols(df)
    except Exception as e:
        sys.exit(f"Put Training_Dataset.arff in data/ ({e})")

def legacy_to_uci(old, ref):
    """_old.arff: same features, 0/1 encoding where the UCI file has -1/1, and Result sign inverted (1 = phishing)."""
    X = old.drop(columns="Result").copy()
    for c in X:
        if set(old[c].unique()) <= {0, 1} and set(ref[c].unique()) != {0, 1}: X[c] = X[c].replace(0, -1)
    return X, (old["Result"] == 1).astype(int)

df = load(); raw_rows = len(df)
df = df.dropna().drop_duplicates().reset_index(drop=True)   # duplicates would leak between train and test
X, y = df.drop(columns="Result"), (df["Result"] == -1).astype(int)   # 1 = phishing
feats = list(X.columns)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
Xo, yo = (legacy_to_uci(read_arff(LEGACY), df) if os.path.exists(LEGACY) else (None, None))

models = {
    "decision_tree": DecisionTreeClassifier(max_depth=12, random_state=SEED),
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "naive_bayes": make_pipeline(FunctionTransformer(shift1), CategoricalNB(min_categories=3)),  # features are categorical (-1/0/1)
    "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, probability=True, random_state=SEED)),
    "boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=SEED),
}
def ext(m, cols):  # external validation on the legacy dataset
    if Xo is None: return None
    p = m.predict(Xo[cols]); return dict(accuracy=accuracy_score(yo, p), f1=f1_score(yo, p))

metrics, external, grid = {}, {}, np.linspace(0, 1, 60)
for k, m in models.items():
    print("Training", k)
    m.fit(Xtr, ytr)
    pred, prob = m.predict(Xte), m.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, prob)
    metrics[k] = dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred), recall=recall_score(yte, pred),
                      f1=f1_score(yte, pred), roc_auc=roc_auc_score(yte, prob), confusion=confusion_matrix(yte, pred).tolist(),
                      roc=dict(fpr=grid.tolist(), tpr=np.interp(grid, fpr, tpr).tolist()))
    external[k] = ext(m, feats)
    joblib.dump(m, f"{OUT}/{k}.joblib")

best = max(metrics, key=lambda k: (metrics[k]["f1"], metrics[k]["roc_auc"]))
imp = permutation_importance(models[best], Xte, yte, n_repeats=5, random_state=SEED, scoring="f1")
fi = sorted(zip(feats, imp.importances_mean), key=lambda t: -t[1])
allowed = {f: sorted(int(v) for v in X[f].unique()) for f in feats}
json.dump(dict(models=metrics, best=best, features=feats, allowed=allowed, external=external if Xo is not None else None,
               importance=[dict(feature=f, value=float(v)) for f, v in fi]), open(f"{OUT}/metrics.json", "w"), default=float)

ds = dict(raw_rows=raw_rows, rows=len(df), features=len(feats), phishing=int(y.sum()), legitimate=int((1 - y).sum()),
          train=len(Xtr), test=len(Xte), legacy_rows=0 if Xo is None else len(Xo),
          value_counts={f: {str(v): int(c) for v, c in X[f].value_counts().items()} for f in feats},
          stats=X.describe().T.round(3).reset_index().rename(columns={"index": "feature"}).to_dict("records"),
          preview=df.head(10).to_dict("records"))
json.dump(ds, open(f"{OUT}/dataset.json", "w"), default=int)
joblib.dump(Xte.assign(Result=yte).reset_index(drop=True), f"{OUT}/test_samples.joblib")

# ---- feature-space map (t-SNE of all unique records) + neighbour model to place new manual scans on it
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
xy = TSNE(n_components=2, perplexity=35, random_state=SEED, init="pca").fit_transform(X.values.astype(float))
xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
joblib.dump(dict(nn=NearestNeighbors(n_neighbors=10).fit(X.values), xy=xy, cols=feats), f"{OUT}/uci_map.joblib")
json.dump([[round(float(a), 4), round(float(b), 4), int(l)] for (a, b), l in zip(xy, y)], open(f"{OUT}/uci_map.json", "w"))

# ---- precision/recall/accuracy at every decision threshold (powers the threshold slider)
def curve(prob, yt):
    out = []
    for t in np.round(np.arange(0, 1.0001, 0.02), 2):
        p = prob >= t; tp = int((p & (yt == 1)).sum()); fp = int((p & (yt == 0)).sum()); fn = int((~p & (yt == 1)).sum()); tn = int((~p & (yt == 0)).sum())
        pr = tp / (tp + fp) if tp + fp else 1.0; rc = tp / (tp + fn) if tp + fn else 0.0
        out.append(dict(t=float(t), tp=tp, fp=fp, fn=fn, tn=tn, precision=pr, recall=rc, f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0, accuracy=(tp + tn) / len(yt)))
    return out
mj = json.load(open(f"{OUT}/metrics.json"))
mj["threshold"] = {k: curve(m.predict_proba(Xte)[:, 1], yte.values) for k, m in models.items()}
json.dump(mj, open(f"{OUT}/metrics.json", "w"), default=float)

print(f"Rows: raw {raw_rows}, unique {len(df)} | legacy validation rows: {0 if Xo is None else len(Xo)}")
for k in metrics: print(f"{k:14s} test acc {metrics[k]['accuracy']:.4f} f1 {metrics[k]['f1']:.4f} auc {metrics[k]['roc_auc']:.4f} | legacy acc {external[k]['accuracy'] if Xo is not None else '-'}")
print("Best:", best)
