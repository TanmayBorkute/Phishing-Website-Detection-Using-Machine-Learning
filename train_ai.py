"""Trains AI-mode models on the PhiUSIIL Phishing URL dataset (UCI #967).
URL features are recomputed from the raw URL with urlfeatures.py (identical to live scans). Train/test split is GROUPED BY
DOMAIN so the same site never appears on both sides. KNN and SVM use a stratified subsample (they do not scale to 190k rows)."""
import json, os, sys, warnings
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from urlfeatures import LEX, PAGE_BOOL, lexical, registered_domain, norm_url
from urllib.parse import urlparse

OUT, SEED, CSV = "models", 42, "data/PhiUSIIL_Phishing_URL_Dataset.csv"
os.makedirs(OUT, exist_ok=True)
if not os.path.exists(CSV):
    try:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=967); df = d.data.features.copy(); df["label"] = d.data.targets.iloc[:, 0].values; df.to_csv(CSV, index=False)
    except Exception as e:
        sys.exit(f"Download PhiUSIIL_Phishing_URL_Dataset.csv from https://archive.ics.uci.edu/dataset/967 into data/ ({e})")
df = pd.read_csv(CSV).drop_duplicates("URL").reset_index(drop=True)
y = (df["label"] == 0).astype(int)                     # dataset: 1 = legitimate -> we use 1 = phishing
print("rows:", len(df), "phishing:", int(y.sum()))
lex = pd.DataFrame([lexical(u) for u in df["URL"]])
groups = [registered_domain((urlparse(norm_url(u)).hostname or "").lower()) for u in df["URL"]]
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(lex, y, groups))
print("train", len(tr), "test", len(te), "| shared domains:", len(set(np.array(groups)[tr]) & set(np.array(groups)[te])))
# TLD legitimacy rate from the TRAIN fold only (smoothed), like the dataset's own TLDLegitimateProb
prior = float(1 - y.iloc[tr].mean()); g = pd.DataFrame({"tld": lex["tld"].iloc[tr].values, "legit": 1 - y.iloc[tr].values}).groupby("tld")["legit"].agg(["sum", "count"])
tld_rate = {k: float((r["sum"] + prior * 10) / (r["count"] + 10)) for k, r in g.iterrows()}
X = lex.drop(columns="tld"); X["tld_legit_rate"] = lex["tld"].map(tld_rate).fillna(prior)
X = X[LEX]
bools = df[[c for c in PAGE_BOOL if c != "HasIframe"]].copy(); bools["HasIframe"] = (df["NoOfiFrame"] > 0).astype(int)
Xp = pd.concat([X, bools[PAGE_BOOL]], axis=1)
sets = {"url": X, "page": Xp}
rng = np.random.RandomState(SEED)
def sub(idx, n): return rng.choice(idx, min(n, len(idx)), replace=False)
models = lambda: {"decision_tree": DecisionTreeClassifier(max_depth=16, random_state=SEED),
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "naive_bayes": GaussianNB(),
    "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, probability=True, random_state=SEED)),
    "boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1, random_state=SEED)}
SUB = {"knn": 30000, "svm": 8000}
ai_models, metrics, medians = {}, {}, {}
for s, D in sets.items():
    ai_models[s], metrics[s] = {}, {}
    medians[s] = {c: float(D.iloc[tr][c].median()) for c in D.columns}
    for k, m in models().items():
        idx = sub(tr, SUB[k]) if k in SUB else tr
        print(f"[{s}] training {k} on {len(idx)} rows", flush=True)
        m.fit(D.iloc[idx], y.iloc[idx])
        pr, pb = m.predict(D.iloc[te]), m.predict_proba(D.iloc[te])[:, 1]
        yt = y.iloc[te]
        metrics[s][k] = dict(accuracy=accuracy_score(yt, pr), precision=precision_score(yt, pr), recall=recall_score(yt, pr),
                             f1=f1_score(yt, pr), roc_auc=roc_auc_score(yt, pb), confusion=confusion_matrix(yt, pr).tolist(), train_rows=len(idx))
        ai_models[s][k] = m
        print(f"   acc {metrics[s][k]['accuracy']:.4f} f1 {metrics[s][k]['f1']:.4f} auc {metrics[s][k]['roc_auc']:.4f}", flush=True)
# 2-D map of the URL feature space (t-SNE on a stratified sample) + neighbour model to place new scans
sm = np.concatenate([sub(np.where(y == v)[0], 3000) for v in (0, 1)])
sc = StandardScaler().fit(X.iloc[sm]); Z = sc.transform(X.iloc[sm])
xy = TSNE(n_components=2, perplexity=40, random_state=SEED, init="pca").fit_transform(Z)
xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
joblib.dump(dict(scaler=sc, nn=NearestNeighbors(n_neighbors=8).fit(Z), xy=xy, cols=LEX), f"{OUT}/ai_map.joblib")
json.dump([[round(float(a), 4), round(float(b), 4), int(l)] for (a, b), l in zip(xy, y.iloc[sm])], open(f"{OUT}/ai_map.json", "w"))
joblib.dump(ai_models, f"{OUT}/ai_models.joblib")
json.dump(dict(sets={"url": LEX, "page": PAGE_BOOL and list(Xp.columns)}, models=metrics, medians=medians, tld_rate=tld_rate, prior=prior,
               rows=len(df), phishing=int(y.sum()), train=len(tr), test=len(te), source="PhiUSIIL (UCI 967)"),
          open(f"{OUT}/ai_meta.json", "w"), default=float)
print("AI models saved.")
