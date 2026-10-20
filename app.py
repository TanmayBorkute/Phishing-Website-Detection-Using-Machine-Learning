"""Flask backend.  Manual mode = UCI Phishing Websites (30 features, 5 models).
AI mode = PhiUSIIL URL models (domain-name features + simple page signals), features extracted live from a URL."""
import json, os, io, csv, random, subprocess, sys, math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import urlparse
import numpy as np, pandas as pd, joblib
from flask import Flask, jsonify, request, render_template, Response
from features import INFO
import urlfeatures as U

app = Flask(__name__)
M = "models"
if not os.path.exists(f"{M}/metrics.json"):
    subprocess.run([sys.executable, "train.py"], check=True)
if not os.path.exists(f"{M}/ai_meta.json") and os.path.exists("data/PhiUSIIL_Phishing_URL_Dataset.csv"):
    subprocess.run([sys.executable, "train_ai.py"], check=True)

METRICS = json.load(open(f"{M}/metrics.json")); DATASET = json.load(open(f"{M}/dataset.json"))
FEATS, ALLOWED = METRICS["features"], METRICS["allowed"]
MODELS = {k: joblib.load(f"{M}/{k}.joblib") for k in METRICS["models"]}
SAMPLES = joblib.load(f"{M}/test_samples.joblib")
_ai = os.path.exists(f"{M}/ai_meta.json")
AI_META = json.load(open(f"{M}/ai_meta.json")) if _ai else {}
AI_MODELS = joblib.load(f"{M}/ai_models.joblib") if _ai else {}
AI_SETS = AI_META.get("sets", {})
AI_BEST = max(AI_META["models"]["page"], key=lambda k: AI_META["models"]["page"][k]["f1"]) if _ai else None
MAPS = {n: dict(pts=json.load(open(f"{M}/{n}_map.json")), obj=joblib.load(f"{M}/{n}_map.joblib")) for n in ("uci", "ai") if os.path.exists(f"{M}/{n}_map.joblib")}
HISTORY = []
CAVEAT = ("AI mode is trained on the PhiUSIIL dataset, where legitimate sites are popular-site homepages and phishing pages are mostly "
          "bare or dead pages. Treat results as indicative; modern phishing kits that copy real pages can evade page checks.")

def place(name, vec):
    """Position of a scan on the feature-space map = mean position of its nearest dataset neighbours."""
    m = MAPS.get(name)
    if not m: return None
    o = m["obj"]; v = np.array([vec], dtype=float)
    if "scaler" in o: v = o["scaler"].transform(pd.DataFrame(v, columns=o["cols"]))
    idx = o["nn"].kneighbors(v, return_distance=False)[0]
    x, y = o["xy"][idx].mean(0)
    return dict(map=name, x=float(x), y=float(y))

def counterfactual(model, cols, row, p, baseline):
    """Impact of each feature on P(phishing): probability with the feature set to its baseline vs the actual value."""
    base = pd.DataFrame([row] * len(cols), columns=cols)
    for i, c in enumerate(cols): base.iloc[i, i] = baseline(c, row[i])
    return p - model.predict_proba(base)[:, 1]

def summarize(label, items):
    risk = [i for i in items if i["impact"] > 0.02][:3]; safe = [i for i in items if i["impact"] < -0.02][:2]
    say = lambda i: f'{i["title"].lower()} ({i["shown"]})'
    parts = (["risk signals: " + ", ".join(map(say, risk))] if risk else []) + (["reassuring signals: " + ", ".join(map(say, safe))] if safe else [])
    return f"{label.capitalize()} - " + ("; ".join(parts) if parts else "no single feature dominated the decision") + "."

def anatomy(url):
    u = U.norm_url(url); p = urlparse(u); host = (p.hostname or "").lower(); lab = host.split(".")
    reg = U.registered_domain(host); sub = host[: -len(reg) - 1] if host.endswith(reg) and host != reg else ""
    return dict(scheme=p.scheme + "://", sub=sub, domain=reg, port=(":" + str(p.port)) if p.port else "", path=(p.path or "") + (("?" + p.query) if p.query else ""),
                flags=[f for f, c in [("IP address as host", U.lexical(u)["is_ip"]), ("'@' in URL", "@" in u), ("hyphens in domain", "-" in reg),
                                      ("many sub-domains", len([x for x in sub.split(".") if x and x != "www"]) >= 2), ("non-standard port", bool(p.port) and p.port not in (80, 443)),
                                      ("URL shortener", U.lexical(u)["is_shortener"]), ("'//' in path", "//" in p.path), ("keyword in domain", U.lexical(u)["n_keywords_host"] > 0)] if c])

def adjudicate(label, conf, mode, f):
    """Safety layer for AI mode: hard red flags and low confidence downgrade a LEGITIMATE verdict; PHISHING is never softened."""
    if label != "LEGITIMATE": return label, None
    fl = []
    if f.get("is_ip") == 1: fl.append("an IP address is used instead of a domain name")
    if f.get("n_at", 0) > 0: fl.append("the URL contains an '@' symbol")
    if fl: return "SUSPICIOUS", "Red flags override the model: " + " and ".join(fl) + "."
    if conf < 0.8: return "INCONCLUSIVE", "The model is not confident enough to clear this site."
    if mode == "url": return "INCONCLUSIVE", "The page could not be fetched. The domain name looks ordinary, but page content was not checked, so the site cannot be cleared."
    return label, None

def ai_run(url, name, fetch=True):
    f, fetched, log = U.extract(url, AI_META["tld_rate"], AI_META["prior"], fetch)
    mode = "page" if fetched else "url"; cols = AI_SETS[mode]
    row = [float(f[c]) for c in cols]; model = AI_MODELS[mode][name]
    p = float(model.predict_proba(pd.DataFrame([row], columns=cols))[0, 1])
    model_label = "PHISHING" if p >= 0.5 else "LEGITIMATE"
    label, note = adjudicate(model_label, max(p, 1 - p), mode, f)
    med = AI_META["medians"][mode]
    imp = counterfactual(model, cols, row, p, lambda c, v: med[c])
    items = [dict(feature=c, title=U.TITLES[c], value=v, shown=(f"{v:g}"), impact=float(d)) for c, v, d in zip(cols, row, imp)]
    items.sort(key=lambda i: -abs(i["impact"]))
    return dict(label=label, model_label=model_label, note=note, probability_phishing=p, confidence=max(p, 1 - p), model=name, mode=mode, page_fetched=fetched,
                explanation=items[:8], impacts={i["feature"]: i["impact"] for i in items}, titles={c: U.TITLES[c] for c in cols}, values=dict(zip(cols, row)),
                summary=("" if not note else note + " Model view: ") + summarize(model_label, items), log=log, anatomy=anatomy(url), caveat=CAVEAT,
                place=place("ai", [float(f[c]) for c in MAPS["ai"]["obj"]["cols"]]) if "ai" in MAPS else None,
                time=datetime.now().strftime("%H:%M:%S"))

@app.route("/")
def index(): return render_template("index.html")

@app.get("/api/meta")
def meta(): return jsonify(features=FEATS, models=list(MODELS), best=METRICS["best"], allowed=ALLOWED, info=INFO, ai_available=bool(AI_MODELS), ai_best=AI_BEST)
@app.get("/api/metrics")
def metrics(): return jsonify(METRICS)
@app.get("/api/dataset")
def dataset(): return jsonify(DATASET)
@app.get("/api/ai_metrics")
def ai_metrics(): return jsonify({k: AI_META.get(k) for k in ("models", "rows", "phishing", "train", "test", "source", "sets")} | {"caveat": CAVEAT})
@app.get("/api/map/<name>")
def get_map(name): return jsonify(MAPS[name]["pts"]) if name in MAPS else (jsonify(error="map not built"), 404)

@app.get("/api/sample")
def sample():
    r = SAMPLES.iloc[random.randrange(len(SAMPLES))]
    return jsonify(features={f: int(r[f]) for f in FEATS}, actual="PHISHING" if r["Result"] == 1 else "LEGITIMATE")

def manual_validate(feats):
    missing = [f for f in FEATS if f not in feats]
    if missing: return None, f"Missing features: {', '.join(missing[:5])}..."
    try: row = [int(feats[f]) for f in FEATS]
    except (TypeError, ValueError): return None, "All feature values must be integers."
    bad = [f for f, v in zip(FEATS, row) if v not in ALLOWED[f]]
    return (row, None) if not bad else (None, f"Invalid value for: {', '.join(bad[:5])}")

@app.post("/api/predict")
def predict():
    """Manual mode: the 30 UCI features, any of the five models (or the best one)."""
    data = request.get_json(silent=True) or {}
    name = METRICS["best"] if data.get("model", "best") == "best" else data["model"]
    if name not in MODELS: return jsonify(error=f"Unknown model '{name}'"), 400
    row, err = manual_validate(data.get("features") or {})
    if err: return jsonify(error=err), 400
    model = MODELS[name]; p = float(model.predict_proba(pd.DataFrame([row], columns=FEATS))[0, 1])
    label = "PHISHING" if p >= 0.5 else "LEGITIMATE"
    imp = counterfactual(model, FEATS, row, p, lambda c, v: -v if v else 1)       # flip each value to its opposite
    items = [dict(feature=c, title=INFO[c]["title"], value=v, shown=INFO[c]["options"][v].lower(), impact=float(d)) for c, v, d in zip(FEATS, row, imp)]
    items.sort(key=lambda i: -abs(i["impact"]))
    res = dict(label=label, note=None, probability_phishing=p, confidence=max(p, 1 - p), model=name, mode="full", explanation=items[:8],
               impacts={i["feature"]: i["impact"] for i in items}, titles={c: INFO[c]["title"] for c in FEATS}, values=dict(zip(FEATS, row)),
               summary=summarize(label, items), place=place("uci", row), time=datetime.now().strftime("%H:%M:%S"))
    HISTORY.insert(0, {k: res[k] for k in ("time", "model", "label", "confidence")}); del HISTORY[200:]
    return jsonify(res)

@app.post("/api/analyze")
def analyze():
    """AI mode: URL in, verdict out."""
    if not AI_MODELS: return jsonify(error="AI models missing - run train_ai.py"), 503
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip(); name = AI_BEST if data.get("model", "best") == "best" else data["model"]
    if not url or len(url) > 2000: return jsonify(error="Enter a valid URL (max 2000 chars)."), 400
    if name not in AI_MODELS["page"]: return jsonify(error="Unknown model."), 400
    try: res = ai_run(url, name)
    except ValueError as e: return jsonify(error=str(e)), 400
    HISTORY.insert(0, {k: res[k] for k in ("time", "model", "label", "confidence")}); del HISTORY[200:]
    return jsonify(res)

@app.post("/api/compare")
def compare():
    """Score the same input with all five models (manual features, or AI-mode feature values)."""
    d = request.get_json(silent=True) or {}; feats = d.get("features") or {}; mode = d.get("mode")
    try:
        if mode in ("url", "page"):
            cols = AI_SETS[mode]; X = pd.DataFrame([[float(feats[c]) for c in cols]], columns=cols); ms = AI_MODELS[mode]
        else:
            row, err = manual_validate(feats)
            if err: return jsonify(error=err), 400
            X = pd.DataFrame([row], columns=FEATS); ms = MODELS
    except (KeyError, TypeError, ValueError): return jsonify(error="Invalid or missing features."), 400
    out = [dict(model=k, probability_phishing=(p := float(m.predict_proba(X)[0, 1])), label="PHISHING" if p >= 0.5 else "LEGITIMATE") for k, m in ms.items()]
    return jsonify(results=out, phishing_votes=sum(o["label"] == "PHISHING" for o in out))

@app.post("/api/batch")
def batch():
    d = request.get_json(silent=True) or {}
    urls = [u.strip() for u in (d.get("urls") or []) if isinstance(u, str) and u.strip()][:8]
    name = AI_BEST if d.get("model", "best") == "best" else d.get("model")
    if not AI_MODELS or name not in AI_MODELS["page"]: return jsonify(error="AI models missing or unknown model."), 400
    if not urls: return jsonify(error="Enter at least one URL."), 400
    def one(u):
        try: r = ai_run(u[:2000], name); return dict(url=u, label=r["label"], confidence=r["confidence"], probability_phishing=r["probability_phishing"], mode=r["mode"], note=r["note"])
        except ValueError as e: return dict(url=u, error=str(e))
    with ThreadPoolExecutor(max_workers=8) as ex: rows = list(ex.map(one, urls))
    for r in rows:
        if "label" in r: HISTORY.insert(0, dict(time=datetime.now().strftime("%H:%M:%S"), model=name, label=r["label"], confidence=r["confidence"]))
    return jsonify(model=name, results=rows)

@app.get("/api/history")
def history():
    ph = sum(h["label"] == "PHISHING" for h in HISTORY); lg = sum(h["label"] == "LEGITIMATE" for h in HISTORY)
    return jsonify(history=HISTORY[:10], total=len(HISTORY), phishing=ph, legitimate=lg, other=len(HISTORY) - ph - lg)

@app.get("/api/report")
def report():
    out = io.StringIO(); w = csv.DictWriter(out, ["time", "model", "label", "confidence"]); w.writeheader(); w.writerows(HISTORY)
    return Response(out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})

if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
