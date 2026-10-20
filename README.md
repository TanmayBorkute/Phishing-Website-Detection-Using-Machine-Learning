# PhishGuard AI – Phishing Website Detection Using ML

Two data sources, two modes:
| Mode | Data | Input | Models |
|---|---|---|---|
| **Manual** | UCI Phishing Websites #327 (`data/Training_Dataset.arff`, 11,055 rows -> 5,849 unique) | the 30 dataset features | Decision Tree, KNN, Naive Bayes, SVM, Boosting |
| **AI Auto-Analyze** | PhiUSIIL Phishing URL #967 (`data/PhiUSIIL_Phishing_URL_Dataset.csv`, 235,795 rows) | a raw URL | the same five algorithms, trained on domain-name features (+ page signals if the page can be fetched) |

## Structure
app.py · train.py (UCI-327) · train_ai.py (PhiUSIIL) · features.py (UCI rules, option labels) · urlfeatures.py (AI-mode extractor) · tests/ · templates/ · static/css · static/js · data/ · models/

## Run
    python -m venv venv && venv\Scripts\activate      # Mac/Linux: source venv/bin/activate
    pip install -r requirements.txt
    # data/ needs: Training_Dataset.arff, _old.arff (or old_dataset.arff), PhiUSIIL_Phishing_URL_Dataset.csv
    #   PhiUSIIL: https://archive.ics.uci.edu/dataset/967  (train_ai.py also tries to download it itself)
    python train.py        # UCI-327 models, feature-space map, threshold curves
    python train_ai.py     # PhiUSIIL AI-mode models (about 6 minutes)
    python tests/test_features.py
    python app.py          # http://localhost:5000   (Ctrl+K opens the command palette)

## Honest notes (useful for the viva)
- Duplicates removed (UCI-327: 5,206 exact duplicates) so train and test never share a row.
- PhiUSIIL artifact: every legitimate URL in it is a bare `https://www.<domain>` homepage, so https / www / path / query features (and even dot counts
  that include "www.") separate the classes perfectly but meaninglessly: 99.8% accuracy, yet `github.com/...` was flagged as phishing.
  They are excluded; the models use only domain-name features with `www.` stripped, and the split is grouped by domain.
  Result: about 84-86% (domain only) and about 98% (domain + page signals).
- Page signals are simple HTML checks (password field, favicon, social links...). Phishing pages in this dataset are mostly bare or dead, so modern kits
  that copy real pages can evade them. A safety layer downgrades LEGITIMATE to SUSPICIOUS (IP address, '@') or INCONCLUSIVE (low confidence / page not fetched).
- Explanations are counterfactual: each feature is replaced by a neutral value and the drop in P(phishing) is shown.

## Deploy (Render)
Build: `pip install -r requirements.txt && python train.py && python train_ai.py` · Start: `gunicorn app:app --timeout 90`
(AI training needs the PhiUSIIL CSV at build time, or commit the generated `models/` folder instead.)
