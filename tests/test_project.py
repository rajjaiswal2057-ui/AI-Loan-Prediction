"""Quick sanity tests.  Run from project root:  python tests/test_project.py"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "app"))
import numpy as np
from ml.logistic_regression import sigmoid, LogisticRegressionScratch
from ml.metrics import confusion_matrix, roc_curve, auc
from app import app

GOOD = {"dependents": 2, "education": "Graduate", "self_employed": "No", "income": 5000000, "loan_amount": 15000000,
        "loan_term": 12, "cibil": 750, "residential": 2000000, "commercial": 3000000, "luxury": 10000000, "bank": 4000000}
c = app.test_client()
post = lambda path, d: c.post(path, data=json.dumps(d), content_type="application/json")


def test_sigmoid():
    assert sigmoid(0) == 0.5 and 0 <= sigmoid(-1000) and sigmoid(1000) <= 1


def test_learns_simple_rule():
    rng = np.random.default_rng(0); X = rng.normal(size=(300, 2)); y = (X[:, 0] > 0).astype(int)
    assert (LogisticRegressionScratch(0.5, 500).fit(X, y).predict(X) == y).mean() > 0.95


def test_metrics():
    y = np.array([0, 0, 1, 1]); p = np.array([.1, .4, .35, .8])
    assert confusion_matrix(y, (p >= .5).astype(int)) == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}
    assert abs(auc(*roc_curve(y, p)) - 0.75) < 1e-9          # matches the textbook value


def test_predict_endpoint():
    r = post("/api/predict", GOOD); j = r.get_json()
    assert r.status_code == 200 and j["decision"] == "APPROVED" and 0 <= j["risk_score"] <= 100
    assert post("/api/predict", {**GOOD, "cibil": 400}).get_json()["decision"] == "REJECTED"


def test_bad_input_is_400_json():
    for bad in ({**GOOD, "cibil": "abc"}, {**GOOD, "income": "nan"}, {**GOOD, "loan_amount": "inf"},
                {**GOOD, "cibil": 1000}, {**GOOD, "income": -5}, [1, 2]):
        r = post("/api/predict", bad); assert r.status_code == 400 and "error" in r.get_json(), bad
    r = c.post("/api/predict", data="{bad", content_type="application/json")
    assert r.status_code == 400 and "error" in r.get_json()


def test_simulate():
    j = post("/api/simulate", {"current": {**GOOD, "cibil": 450}, "improved": GOOD}).get_json()
    assert j["change"] > 0 and post("/api/simulate", {"current": GOOD}).status_code == 400


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
