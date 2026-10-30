"""Flask backend. Loads the trained model ONCE at start-up; every request only does a forward pass (no retraining).
Run from project root:  python app/app.py   ->  http://127.0.0.1:5000"""
import sys, os, json, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flask import Flask, jsonify, request, render_template
from ml.risk_engine import Predictor

app = Flask(__name__)
predictor = Predictor(os.path.join(ROOT, "model/saved_model/model.json"))
RESULTS = json.load(open(os.path.join(ROOT, "reports/results.json")))
EDA = json.load(open(os.path.join(ROOT, "reports/eda.json")))

NUM = ["dependents", "income", "loan_amount", "loan_term", "cibil", "residential", "commercial", "luxury", "bank"]


def parse(d):
    """Validate and normalise an applicant dictionary; raises ValueError with a readable message."""
    if not isinstance(d, dict):
        raise ValueError("Request body must be a JSON object")
    a = {}
    for k in NUM:
        try:
            a[k] = float(d[k])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"'{k}' is missing or not a number")
        if not math.isfinite(a[k]):
            raise ValueError(f"'{k}' must be a finite number")
        if a[k] < 0:
            raise ValueError(f"'{k}' cannot be negative")
    if not 300 <= a["cibil"] <= 900:
        raise ValueError("CIBIL score must be between 300 and 900")
    if a["income"] <= 0 or a["loan_amount"] <= 0 or a["loan_term"] <= 0:
        raise ValueError("Income, loan amount and loan term must be greater than zero")
    if max(a[k] for k in NUM) > 1e13:
        raise ValueError("One of the values is unrealistically large")
    for k in ("dependents", "loan_term", "cibil"):
        a[k] = int(round(a[k]))
    a["education"] = "Graduate" if d.get("education") == "Graduate" else "Not Graduate"
    a["self_employed"] = "Yes" if d.get("self_employed") == "Yes" else "No"
    return a


def body():
    """Parse the JSON body; malformed JSON becomes a readable ValueError instead of an HTML error page."""
    data = request.get_json(force=True, silent=True)
    if data is None:
        raise ValueError("Request body must be valid JSON")
    return data


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/bootstrap")
def bootstrap():
    return jsonify({"eda": EDA, "results": RESULTS})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        return jsonify(predictor.predict(parse(body())))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/simulate", methods=["POST"])
def simulate():
    try:
        data = body()
        return jsonify(predictor.simulate(parse(data["current"]), parse(data["improved"])))
    except KeyError as e:
        return jsonify({"error": f"Missing field {e}"}), 400
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
