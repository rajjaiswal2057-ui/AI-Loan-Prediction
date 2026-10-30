"""Prediction, Financial Risk Score, explanations and What-If analysis.

RISK SCORE FORMULA (project-generated, NOT an official credit score), 0 = safest, 100 = riskiest:

  Risk = 100 * [ 0.45*(1 - P_approve)      model probability of approval
               + 0.25*(1 - C)              credit strength   C = (CIBIL - 300) / 600
               + 0.15*B                    loan burden       B = scaled loan-to-income ratio
               + 0.10*(1 - A)              asset strength    A = scaled (total assets / loan amount)
               + 0.05*(1 - I) ]            income strength   I = scaled annual income
  B, A, I are min-max scaled between the 5th and 95th percentile of the training data, clipped to [0,1].
  Bands:  0-30 Low | 31-60 Medium | 61-100 High
"""
import json
import numpy as np
from .logistic_regression import LogisticRegressionScratch
from .preprocessing import StandardScaler, FEATURES

LABELS = {"no_of_dependents": "Dependents", "education": "Education (Graduate)", "self_employed": "Self-employed",
          "income_annum": "Annual income", "loan_amount": "Loan amount", "loan_term": "Loan term",
          "cibil_score": "CIBIL score", "residential_assets_value": "Residential assets",
          "commercial_assets_value": "Commercial assets", "luxury_assets_value": "Luxury assets",
          "bank_asset_value": "Bank assets"}
WEIGHTS = {"model": 0.45, "credit": 0.25, "burden": 0.15, "assets": 0.10, "income": 0.05}


def clip01(x):
    return float(min(1.0, max(0.0, x)))


def risk_category(score):
    return "LOW" if score <= 30 else "MEDIUM" if score <= 60 else "HIGH"


def confidence_level(p):
    margin = abs(2 * p - 1)            # 0 = coin-flip, 1 = fully certain
    return "HIGH" if margin >= 0.6 else "MEDIUM" if margin >= 0.3 else "LOW"


def fmt_money(v):
    return f"₹{v/1e7:.2f} Cr" if v >= 1e7 else f"₹{v/1e5:.1f} L"


class Predictor:
    def __init__(self, path):
        art = json.load(open(path))
        self.model = LogisticRegressionScratch.from_dict(art["model"])
        self.scaler = StandardScaler.from_dict(art["scaler"])
        self.bounds = art["risk_bounds"]
        self.ranges = art["ranges"]

    def range_warnings(self, a):
        """Logistic regression extrapolates linearly - tell the user when inputs leave the training range."""
        names = {"cibil": "CIBIL score", "income": "Annual income", "loan_amount": "Loan amount", "loan_term": "Loan term"}
        w = [f"{n} is outside the range seen in training data" for k, n in names.items()
             if not self.ranges[k][0] <= a[k] <= self.ranges[k][1]]
        lti = a["loan_amount"] / max(a["income"], 1)
        if not self.ranges["lti"][0] <= lti <= self.ranges["lti"][1]:
            w.append(f"Loan-to-income ratio ({lti:.1f}x) is outside the {self.ranges['lti'][0]:.1f}x-{self.ranges['lti'][1]:.1f}x range seen in training data")
        return w

    @staticmethod
    def vector(a):
        assets = [a["residential"], a["commercial"], a["luxury"], a["bank"]]
        total = float(sum(assets))
        row = [a["dependents"], 1 if a["education"] == "Graduate" else 0, 1 if a["self_employed"] == "Yes" else 0,
               a["income"], a["loan_amount"], a["loan_term"], a["cibil"], *assets]
        return np.array(row, dtype=float), total

    def components(self, a, p):
        _, total = self.vector(a)
        b = self.bounds
        scale = lambda v, k: clip01((v - b[k][0]) / (b[k][1] - b[k][0]))
        lti = a["loan_amount"] / max(a["income"], 1)
        return {"credit": clip01((a["cibil"] - 300) / 600), "burden": scale(lti, "lti"),
                "assets": scale(total / max(a["loan_amount"], 1), "atl"), "income": scale(a["income"], "income")}

    def predict(self, a, explain=True):
        x, total = self.vector(a)
        z = self.scaler.transform(x)
        p = float(self.model.predict_proba(z))
        c = self.components(a, p)
        risk = 100 * (WEIGHTS["model"] * (1 - p) + WEIGHTS["credit"] * (1 - c["credit"]) + WEIGHTS["burden"] * c["burden"]
                      + WEIGHTS["assets"] * (1 - c["assets"]) + WEIGHTS["income"] * (1 - c["income"]))
        risk = int(round(risk))
        lti = a["loan_amount"] / max(a["income"], 1)
        out = {"warnings": self.range_warnings(a), "decision": "APPROVED" if p >= self.model.threshold else "REJECTED", "probability": round(p * 100, 2),
               "risk_score": risk, "risk_level": risk_category(risk), "confidence": confidence_level(p),
               "summary": {"cibil": a["cibil"], "income": a["income"], "loan_amount": a["loan_amount"],
                           "loan_to_income": round(lti, 2), "total_assets": total, "loan_term": a["loan_term"]},
               "health": {"credit": round(c["credit"] * 100), "income": round(c["income"] * 100),
                          "burden": round(c["burden"] * 100), "assets": round(c["assets"] * 100)},
               "risk_formula": {"weights": WEIGHTS, "parts": {
                   "model": round(100 * WEIGHTS["model"] * (1 - p), 1), "credit": round(100 * WEIGHTS["credit"] * (1 - c["credit"]), 1),
                   "burden": round(100 * WEIGHTS["burden"] * c["burden"], 1), "assets": round(100 * WEIGHTS["assets"] * (1 - c["assets"]), 1),
                   "income": round(100 * WEIGHTS["income"] * (1 - c["income"]), 1)}}}
        if explain:
            out["factors"] = self.explain(a, lti, total, z)
        return out

    def explain(self, a, lti, total, z):
        t = self.bounds["thresholds"]
        pos, neg = [], []
        cibil = a["cibil"]
        if cibil >= 750: pos.append(f"Excellent CIBIL score ({cibil})")
        elif cibil >= 700: pos.append(f"Good CIBIL score ({cibil})")
        elif cibil < 550: neg.append(f"Low CIBIL score ({cibil})")
        elif cibil < 650: neg.append(f"Below-average CIBIL score ({cibil})")
        if lti <= t["lti"][0]: pos.append(f"Manageable loan-to-income ratio ({lti:.1f}x)")
        elif lti >= t["lti"][1]: neg.append(f"High loan-to-income ratio ({lti:.1f}x)")
        atl = total / max(a["loan_amount"], 1)
        if atl >= t["atl"][1]: pos.append(f"Strong asset coverage ({atl:.1f}x the loan amount)")
        elif atl <= t["atl"][0]: neg.append(f"Weak asset coverage ({atl:.1f}x the loan amount)")
        if a["income"] >= t["income"][1]: pos.append(f"Healthy annual income ({fmt_money(a['income'])})")
        elif a["income"] <= t["income"][0]: neg.append(f"Low annual income ({fmt_money(a['income'])})")
        if a["loan_amount"] >= t["loan"][1]: neg.append(f"High loan amount ({fmt_money(a['loan_amount'])})")
        elif a["loan_amount"] <= t["loan"][0]: pos.append(f"Modest loan amount ({fmt_money(a['loan_amount'])})")
        # Model-driven: contribution of each feature to the log-odds = weight * standardised value
        contrib = self.model.weights * z
        order = np.argsort(contrib)
        drivers = [{"feature": LABELS[FEATURES[i]], "impact": round(float(contrib[i]), 3)} for i in order[::-1]]
        return {"positive": pos, "negative": neg,
                "drivers_up": [d for d in drivers if d["impact"] > 0][:4],
                "drivers_down": [d for d in drivers[::-1] if d["impact"] < 0][:4]}

    def simulate(self, current, improved):
        cur, imp = self.predict(current, False), self.predict(improved, False)
        labels = {"cibil": "CIBIL score", "income": "annual income", "loan_amount": "loan amount", "loan_term": "loan term",
                  "residential": "residential assets", "commercial": "commercial assets", "luxury": "luxury assets", "bank": "bank assets"}
        steps = []
        for k, name in labels.items():
            if current[k] != improved[k]:
                one = dict(current); one[k] = improved[k]
                dp = self.predict(one, False)["probability"] - cur["probability"]
                f = (lambda v: str(v)) if k in ("cibil", "loan_term") else fmt_money
                verb = "Increasing" if improved[k] > current[k] else "Reducing"
                eff = "improved" if dp > 0.005 else "lowered" if dp < -0.005 else "did not noticeably change"
                steps.append({"field": k, "delta": round(dp, 2),
                              "text": f"{verb} the {name} from {f(current[k])} to {f(improved[k])} {eff} the predicted approval probability"
                                      + (f" by {abs(dp):.1f} points." if abs(dp) > 0.005 else ".")})
        return {"current": cur, "improved": imp, "change": round(imp["probability"] - cur["probability"], 2), "steps": steps}
