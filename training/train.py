"""Train the from-scratch Logistic Regression, evaluate it and save every artefact the web app needs.
Run from project root:  python training/train.py"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # work from any directory
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ml.preprocessing import load_dataset, clean, add_features, StandardScaler, stratified_split, FEATURES, WITH_RATIOS
from ml.logistic_regression import LogisticRegressionScratch, binary_cross_entropy
from ml.metrics import classification_report, roc_curve
from ml.risk_engine import Predictor, LABELS

LR, EPOCHS, LAMBDAS = 0.1, 2000, [0, 0.0001, 0.001, 0.01, 0.1, 1.0]
df = add_features(clean(load_dataset("data/loan_dataset.csv")))
X, y = df[FEATURES].values.astype(float), df["loan_status"].values
tr, va, te = stratified_split(y)
scaler = StandardScaler().fit(X[tr])                 # fit on training rows only
Xtr, Xva, Xte = (scaler.transform(X[i]) for i in (tr, va, te))
ytr, yva, yte = y[tr], y[va], y[te]
print(f"split: train={len(tr)} val={len(va)} test={len(te)}")

# --- 1. regularisation study: choose lambda on the VALIDATION set ---
study = []
for lam in LAMBDAS:
    m = LogisticRegressionScratch(LR, EPOCHS, lam).fit(Xtr, ytr, Xva, yva)
    rt = classification_report(ytr, m.predict_proba(Xtr)); rv = classification_report(yva, m.predict_proba(Xva))
    study.append({"l2": lam, "train_acc": rt["accuracy"], "val_acc": rv["accuracy"],
                  "train_loss": binary_cross_entropy(ytr, m.predict_proba(Xtr)),
                  "val_loss": binary_cross_entropy(yva, m.predict_proba(Xva)),
                  "weight_norm": float(np.linalg.norm(m.weights))})
    print(study[-1])
# Parsimony rule: strongest lambda whose validation loss is within 1% of the best (prefers the simpler model).
min_val = min(s["val_loss"] for s in study)
best = max(s["l2"] for s in study if s["val_loss"] <= min_val * 1.01)
print("selected lambda:", best)

# --- 2. underfitting demonstration (too few / too small updates) ---
under = LogisticRegressionScratch(0.001, 20, 0.0).fit(Xtr, ytr, Xva, yva)
under_rep = {"train_acc": float(np.mean(under.predict(Xtr) == ytr)), "test_acc": float(np.mean(under.predict(Xte) == yte)),
             "train_loss": under.history["train_loss"][-1], "train_auc": classification_report(ytr, under.predict_proba(Xtr))["roc_auc"], "history": under.history}

# --- 3. final model ---
model = LogisticRegressionScratch(LR, EPOCHS, best).fit(Xtr, ytr, Xva, yva)
rep = {n: classification_report(yy, model.predict_proba(XX)) for n, XX, yy in
       (("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte))}
gap = rep["train"]["accuracy"] - rep["test"]["accuracy"]
verdict = ("UNDERFITTING" if rep["train"]["accuracy"] < 0.80 else "OVERFITTING" if gap > 0.05 else "WELL FITTED")
print({k: round(v["accuracy"], 4) for k, v in rep.items()}, "gap", round(gap, 4), verdict)

# --- 3b. ablation: do the engineered ratio features help? ---
Xr = df[WITH_RATIOS].values.astype(float); sr = StandardScaler().fit(Xr[tr])
mr = LogisticRegressionScratch(LR, EPOCHS, best).fit(sr.transform(Xr[tr]), ytr)
rr = classification_report(yte, mr.predict_proba(sr.transform(Xr[te])))
ablation = {"raw_features": {"n": len(FEATURES), "test_acc": rep["test"]["accuracy"], "test_auc": rep["test"]["roc_auc"]},
            "with_ratios": {"n": len(WITH_RATIOS), "test_acc": rr["accuracy"], "test_auc": rr["roc_auc"]}}
print("ablation", ablation)
ranges = {"cibil": [int(df.cibil_score.min()), int(df.cibil_score.max())], "income": [int(df.income_annum.min()), int(df.income_annum.max())],
          "loan_amount": [int(df.loan_amount.min()), int(df.loan_amount.max())], "loan_term": [int(df.loan_term.min()), int(df.loan_term.max())],
          "lti": [float(df.loan_to_income.min()), float(df.loan_to_income.max())]}

# --- 4. risk-score bounds (training data only) ---
d_tr = df.iloc[tr]
pc = lambda s, a, b: [float(np.percentile(s, a)), float(np.percentile(s, b))]
bounds = {"lti": pc(d_tr.loan_to_income, 5, 95), "atl": pc(d_tr.assets_to_loan, 5, 95), "income": pc(d_tr.income_annum, 5, 95),
          "thresholds": {"lti": pc(d_tr.loan_to_income, 33, 67), "atl": pc(d_tr.assets_to_loan, 33, 67),
                         "income": pc(d_tr.income_annum, 33, 67), "loan": pc(d_tr.loan_amount, 33, 67)}}
json.dump({"model": model.to_dict(), "scaler": scaler.to_dict(), "features": FEATURES, "risk_bounds": bounds, "ranges": ranges},
          open("model/saved_model/model.json", "w"), indent=1)

# --- 5. whole-dataset predictions for dashboard + test-set curves ---
pred = Predictor("model/saved_model/model.json")
pall = model.predict_proba(scaler.transform(X))
risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
for i in range(len(df)):
    r = df.iloc[i]
    a = {"dependents": r.no_of_dependents, "education": "Graduate" if r.education else "Not Graduate",
         "self_employed": "Yes" if r.self_employed else "No", "income": r.income_annum, "loan_amount": r.loan_amount,
         "loan_term": r.loan_term, "cibil": r.cibil_score, "residential": r.residential_assets_value,
         "commercial": r.commercial_assets_value, "luxury": r.luxury_assets_value, "bank": r.bank_asset_value}
    risk_counts[pred.predict(a, False)["risk_level"]] += 1
bins = np.linspace(0, 1, 11)
hist = lambda v: np.histogram(v, bins)[0].tolist()
ptest = model.predict_proba(Xte)
fpr, tpr = roc_curve(yte, ptest)
step = max(1, len(fpr) // 200)
w = model.weights
coef = sorted([{"feature": LABELS[f], "weight": float(w[i])} for i, f in enumerate(FEATURES)], key=lambda d: -abs(d["weight"]))
out = {"config": {"lr": LR, "epochs": EPOCHS, "l2": best, "threshold": 0.5, "n_features": len(FEATURES), "bias": model.bias},
       "split": {"train": len(tr), "val": len(va), "test": len(te)}, "metrics": rep, "verdict": verdict, "acc_gap": gap,
       "history": model.history, "study": study, "ablation": ablation, "ranges": ranges, "underfit": under_rep,
       "roc": {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()}, "coefficients": coef,
       "prob_bins": [f"{int(b*100)}-{int(b*100)+10}%" for b in bins[:-1]],
       "prob_hist_all": hist(pall), "prob_hist_test_approved": hist(ptest[yte == 1]), "prob_hist_test_rejected": hist(ptest[yte == 0]),
       "risk_counts": risk_counts, "pred_test": {"actual_approved": int(yte.sum()), "pred_approved": int((ptest >= .5).sum()), "n": len(yte)}}
json.dump(out, open("reports/results.json", "w"), indent=1)

# --- 6. figures for the report ---
F = "reports/figures/"
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(model.history["train_loss"], label="Train"); ax[0].plot(model.history["val_loss"], label="Validation")
ax[0].set(title="Loss vs Epoch (BCE)", xlabel="Epoch", ylabel="Loss"); ax[0].legend()
ax[1].plot(model.history["train_acc"], label="Train"); ax[1].plot(model.history["val_acc"], label="Validation")
ax[1].set(title="Accuracy vs Epoch", xlabel="Epoch", ylabel="Accuracy"); ax[1].legend(); plt.tight_layout(); plt.savefig(F + "loss_curve.png", dpi=150); plt.close()
plt.figure(figsize=(4.5, 4.5)); plt.plot(fpr, tpr, label=f"AUC = {rep['test']['roc_auc']:.3f}"); plt.plot([0, 1], [0, 1], "--", c="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve (test)"); plt.legend(); plt.tight_layout(); plt.savefig(F + "roc.png", dpi=150); plt.close()
c = rep["test"]["confusion"]; M = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
plt.figure(figsize=(4.5, 4)); plt.imshow(M, cmap="Blues")
for i in range(2):
    for j in range(2): plt.text(j, i, M[i, j], ha="center", va="center", fontsize=16, color="white" if M[i, j] > M.max() / 2 else "black")
plt.xticks([0, 1], ["Rejected", "Approved"]); plt.yticks([0, 1], ["Rejected", "Approved"]); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix (test)"); plt.tight_layout(); plt.savefig(F + "confusion.png", dpi=150); plt.close()
plt.figure(figsize=(6, 4)); plt.semilogx([max(s["l2"], 1e-5) for s in study], [s["train_acc"] for s in study], "o-", label="Train")
plt.semilogx([max(s["l2"], 1e-5) for s in study], [s["val_acc"] for s in study], "s-", label="Validation")
plt.xlabel("L2 strength λ (0 plotted at 1e-5)"); plt.ylabel("Accuracy"); plt.title("Regularisation study"); plt.legend(); plt.tight_layout(); plt.savefig(F + "regularisation.png", dpi=150); plt.close()
plt.figure(figsize=(6, 4)); plt.bar(np.arange(10) - .2, out["prob_hist_test_approved"], .4, label="Actually approved")
plt.bar(np.arange(10) + .2, out["prob_hist_test_rejected"], .4, label="Actually rejected"); plt.xticks(range(10), [f"{i*10}" for i in range(10)])
plt.xlabel("Predicted approval probability (%)"); plt.ylabel("Applicants"); plt.title("Actual vs predicted distribution (test)"); plt.legend(); plt.tight_layout(); plt.savefig(F + "prob_dist.png", dpi=150); plt.close()
plt.figure(figsize=(6, 4.5)); plt.barh([d["feature"] for d in coef][::-1], [d["weight"] for d in coef][::-1],
                                        color=["#16a34a" if d["weight"] > 0 else "#dc2626" for d in coef][::-1])
plt.title("Learned coefficients (standardised features)"); plt.tight_layout(); plt.savefig(F + "weights.png", dpi=150); plt.close()
print("saved")
