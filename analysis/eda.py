"""Exploratory data analysis: prints the audit, saves reports/eda.json (for the web app) and figures (for the report).
Run from project root:  python analysis/eda.py"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # work from any directory
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ml.preprocessing import load_dataset, clean, add_features, ASSET_COLS

raw = load_dataset("data/loan_dataset.csv")
df = add_features(clean(raw))
F = "reports/figures/"
num = raw.drop(columns=["loan_id"]).select_dtypes("number")
q1, q3 = num.quantile(.25), num.quantile(.75); iqr = q3 - q1
outliers = ((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).sum()
audit = {"rows": len(raw), "columns": raw.shape[1], "missing": int(raw.isna().sum().sum()), "duplicates": int(raw.duplicated().sum()),
         "negative_residential": int((raw["residential_assets_value"] < 0).sum()),
         "approved": int(df.loan_status.sum()), "rejected": int((1 - df.loan_status).sum()),
         "outliers": {k: int(v) for k, v in outliers.items() if v > 0},
         "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
         "lti_range": [float(df.loan_to_income.min()), float(df.loan_to_income.max())]}
audit["approval_pct"] = round(100 * audit["approved"] / audit["rows"], 2)
stats = df.drop(columns=["loan_status", "total_assets", "loan_to_income", "assets_to_loan"]).describe().T[["mean", "std", "min", "50%", "max"]]
audit["stats"] = {k: {a: float(b) for a, b in v.items()} for k, v in stats.iterrows()}
print(json.dumps({k: v for k, v in audit.items() if k != "stats"}, indent=1))


def hist(s, lo, hi, n):
    edges = np.linspace(lo, hi, n + 1)
    return {"labels": [f"{edges[i]/1e6:.1f}" if hi > 1e5 else f"{int(edges[i])}" for i in range(n)],
            "all": np.histogram(s, edges)[0].tolist(),
            "approved": np.histogram(s[df.loan_status == 1], edges)[0].tolist(),
            "rejected": np.histogram(s[df.loan_status == 0], edges)[0].tolist()}


bands = pd.cut(df.cibil_score, [299, 400, 500, 550, 600, 700, 800, 900])
rate_cibil = df.groupby(bands, observed=True).loan_status.agg(["mean", "count"])
corr = df[["no_of_dependents", "education", "self_employed", "income_annum", "loan_amount", "loan_term", "cibil_score"] + ASSET_COLS + ["loan_status"]].corr()
smp = df.sample(500, random_state=1)
eda = {"audit": audit,
       "cibil": hist(df.cibil_score, 300, 900, 12), "income": hist(df.income_annum, 0, 10e6, 10),
       "loan": hist(df.loan_amount, 0, 40e6, 10), "assets": hist(df.total_assets, 0, 80e6, 10),
       "cibil_rate": {"labels": [str(i) for i in rate_cibil.index], "rate": (rate_cibil["mean"] * 100).round(1).tolist(), "count": rate_cibil["count"].tolist()},
       "education": {k: round(100 * df[df.education == v].loan_status.mean(), 1) for k, v in (("Graduate", 1), ("Not Graduate", 0))},
       "self_employed": {k: round(100 * df[df.self_employed == v].loan_status.mean(), 1) for k, v in (("Self-employed", 1), ("Salaried / Other", 0))},
       "scatter": [{"x": float(r.income_annum) / 1e6, "y": float(r.loan_amount) / 1e6, "s": int(r.loan_status)} for r in smp.itertuples()],
       "asset_means": {c.replace("_assets_value", "").replace("_asset_value", "").title(): [float(df[df.loan_status == 1][c].mean()), float(df[df.loan_status == 0][c].mean())] for c in ASSET_COLS},
       "corr": {"labels": [c.replace("_value", "").replace("_assets", "_assets") for c in corr.columns], "matrix": corr.round(2).values.tolist()},
       "dash": {"total": len(df), "approved": audit["approved"], "rejected": audit["rejected"], "approval_rate": audit["approval_pct"],
                "avg_cibil": round(float(df.cibil_score.mean()), 1), "avg_loan": float(df.loan_amount.mean())}}
json.dump(eda, open("reports/eda.json", "w"))

# ---- figures for the report ----
C = {1: "#16a34a", 0: "#dc2626"}
fig, ax = plt.subplots(2, 4, figsize=(18, 8))
ax[0, 0].bar(["Approved", "Rejected"], [audit["approved"], audit["rejected"]], color=[C[1], C[0]]); ax[0, 0].set_title("Approval distribution")
for st in (1, 0): ax[0, 1].hist(df[df.loan_status == st].cibil_score, bins=24, alpha=.65, color=C[st], label=["Rejected", "Approved"][st])
ax[0, 1].set_title("CIBIL distribution by outcome"); ax[0, 1].legend()
ax[0, 2].scatter(smp.income_annum / 1e6, smp.loan_amount / 1e6, c=smp.loan_status.map(C), s=8); ax[0, 2].set(title="Income vs Loan amount (₹ M)", xlabel="Income", ylabel="Loan")
ax[0, 3].bar([str(i) for i in rate_cibil.index], rate_cibil["mean"] * 100, color="#2563eb"); ax[0, 3].set_title("Approval rate by CIBIL band (%)"); ax[0, 3].tick_params(axis="x", rotation=45)
ax[1, 0].bar(eda["education"].keys(), eda["education"].values(), color="#2563eb"); ax[1, 0].set_title("Approval % by education")
ax[1, 1].bar(eda["self_employed"].keys(), eda["self_employed"].values(), color="#2563eb"); ax[1, 1].set_title("Approval % by employment")
ax[1, 2].hist(df.total_assets / 1e6, bins=30, color="#7c3aed"); ax[1, 2].set_title("Total assets (₹ M)")
im = ax[1, 3].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1); ax[1, 3].set_title("Correlation heatmap")
ax[1, 3].set_xticks(range(len(corr))); ax[1, 3].set_xticklabels([c[:8] for c in corr.columns], rotation=90, fontsize=6); ax[1, 3].set_yticks(range(len(corr))); ax[1, 3].set_yticklabels([c[:8] for c in corr.columns], fontsize=6)
plt.tight_layout(); plt.savefig(F + "eda_overview.png", dpi=130); plt.close()
print("corr with status:", corr["loan_status"].round(3).to_dict())
