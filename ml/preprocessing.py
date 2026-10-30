"""Data loading, cleaning, feature engineering, scaling and splitting (all from scratch)."""
import numpy as np
import pandas as pd

RAW_FEATURES = ["no_of_dependents", "education", "self_employed", "income_annum", "loan_amount",
                "loan_term", "cibil_score", "residential_assets_value", "commercial_assets_value",
                "luxury_assets_value", "bank_asset_value"]
ENGINEERED = ["loan_to_income", "assets_to_loan"]
FEATURES = RAW_FEATURES                      # features fed to the model (ablation showed ratios add nothing)
WITH_RATIOS = RAW_FEATURES + ENGINEERED      # used only for the ablation study
ASSET_COLS = ["residential_assets_value", "commercial_assets_value", "luxury_assets_value", "bank_asset_value"]


def load_dataset(path):
    """Read CSV; the raw file has leading spaces in headers and string values."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()
    return df


def clean(df):
    """Drop ID, clip impossible negative asset values to 0, encode categoricals and target."""
    df = df.drop(columns=["loan_id"]).copy()
    for c in ASSET_COLS:
        df[c] = df[c].clip(lower=0)
    df["education"] = (df["education"] == "Graduate").astype(int)      # 1 = Graduate
    df["self_employed"] = (df["self_employed"] == "Yes").astype(int)   # 1 = Yes
    df["loan_status"] = (df["loan_status"] == "Approved").astype(int)  # 1 = Approved
    return df


def add_features(df):
    """Feature engineering: two ratio features that mirror how lenders reason."""
    df = df.copy()
    df["total_assets"] = df[ASSET_COLS].sum(axis=1)
    df["loan_to_income"] = df["loan_amount"] / df["income_annum"].clip(lower=1)
    df["assets_to_loan"] = df["total_assets"] / df["loan_amount"].clip(lower=1)
    return df


class StandardScaler:
    """z = (x - mean) / std, fitted on TRAINING data only (prevents data leakage)."""
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def to_dict(self):
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d):
        s = cls(); s.mean_ = np.array(d["mean"]); s.std_ = np.array(d["std"]); return s


def stratified_split(y, fractions=(0.70, 0.15, 0.15), seed=42):
    """Shuffle indices per class so every split keeps the same approval ratio."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        n1 = int(len(idx) * fractions[0]); n2 = n1 + int(len(idx) * fractions[1])
        parts[0] += idx[:n1].tolist(); parts[1] += idx[n1:n2].tolist(); parts[2] += idx[n2:].tolist()
    return [np.array(rng.permutation(p)) for p in parts]
