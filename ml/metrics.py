"""Evaluation metrics written from scratch (no sklearn)."""
import numpy as np


def confusion_matrix(y, yhat):
    tn = int(np.sum((y == 0) & (yhat == 0))); fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0))); tp = int(np.sum((y == 1) & (yhat == 1)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def roc_curve(y, p):
    """Sweep thresholds from high to low and record (FPR, TPR)."""
    order = np.argsort(-p)
    y_s = y[order]
    tps = np.cumsum(y_s); fps = np.cumsum(1 - y_s)
    tpr = np.concatenate([[0], tps / tps[-1]]); fpr = np.concatenate([[0], fps / fps[-1]])
    return fpr, tpr


def auc(fpr, tpr):
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))  # trapezoid rule


def classification_report(y, p, threshold=0.5):
    yhat = (p >= threshold).astype(int)
    cm = confusion_matrix(y, yhat)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr, tpr = roc_curve(y, p)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc(fpr, tpr), "confusion": cm}
