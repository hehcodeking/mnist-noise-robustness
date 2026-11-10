"""Auditable multiclass metrics without external metric dependencies."""
import numpy as np


def classification_metrics(y, predictions):
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (y, predictions), 1)
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(10, dtype=float), where=denom != 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()),
            "correct": int(tp.sum()), "n": int(cm.sum())}


def paired_bootstrap_delta(y, a, b, seed=1901, repetitions=2000):
    """Paired image bootstrap: B minus A, conditional on the fitted models."""
    differences = (b == y).astype(np.float64) - (a == y).astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for start in range(0, repetitions, 100):
        size = min(100, repetitions - start)
        means[start:start + size] = differences[rng.integers(0, len(y), (size, len(y)))].mean(axis=1)
    return {"delta": float(differences.mean()),
            "ci_low": float(np.quantile(means, .025)),
            "ci_high": float(np.quantile(means, .975)),
            "repetitions": repetitions, "seed": seed,
            "a_wrong_b_right": int(((a != y) & (b == y)).sum()),
            "a_right_b_wrong": int(((a == y) & (b != y)).sum())}
