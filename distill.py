"""Knowledge distillation (Hinton et al., 2015) from scratch.

Student loss = alpha * T^2 * KL(p_teacher^T || p_student^T) + (1 - alpha) * CE(y, p_student)
where p^T = softmax(logits / T). The gradient w.r.t. the student logits is

    alpha * T * (p_student^T - p_teacher^T) + (1 - alpha) * (p_student - onehot(y))

(the T^2 factor keeps the soft-target gradient on the same scale as T changes).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from mlp import Adam, backward, forward, softmax


def train_distill(p: Dict[str, np.ndarray], X, y, teacher_logits, T: float, alpha: float, epochs: int, lr: float,
                  batch: int, rng) -> List[float]:
    opt, n, hist = Adam(p, lr), len(X), []
    pt_all = softmax(teacher_logits, T)
    for _ in range(epochs):
        idx, tot = rng.permutation(n), 0.0
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            logits, cache = forward(p, X[b])
            ps_T, ps = softmax(logits, T), softmax(logits)
            pt = pt_all[b]
            kl = (pt * (np.log(pt + 1e-12) - np.log(ps_T + 1e-12))).sum()
            ce = -np.log(ps[np.arange(len(b)), y[b]] + 1e-12).sum()
            tot += alpha * T * T * kl + (1 - alpha) * ce
            hard = ps.copy()
            hard[np.arange(len(b)), y[b]] -= 1
            d = alpha * T * (ps_T - pt) + (1 - alpha) * hard
            opt.step(p, backward(p, cache, d / len(b)))
        hist.append(tot / n)
    return hist


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: sum_b |B_b|/n * |acc(B_b) - conf(B_b)| over confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """(bin centres, accuracy per bin, count per bin) for a reliability diagram."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    accs, counts = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        accs.append(float((pred[m] == y[m]).mean()) if m.any() else None)
        counts.append(int(m.sum()))
    return ((edges[:-1] + edges[1:]) / 2).tolist(), accs, counts
