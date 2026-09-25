#!/usr/bin/env python3
"""Distillation smoke: train big teacher -> small student on a small transfer set (hard labels vs soft targets, T sweep) -> results/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from distill import ece, reliability, train_distill
from mlp import accuracy, forward, init_mlp, make_data, softmax, train
from smoke_plots import make_plots, write_results_md

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
CFG = {"teacher_sizes": [20, 256, 256, 8], "student_sizes": [20, 16, 8], "n_teacher_train": 6000, "n_transfer": 600,
       "n_test": 3000, "teacher_epochs": 20, "student_epochs": 200, "lr_teacher": 1e-3, "lr_student": 3e-3,
       "batch_teacher": 64, "batch_student": 32, "alpha": 0.9, "temperatures": [1, 2, 4, 8, 16], "ece_bins": 15}


def _compact(js: str) -> str:
    js = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", js)
    return re.sub(r"\{\n([^{}\[\]]*?)\n\s*\}", lambda m: "{" + re.sub(r"\s*\n\s*", " ", m.group(1)).strip() + "}", js)


def n_params(p) -> int:
    return int(sum(v.size for v in p.values()))


def evaluate(p, Xt, yt):
    probs = softmax(forward(p, Xt)[0])
    return {"acc": round(accuracy(p, Xt, yt), 4), "ece": round(ece(probs, yt, CFG["ece_bins"]), 4)}, probs


def student(fit):
    """Fresh student with a fixed init/shuffle seed so every run differs only in its targets."""
    s = init_mlp(CFG["student_sizes"], np.random.default_rng(SEED + 1))
    fit(s, np.random.default_rng(SEED + 1))
    return s


def main() -> None:
    t0 = time.perf_counter()
    X, y = make_data(CFG["n_teacher_train"], 1)
    Xt, yt = make_data(CFG["n_test"], 2)
    Xs, ys = X[:CFG["n_transfer"]], y[:CFG["n_transfer"]]
    ep, lr, bs = CFG["student_epochs"], CFG["lr_student"], CFG["batch_student"]

    rng = np.random.default_rng(SEED)
    teacher = init_mlp(CFG["teacher_sizes"], rng)
    train(teacher, X, y, CFG["teacher_epochs"], CFG["lr_teacher"], CFG["batch_teacher"], rng)
    t_eval, t_probs = evaluate(teacher, Xt, yt)
    t_logits = forward(teacher, Xs)[0]

    hard = student(lambda s, r: train(s, Xs, ys, ep, lr, bs, r))
    h_eval, h_probs = evaluate(hard, Xt, yt)
    full = student(lambda s, r: train(s, X, y, max(1, ep * CFG["n_transfer"] // CFG["n_teacher_train"]), lr, bs, r))
    f_eval, _ = evaluate(full, Xt, yt)

    sweep, kd_probs = [], {}
    for T in CFG["temperatures"]:
        for alpha in (CFG["alpha"], 1.0):
            s = student(lambda s, r: train_distill(s, Xs, ys, t_logits, T, alpha, ep, lr, bs, r))
            ev, pr = evaluate(s, Xt, yt)
            sweep.append({"T": T, "alpha": alpha, **ev})
            kd_probs[(T, alpha)] = pr
    best = max((r for r in sweep if r["alpha"] == CFG["alpha"]), key=lambda r: (r["acc"], -r["ece"]))
    b_probs = kd_probs[(best["T"], best["alpha"])]

    # Teacher soft targets for one transfer example at each temperature (what "dark knowledge" looks like)
    soft = {str(T): [round(float(v), 4) for v in softmax(t_logits[:1], T)[0]] for T in CFG["temperatures"]}
    rel = {name: reliability(pr, yt) for name, pr in
           (("teacher", t_probs), ("student_hard", h_probs), (f"student_kd_T{best['T']}", b_probs))}
    rel = {k: {"centres": [round(c, 3) for c in v[0]], "acc": [None if a is None else round(a, 4) for a in v[1]],
               "count": v[2]} for k, v in rel.items()}

    m = {"project": "ai-learn-22-knowledge-distillation", "seed": SEED, "config": CFG,
         "params": {"teacher": n_params(teacher), "student": n_params(hard)},
         "teacher": t_eval, "student_hard_transfer": h_eval, "student_hard_full_6000": f_eval,
         "kd_sweep": sweep, "best_kd": best, "teacher_soft_targets_example0": soft, "reliability": rel,
         "headline": {
             "teacher_acc": t_eval["acc"], "teacher_params": n_params(teacher), "student_params": n_params(hard),
             "compression_x": round(n_params(teacher) / n_params(hard), 1),
             "student_hard_acc": h_eval["acc"], "student_hard_ece": h_eval["ece"],
             "student_kd_best_T": best["T"], "student_kd_best_acc": best["acc"], "student_kd_best_ece": best["ece"],
             "student_kd_T1_acc": next(r["acc"] for r in sweep if r["T"] == 1 and r["alpha"] == CFG["alpha"]),
             "student_hard_full_data_acc": f_eval["acc"],
         }}
    m["wall_time_s"] = round(time.perf_counter() - t0, 2)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(_compact(json.dumps(m, indent=1)) + "\n")
    shot = {"project": m["project"], "seed": SEED, "config": CFG, "params": m["params"], "headline": m["headline"]}
    (RESULTS / "JSON.shot").write_text(_compact(json.dumps(shot, indent=2)) + "\n")
    plots = make_plots(RESULTS, m)
    write_results_md(RESULTS / "RESULTS.md", m, plots)
    json.loads((RESULTS / "JSON.shot").read_text())
    print(json.dumps(m["headline"], indent=2), "\nwall", m["wall_time_s"])


if __name__ == "__main__":
    main()
