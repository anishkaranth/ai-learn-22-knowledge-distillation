"""Matplotlib SVG plots + RESULTS.md writer for the distillation smoke run."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams.update({"svg.hashsalt": "ai-learn-22", "svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False})


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    names, a = [], m["config"]["alpha"]
    Ts = m["config"]["temperatures"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for alpha, c in ((a, "#126782"), (1.0, "#e76f51")):
        rows = [r for r in m["kd_sweep"] if r["alpha"] == alpha]
        ax[0].plot(Ts, [r["acc"] for r in rows], color=c, label=f"KD alpha={alpha}")
        ax[1].plot(Ts, [r["ece"] for r in rows], color=c, label=f"KD alpha={alpha}")
    for i, k in enumerate(("acc", "ece")):
        ax[i].axhline(m["student_hard_transfer"][k], color="#555", ls="--", label="hard labels")
        ax[i].axhline(m["teacher"][k], color="#2a9d8f", ls=":", label="teacher")
        ax[i].set_xscale("log", base=2)
        ax[i].set_xlabel("temperature T")
        ax[i].set_ylabel(["test accuracy", "ECE (lower is better)"][i])
    ax[0].legend(fontsize=8)
    ax[0].set_title("Student accuracy vs T")
    ax[1].set_title("Student calibration vs T")
    names.append(_save(fig, out / "accuracy_ece_vs_temperature.svg"))

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot([0, 1], [0, 1], color="#999", ls="--", label="perfect")
    for (k, v), c in zip(m["reliability"].items(), ("#2a9d8f", "#555", "#126782")):
        pts = [(x, y) for x, y in zip(v["centres"], v["acc"]) if y is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, label=k)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram (test)")
    ax.legend(fontsize=8)
    names.append(_save(fig, out / "reliability.svg"))

    fig, ax = plt.subplots(figsize=(6, 3.2))
    soft = m["teacher_soft_targets_example0"]
    w = 0.8 / len(soft)
    for i, (T, probs) in enumerate(soft.items()):
        ax.bar([c + i * w for c in range(len(probs))], probs, width=w, label=f"T={T}")
    ax.set_xlabel("class")
    ax.set_ylabel("teacher prob")
    ax.set_title("Teacher soft targets for one example")
    ax.legend(fontsize=8)
    names.append(_save(fig, out / "soft_targets.svg"))
    return names


def write_results_md(path: Path, m: Dict[str, Any], plots: List[str]) -> None:
    h = m["headline"]
    t1 = next(r for r in m["kd_sweep"] if r["T"] == 1 and r["alpha"] == m["config"]["alpha"])
    rows = "\n".join(f"| {r['T']} | {r['alpha']} | {r['acc']} | {r['ece']} |" for r in m["kd_sweep"])
    txt = f"""# Results: ai-learn-22-knowledge-distillation

Real output of `python run_smoke.py` (seed {m['seed']}, CPU, {m['wall_time_s']} s wall time).

## Setup
- Teacher MLP {m['config']['teacher_sizes']} ({h['teacher_params']:,} params) trained on {m['config']['n_teacher_train']} examples.
- Student MLP {m['config']['student_sizes']} ({h['student_params']:,} params, {h['compression_x']}x smaller) trained on a
  **transfer set of only {m['config']['n_transfer']} examples** for {m['config']['student_epochs']} epochs; same init and shuffle for every run.
- Test set: {m['config']['n_test']} held-out examples. ECE uses {m['config']['ece_bins']} bins.

## Headline
| model | test acc | ECE |
|---|---|---|
| teacher | {m['teacher']['acc']} | {m['teacher']['ece']} |
| student, hard labels (600) | {m['student_hard_transfer']['acc']} | {m['student_hard_transfer']['ece']} |
| student, KD T=1 alpha={m['config']['alpha']} | {t1['acc']} | {t1['ece']} |
| student, KD best T={h['student_kd_best_T']} alpha={m['config']['alpha']} | {h['student_kd_best_acc']} | {h['student_kd_best_ece']} |
| student, hard labels on all 6000 (reference) | {m['student_hard_full_6000']['acc']} | {m['student_hard_full_6000']['ece']} |

## Temperature sweep
| T | alpha | acc | ECE |
|---|---|---|---|
{rows}

## Plots
""" + "\n".join(f"![{p}]({p})" for p in plots) + """

## Honest notes
- The dataset is synthetic Gaussian clusters, so absolute numbers only mean something relative to each other.
- A single seed was run, so small differences (under about 0.5 pt) are within noise.
"""
    path.write_text(txt, encoding="utf-8")
