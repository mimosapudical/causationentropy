"""Generate manuscript figures from the frozen Path-B paper summary.

Usage:
    python -m experiments.make_path_b_paper_figures

Outputs SVG files to experiments/results/path_b_paper_figures/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/results/path_b_paper_summary_20260929.json"
OUT = ROOT / "experiments/results/path_b_paper_figures"


def load():
    with DATA.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, format="svg", bbox_inches="tight")
    plt.close(fig)


def scaling(data):
    rows = data["scaling"]
    n = np.asarray([r["n_nodes"] for r in rows])
    fig, ax = plt.subplots()
    ax.plot(n, [r["cmi_reduction"] * 100 for r in rows], marker="o", label="CMI reduction (%)")
    ax.plot(n, [r["speedup"] for r in rows], marker="o", label="end-to-end speedup (x)")
    ax.set_xlabel("network size N")
    ax.set_title("Screened-oCSE scaling")
    ax.legend()
    save(fig, "scaling.svg")


def nonlinear(data):
    rows = data["nonlinear_cancellation"]["cells"]
    x = np.asarray([r["realized_retention"] * 100 for r in rows])
    fig, ax = plt.subplots()
    ax.plot(x, [r["path_b_hidden_recall"] for r in rows], marker="o", label="Path-B conditional rescue")
    ax.plot(x, [r["forward_hidden_recall"] for r in rows], marker="o", label="forward regression")
    ax.plot(x, [r["marginal_kde_hidden_recall"] for r in rows], marker="o", label="marginal KDE top-k")
    ax.plot(x, [r["random_hidden_recall"] for r in rows], marker="o", label="random")
    ax.set_xlabel("realized candidate retention (%)")
    ax.set_ylabel("hidden-parent recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("Nonlinear conditional cancellation")
    ax.legend()
    save(fig, "nonlinear_cancellation.svg")


def suppression(data):
    rows = data["linear_suppression"]["endpoint_vs_rescue"]
    x = [r["rho"] for r in rows]
    fig, ax = plt.subplots()
    ax.plot(x, [r["endpoint_hidden_recall"] for r in rows], marker="o", label="Information-LASSO endpoint")
    ax.plot(x, [r["rescue_hidden_recall"] for r in rows], marker="o", label="+ conditional rescue")
    ax.set_xlabel("predictor correlation rho")
    ax.set_ylabel("hidden-parent recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("Linear suppression / marginal cancellation")
    ax.legend()
    save(fig, "linear_suppression.svg")


def main():
    data = load()
    scaling(data)
    nonlinear(data)
    suppression(data)


if __name__ == "__main__":
    main()
