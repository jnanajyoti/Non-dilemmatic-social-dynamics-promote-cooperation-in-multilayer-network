"""
Fig A of S1 Text (Text E.1): Pearson correlation matrix of five per-graph
quantities of the 112 connected six-node graphs (single layer, dB rule).

Reads r3_naoki_per_graph.csv (written by r3_naoki_revisions.py) and computes
the Pearson correlation coefficient R and its p-value (scipy.stats.pearsonr)
for every pair of the columns

    theta_diff_mean  mean over the six initial nodes of theta_1 - theta_3
    k_coop           n_init,C: number of initial nodes with (b/c)*_single > 0
    mean_degree      mean degree <k>
    mean_closeness   mean closeness centrality
    clustering       mean local clustering coefficient

with the 112 graphs as samples. The heatmap shows R in each cell with two
decimals; the diagonal cells show 1.00, and minus signs are drawn as the
typographic minus U+2212.

The "6x6" in the name of this script refers to an earlier version of the
figure with six quantities; the script draws the 5 x 5 matrix.

Command
-------
From the revision folder, after python r3_naoki_revisions.py:

    python r3_naoki_6x6_heatmap.py

Inputs:  r3_naoki_per_graph.csv (path relative to the working directory).
Outputs (written to the working directory):
    fig1_R3_pairwise_5x5.png   Fig A
    r3_pairwise_5x5_R.csv      the matrix of R drawn in Fig A
    r3_pairwise_5x5_p.csv      the matrix of p-values
Runtime: about 2 s on an Apple M1 laptop.
Dependencies: numpy, pandas, scipy, matplotlib. No local module is imported.
"""
from __future__ import annotations
import warnings, scipy.sparse as sp
# Warning filter shared with the other scripts; no sparse code runs here.
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# Paths relative to the working directory (run from the revision folder).
CSV_IN  = "r3_naoki_per_graph.csv"      # written by r3_naoki_revisions.py
OUT_FIG = "fig1_R3_pairwise_5x5.png"    # Fig A

# Rows and columns of Fig A, in order; k_coop (n_init,C) is the second row
# and column. QTY_LABELS are the axis labels.
QTY_COLS = [
    "theta_diff_mean",
    "k_coop",
    "mean_degree",
    "mean_closeness",
    "clustering",
]
QTY_LABELS = [
    r"$\theta_1 - \theta_3$",
    r"$n_{\rm init,C}$",
    r"$\langle k \rangle$",
    "closeness",
    "clustering",
]


def main():
    """Compute the 5 x 5 matrices of R and p, draw Fig A and write the two CSVs.

    Returns None. Reads CSV_IN; writes OUT_FIG, r3_pairwise_5x5_R.csv and
    r3_pairwise_5x5_p.csv to the working directory.
    """
    df = pd.read_csv(CSV_IN)
    print(f"# Loaded {len(df)} per-graph rows from {CSV_IN}")

    n = len(QTY_COLS)
    R = np.full((n, n), np.nan)
    P = np.full((n, n), np.nan)
    for i, ci in enumerate(QTY_COLS):
        for j, cj in enumerate(QTY_COLS):
            r, p = pearsonr(df[ci].values, df[cj].values)
            R[i, j] = r
            P[i, j] = p

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 7.5))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(QTY_LABELS, rotation=30, ha="right", fontsize=12)
    ax.set_yticklabels(QTY_LABELS, fontsize=12)
    for i in range(n):
        for j in range(n):
            v = R[i, j]
            color = "white" if abs(v) > 0.5 else "black"   # contrast with cell color
            # No leading "+"; render the minus as a typographic minus (U+2212),
            # which is longer than the ASCII hyphen (matches the axis-tick style).
            # Diagonal cells are written as 1.00.
            txt = ("1.00" if i == j else f"{v:.2f}").replace("-", "−")
            ax.text(j, i, txt, ha="center", va="center",
                     color=color, fontsize=14, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation coefficient", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    print(f"# Saved {OUT_FIG}")

    # Also save the matrices of R and p-values as CSV (rows and columns in
    # QTY_COLS order).
    Rdf = pd.DataFrame(R, index=QTY_COLS, columns=QTY_COLS)
    Pdf = pd.DataFrame(P, index=QTY_COLS, columns=QTY_COLS)
    Rdf.to_csv("r3_pairwise_5x5_R.csv")
    Pdf.to_csv("r3_pairwise_5x5_p.csv")
    print("# Saved r3_pairwise_5x5_R.csv and r3_pairwise_5x5_p.csv")


if __name__ == "__main__":
    main()
