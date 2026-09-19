"""
Six-node columns of Table A in S1 Text (Text E.1): Pearson correlation of
|d(b/c)*/dr| with structural properties of layer 1 and of layer 2.

Reads the exhaustive dB-dB scan r3_n6_exhaustive_results.csv (one row per
ordered pair (G_1, G_2) of connected six-node graphs, G_1 = layer 1 with the
donation game, G_2 = layer 2 with constant selection; written by
r3_topology_correlation_n6_exhaustive.py) and r3_naoki_per_graph.csv
(written by r3_naoki_revisions.py).

It keeps the pairs whose G_1 has k_coop = 6 in r3_naoki_per_graph.csv, i.e.
(b/c)*_single > 0 for all six choices of the initial cooperator's node. There
are 32 such graphs, so 32 x 112 = 3,584 pairs are kept; G_2 is not
restricted. For each kept pair the script forms

    |d(b/c)*/dr| = |phi20 / (C theta_diff)|,  C = 1,

from the columns phi20 and theta_diff of the scan (each averaged over the six
co-located initial conditions; NaN if |theta_diff| <= 1e-12), and computes
its Pearson correlation over the kept pairs with each of eight properties of
G_1 ("layer 1") and of G_2 ("layer 2"): mean_degree, max_degree, var_degree,
mean_closeness, mean_betweenness, clustering, diameter and alg_conn (defined
in r3_topology_correlation_n6_exhaustive.py). Table A uses the rows
mean_degree, max_degree, var_degree, mean_closeness, clustering, diameter and
alg_conn; the output also has a mean_betweenness row.

Command
-------
From the revision folder, after python r3_topology_correlation_n6_exhaustive.py
and python r3_naoki_revisions.py:

    python r3_naoki_slope_bilayer_table.py

Inputs (paths relative to the working directory):
    r3_n6_exhaustive_results.csv, r3_naoki_per_graph.csv
Outputs (written to the working directory):
    r3_part_4_bilayer_slope_table.csv   columns property, layer 1, layer 2
                                        (Table A, six-node columns)
    r3_part_4_bilayer_slope_table.txt   the same values as a text table
    fig4_R3_bilayer_slope_table.png     the same values as a table image;
                                        not part of S1 Text
Runtime: about 4 s on an Apple M1 laptop (the scan that produces the input
takes about 2.5 minutes).
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
EXHAUSTIVE_CSV = "r3_n6_exhaustive_results.csv"   # from r3_topology_correlation_n6_exhaustive.py
PER_GRAPH_CSV  = "r3_naoki_per_graph.csv"         # from r3_naoki_revisions.py

OUT_CSV  = "r3_part_4_bilayer_slope_table.csv"
OUT_TXT  = "r3_part_4_bilayer_slope_table.txt"
OUT_PNG  = "fig4_R3_bilayer_slope_table.png"

C = 1.0       # cost c in |phi20 / (c theta_diff)|
TOL = 1e-12   # slope is NaN when |theta_diff| <= TOL

# Properties correlated with |d(b/c)*/dr|, each for G_1 (g1_*) and G_2 (g2_*).
PROP_KEYS = [
    "mean_degree",
    "max_degree",
    "var_degree",
    "mean_closeness",
    "mean_betweenness",
    "clustering",
    "diameter",
    "alg_conn",
]


def main():
    """Filter the scan, compute the correlations and write the CSV, TXT and PNG.

    Returns None. See the module docstring for inputs and outputs.
    """
    # ----- Load -----
    # graph_idx in PER_GRAPH_CSV and g1_idx in EXHAUSTIVE_CSV are both
    # positions in the same list of 112 connected six-node atlas graphs.
    pg = pd.read_csv(PER_GRAPH_CSV)
    coop_ids = sorted(pg.loc[pg["k_coop"] == 6, "graph_idx"].astype(int).tolist())
    print(f"# k_coop==6 graphs (G_1 candidates): {len(coop_ids)}")

    df_all = pd.read_csv(EXHAUSTIVE_CSV)
    print(f"# Loaded {len(df_all)} (G_1, G_2) pairs from {EXHAUSTIVE_CSV}")

    # ----- Keep the pairs whose G_1 has k_coop = 6 (all 112 G_2 kept) -----
    df = df_all[df_all["g1_idx"].isin(coop_ids)].copy()
    print(f"# Filtered to {len(df)} rows (expected {len(coop_ids) * 112})")
    assert len(df) == len(coop_ids) * 112, "filter row count mismatch"

    # ----- |slope| = |d(b/c)*/dr| = |phi_{2,0} / (c * (theta_1 - theta_3))| -----
    # computed from the columns phi20 and theta_diff of the scan (averages
    # over the six co-located initial conditions)
    td = df["theta_diff"].values
    ph = df["phi20"].values
    slope = np.where(np.abs(td) > TOL, ph / (C * td), np.nan)
    df["slope"] = slope
    df["abs_slope"] = np.abs(slope)
    finite = np.isfinite(df["abs_slope"].values)
    print(f"# {finite.sum()} of {len(df)} filtered pairs have finite |slope|")
    y = df["abs_slope"].values[finite]

    # ----- Correlations: Pearson R over the kept pairs with finite |slope| -----
    rows = []
    print()
    print(f"# Pearson R of |slope| against each property "
          f"(filtered: G_1 in {len(coop_ids)} all-init cooperative graphs)")
    print(f"  {'property':22s}  {'R (layer 1)':>12s}  {'R (layer 2)':>12s}")
    print("  " + "-" * 52)
    for key in PROP_KEYS:
        x1 = df[f"g1_{key}"].values[finite]
        x2 = df[f"g2_{key}"].values[finite]
        r1, _ = pearsonr(x1, y)
        r2, _ = pearsonr(x2, y)
        rows.append({"property": key, "layer 1": r1, "layer 2": r2})
        print(f"  {key:22s}  {r1:+12.4f}  {r2:+12.4f}")

    out_df = pd.DataFrame(rows, columns=["property", "layer 1", "layer 2"])

    # ----- CSV -----
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n# Saved {OUT_CSV}")

    # ----- TXT -----
    name_w = max(len("property"), max(len(r["property"]) for r in rows))
    col_w = max(len("layer 1"), len("layer 2"), 8)
    lines = []
    header = f"{'property':<{name_w}}  {'layer 1':>{col_w}}  {'layer 2':>{col_w}}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"{r['property']:<{name_w}}  "
            f"{r['layer 1']:>+{col_w}.4f}  "
            f"{r['layer 2']:>+{col_w}.4f}"
        )
    txt = "\n".join(lines) + "\n"
    with open(OUT_TXT, "w") as f:
        f.write(txt)
    print(f"# Saved {OUT_TXT}")
    print()
    print(txt)

    # ----- PNG table -----
    cell_text = [
        [r["property"], f"{r['layer 1']:+.3f}", f"{r['layer 2']:+.3f}"]
        for r in rows
    ]
    col_labels = ["property", "layer 1", "layer 2"]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.4)
    # Light styling: bold header, plain interior, no color shading
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_text_props(weight="bold")
        cell.set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"# Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
