"""
Exhaustive dB-dB scan over all ordered pairs of connected six-node graphs.

Purpose
-------
Produces r3_n6_exhaustive_results.csv, the input of the six-node columns of
Table A in S1 Text (Text E.1; the table itself is computed by
r3_naoki_slope_bilayer_table.py). The network pairs of Fig C (Text F) were
also selected from this file (see r3_strategy_coupling_ic.py).

For every ordered pair (G_1, G_2) of the 112 connected six-node graphs
(112 x 112 = 12,544 pairs), with G_1 as layer 1 (donation game) and G_2 as
layer 2 (constant selection) and the dB-dB rule, the script calls
larger_networks_solver.bc_star_two_layer and records the weak-selection
quantities together with structural properties of both layers.

Quantities per pair
-------------------
For v = 0, ..., 5, the single initial cooperator (layer 1) and the single
initial mutant (layer 2) are both placed on node v. theta_1 - theta_3,
theta_2 and phi_{2,0} are averaged over the six values of v, and the
columns are

    theta_diff   mean over v of theta_1 - theta_3
    theta2       mean over v of theta_2
    phi20        mean over v of phi_{2,0}
    bc_single    -theta2 / theta_diff
    bc_two       bc_single + (R - 1) phi20 / (C theta_diff), R = 5, C = 1
    enhancement  bc_single - bc_two

bc_single, bc_two and enhancement are NaN when |theta_diff| <= 1e-12.
theta_diff, theta2 and phi20 do not depend on R.

Nine properties of each layer are stored with the prefixes g1_ and g2_:
n_edges, mean_degree, max_degree, var_degree (population variance of the
degrees), mean_closeness, mean_betweenness (networkx normalized betweenness),
clustering (nx.average_clustering), diameter and alg_conn
(nx.algebraic_connectivity).

g1_idx and g2_idx are positions 0-111 in the list returned by
all_connected_n6_graphs(). They are not graph-atlas indices (the atlas
indices of these graphs lie between 77 and 208). same_graph is 1 if
g1_idx == g2_idx and 0 otherwise.

After writing the CSV, the script prints the Pearson correlation of
enhancement and of phi20 with each of the eight properties other than
n_edges, for G_1 and for G_2, and draws two figures. The printed correlations
and both PNGs are exploratory and are not part of S1 Text.

Command
-------
From the revision folder:

    python r3_topology_correlation_n6_exhaustive.py

Inputs:  none (the graphs come from networkx.graph_atlas_g()).
Outputs (written to the working directory):
    r3_n6_exhaustive_results.csv           12,544 rows x 27 columns
    r3_n6_exhaustive_correlations.png      exploratory scatter plots
    r3_n6_exhaustive_full_corr_matrix.png  exploratory heatmap
Runtime: about 2.5 minutes on an Apple M1 laptop.
Dependencies: numpy, pandas, networkx, scipy, matplotlib;
larger_networks_solver.py (bc_star_two_layer).
"""
from __future__ import annotations
import warnings, scipy.sparse as sp
# Hide scipy SparseEfficiencyWarning messages (no effect on results).
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from larger_networks_solver import bc_star_two_layer


OUT_CSV = "r3_n6_exhaustive_results.csv"
OUT_FIG_SCATTER = "r3_n6_exhaustive_correlations.png"
OUT_FIG_HEATMAP = "r3_n6_exhaustive_full_corr_matrix.png"
RULE = "dB-dB"
R = 5.0     # mutant fitness r; enters only bc_two and enhancement
C = 1.0     # cost c


def all_connected_n6_graphs():
    """The 112 connected six-node graphs of networkx.graph_atlas_g(), in atlas order.

    Returns a list of networkx graphs, each with nodes 0..5. The position of
    a graph in this list is the graph index used in the output (g1_idx,
    g2_idx).
    """
    atlas = nx.graph_atlas_g()
    return [G for G in atlas
            if G.number_of_nodes() == 6 and nx.is_connected(G)]


def graph_properties(G):
    """Nine structural properties of one graph.

    Returns a dict with
        n_edges           number of edges
        mean_degree       mean degree
        max_degree        maximum degree
        var_degree        population variance of the degrees
        mean_closeness    mean of nx.closeness_centrality
        mean_betweenness  mean of nx.betweenness_centrality (normalized)
        clustering        nx.average_clustering
        diameter          nx.diameter
        alg_conn          nx.algebraic_connectivity (second-smallest
                          eigenvalue of the graph Laplacian); NaN if networkx
                          raises an exception
    """
    n = G.number_of_nodes()
    deg = np.array([d for _, d in G.degree()])
    cl = np.array(list(nx.closeness_centrality(G).values()))
    bt = np.array(list(nx.betweenness_centrality(G).values()))
    try:
        ac = float(nx.algebraic_connectivity(G))
    except Exception:
        ac = np.nan
    return {
        "n_edges":          G.number_of_edges(),
        "mean_degree":      float(deg.mean()),
        "max_degree":       int(deg.max()),
        "var_degree":       float(deg.var()),
        "mean_closeness":   float(cl.mean()),
        "mean_betweenness": float(bt.mean()),
        "clustering":       float(nx.average_clustering(G)),
        "diameter":         int(nx.diameter(G)),
        "alg_conn":         ac,
    }


def main():
    """Run the scan, write OUT_CSV, print the correlations and save the two PNGs.

    Returns None. See the module docstring for the columns of OUT_CSV.
    """
    graphs = all_connected_n6_graphs()
    print(f"# Found {len(graphs)} connected 6-node graphs", flush=True)
    assert len(graphs) == 112

    # Properties of each graph, computed once and reused for every pair.
    g_props = [graph_properties(G) for G in graphs]
    print("# Pre-computed network properties for all 112 graphs", flush=True)

    rows = []
    t0 = time.time()
    n_pairs = 0
    n_total = len(graphs) ** 2
    for i, G1 in enumerate(graphs):
        for j, G2 in enumerate(graphs):
            n_pairs += 1
            # Average theta and phi over v=0..5, init_C = init_M = v
            theta_diffs = []
            theta2s = []
            phi20s = []
            for v in range(6):
                s = bc_star_two_layer(G1, G2, v, v, RULE, r=R, c=C)
                theta_diffs.append(s["theta1"] - s["theta3"])
                theta2s.append(s["theta2"])
                phi20s.append(s["phi20"])
            td = float(np.mean(theta_diffs))
            t2 = float(np.mean(theta2s))
            ph = float(np.mean(phi20s))
            # Thresholds from the averaged quantities.
            bc_s = -t2 / td if abs(td) > 1e-12 else np.nan
            bc_t = bc_s + (R - 1.0) * ph / (C * td) if abs(td) > 1e-12 else np.nan
            row = {
                "g1_idx": i, "g2_idx": j, "same_graph": int(i == j),
                "theta_diff": td, "theta2": t2, "phi20": ph,
                "bc_single": bc_s, "bc_two": bc_t,
                "enhancement": (bc_s - bc_t) if np.isfinite(bc_s) else np.nan,
            }
            for k, val in g_props[i].items():
                row[f"g1_{k}"] = val
            for k, val in g_props[j].items():
                row[f"g2_{k}"] = val
            rows.append(row)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = n_pairs / elapsed
            eta = (n_total - n_pairs) / rate
            print(f"  {i+1}/112 G1's done  ({n_pairs}/{n_total} pairs, "
                  f"{elapsed:.1f}s elapsed, ETA {eta:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n# Saved {OUT_CSV}  ({len(df)} rows, "
          f"{time.time()-t0:.1f}s total)", flush=True)

    # ----- Correlation analysis (printed and plotted only; exploratory) -----
    prop_keys = ["mean_degree", "max_degree", "var_degree",
                 "mean_closeness", "mean_betweenness",
                 "clustering", "diameter", "alg_conn"]
    # Pearson R (and p-value) of enhancement with each property; column 0 is
    # the property of G_1, column 1 the property of G_2. Pairs with NaN
    # enhancement are left out.
    corr_matrix = np.full((len(prop_keys), 2), np.nan)
    p_matrix = np.full((len(prop_keys), 2), np.nan)
    finite = np.isfinite(df["enhancement"].values)
    y = df["enhancement"].values[finite]
    print(f"\n# Pearson correlations of Delta = (b/c)*_single - (b/c)*_two:")
    print(f"  ({finite.sum()} of {len(df)} pairs have finite Delta)")
    print(f"  {'property':22s}  {'r vs G1 prop':>14s}  {'r vs G2 prop':>14s}")
    for k, key in enumerate(prop_keys):
        x1 = df[f"g1_{key}"].values[finite]
        x2 = df[f"g2_{key}"].values[finite]
        r1, p1 = pearsonr(x1, y)
        r2, p2 = pearsonr(x2, y)
        corr_matrix[k, 0] = r1
        corr_matrix[k, 1] = r2
        p_matrix[k, 0] = p1
        p_matrix[k, 1] = p2
        print(f"  {key:22s}  {r1:+.4f} (p={p1:.1e})  "
              f"{r2:+.4f} (p={p2:.1e})")

    # Pearson correlation of the averaged phi_{2,0} with each property of G_1
    # and of G_2, over all 12,544 pairs (printed only).
    print(f"\n# Pearson correlations of phi_{{2,0}}:")
    for k, key in enumerate(prop_keys):
        x1 = df[f"g1_{key}"].values
        x2 = df[f"g2_{key}"].values
        ph = df["phi20"].values
        r1, p1 = pearsonr(x1, ph)
        r2, p2 = pearsonr(x2, ph)
        print(f"  {key:22s}  G1: r={r1:+.4f} (p={p1:.1e})  "
              f"G2: r={r2:+.4f} (p={p2:.1e})")

    # ----- Plot 1 (exploratory): scatter of enhancement vs three
    # properties, for both layers, colored by same_graph.
    centrality_cols = [
        ("mean_degree",      r"$\langle k \rangle$"),
        ("mean_closeness",   r"mean closeness"),
        ("mean_betweenness", r"mean betweenness"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    for layer_idx, layer_label in enumerate(["G_1", "G_2"]):
        for j, (col, lbl) in enumerate(centrality_cols):
            ax = axes[layer_idx, j]
            x = df[f"g{layer_idx+1}_{col}"].values[finite]
            yj = df["enhancement"].values[finite]
            sg = df["same_graph"].values[finite]
            r, p = pearsonr(x, yj)
            ax.scatter(x[sg == 0], yj[sg == 0], s=8, alpha=0.3,
                        color="#1f77b4", label="$G_1 \\neq G_2$",
                        edgecolor="none")
            ax.scatter(x[sg == 1], yj[sg == 1], s=24, alpha=0.85,
                        color="#d62728", label="$G_1 = G_2$",
                        edgecolor="black", linewidth=0.4)
            ax.set_xlabel(f"{lbl} of ${layer_label}$", fontsize=11)
            if j == 0:
                ax.set_ylabel(r"$\Delta = (b/c)^{*}_{\rm single} - "
                               r"(b/c)^{*}_{\rm two}$", fontsize=11)
            ax.set_title(f"$R$ = {r:+.3f},  $p$ = {p:.2e}", fontsize=10)
            ax.grid(alpha=0.25)
            if layer_idx == 0 and j == 2:
                ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"Reviewer 3 — exhaustive bilayer scan at $N=6$ "
        f"({len(df)} ordered $(G_1, G_2)$ pairs, $r={R:.0f}$, dB-dB)",
        fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT_FIG_SCATTER, dpi=180, bbox_inches="tight")
    print(f"\n# Figure saved: {OUT_FIG_SCATTER}")

    # ----- Plot 2 (exploratory): heatmap of the Pearson R of Delta
    # (enhancement) with every property of both layers.
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    # corr_matrix (computed above): one row per property, one column per layer.
    im = ax2.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([r"$G_1$ property", r"$G_2$ property"])
    ax2.set_yticks(range(len(prop_keys)))
    ax2.set_yticklabels(prop_keys)
    for k in range(len(prop_keys)):
        for j in range(2):
            v = corr_matrix[k, j]
            ax2.text(j, k, f"{v:+.2f}",
                      ha="center", va="center",
                      color="white" if abs(v) > 0.5 else "black",
                      fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax2, label="Pearson $R$ with $\\Delta$")
    ax2.set_title(r"$\Delta = (b/c)^{*}_{\rm single} - (b/c)^{*}_{\rm two}$"
                   "  vs  network property", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(OUT_FIG_HEATMAP, dpi=180, bbox_inches="tight")
    print(f"# Heatmap saved: {OUT_FIG_HEATMAP}")


if __name__ == "__main__":
    main()
