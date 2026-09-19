"""
Single-layer analysis of the 112 connected six-node graphs under the dB rule:
per-graph data for Fig A, the partial correlations of Text E.1, and Fig B of
S1 Text (Text E.2).

Steps
-----
0. For each graph G and each node v, bc_star_two_layer(G, G, v, v, 'dB-dB',
   r=2.0) from larger_networks_solver.py gives theta_1, theta_2 and theta_3
   with the single initial cooperator on node v, and
   (b/c)*_single = -theta_2 / (theta_1 - theta_3), set to NaN when
   |theta_1 - theta_3| <= 1e-9. The layer-2 arguments and r do not affect
   these single-layer quantities.

1. r3_naoki_per_graph.csv, one row per graph, with columns
       graph_idx         position 0-111 in all_connected_n6_graphs() (not the
                         graph-atlas index)
       theta_diff_mean   mean over the six nodes v of theta_1 - theta_3
       k_coop            n_init,C in S1 Text: the number of nodes v with
                         (b/c)*_single > 0
       mean_degree, mean_closeness, mean_betweenness (networkx normalized),
       clustering        mean local clustering coefficient
   This file is read by r3_naoki_6x6_heatmap.py (Fig A) and by
   r3_naoki_slope_bilayer_table.py (Table A).

2. fig1_R3_scatter_matrix_5x5.png: scatterplot matrix of theta_diff_mean,
   mean degree, mean closeness, mean betweenness and clustering, with the
   Pearson R, its p-value and a least-squares line in each off-diagonal
   panel and a histogram on the diagonal. Exploratory; not part of S1 Text.

3. Partial Pearson correlations of theta_diff_mean and of k_coop with
   clustering, mean closeness and mean betweenness, with the mean degree
   partialled out, over the 112 graphs. Written to
   r3_partial_correlations.csv and r3_partial_correlations.txt. The
   clustering and mean-closeness rows are the values reported in Text E.1.

4. Fig B: for each of the 32 graphs with k_coop = 6, the Pearson R across
   the six nodes between (b/c)*_single (cooperator on node v) and the degree,
   closeness centrality, betweenness centrality and local clustering
   coefficient of v (nx.clustering, which is 0 for nodes of degree 1). A
   graph is skipped for a centrality whose value is the same at all six
   nodes. Outputs:
       r3_per_node_centrality_results.csv  one row per graph and centrality
                                           (graph_idx, centrality, R, p)
       r3_per_node_centrality_summary.csv  per centrality: number of graphs,
                                           mean and standard deviation of R
       fig5_R3_per_node_centrality.png     Fig B: box plot of R with the
                                           individual graphs as jittered
                                           points (jitter seeded with
                                           numpy.random.default_rng(0))

Command
-------
From the revision folder:

    python r3_naoki_revisions.py

Inputs:  none (the graphs come from networkx.graph_atlas_g()).
Outputs: the files named above, written to the working directory.
Runtime: about 5 s on an Apple M1 laptop.
Dependencies: numpy, networkx, pandas, scipy, and matplotlib < 3.11 (the box
plot uses boxplot(labels=...), an argument name that matplotlib 3.11
removes); larger_networks_solver.py (bc_star_two_layer).
"""
from __future__ import annotations
import warnings, scipy.sparse as sp
# Hide scipy SparseEfficiencyWarning messages (no effect on results).
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from larger_networks_solver import bc_star_two_layer

RULE = "dB-dB"
TOL = 1e-9      # (b/c)*_single is NaN when |theta_1 - theta_3| <= TOL


def all_connected_n6_graphs():
    """The 112 connected six-node graphs of networkx.graph_atlas_g(), in atlas order.

    Returns a list of networkx graphs, each with nodes 0..5. The position of
    a graph in this list is graph_idx in the outputs.
    """
    return [G for G in nx.graph_atlas_g()
            if G.number_of_nodes() == 6 and nx.is_connected(G)]


def per_node_quantities(G, init):
    """Single-layer quantities of G with the single initial cooperator on node init.

    Calls bc_star_two_layer(G, G, init, init, RULE, r=2.0, c=1.0); the
    layer-2 graph, the mutant's node and r do not affect theta_n.

    Returns
    -------
    (theta_1 - theta_3, theta_2, (b/c)*_single), where
    (b/c)*_single = -theta_2 / (theta_1 - theta_3), or NaN if
    |theta_1 - theta_3| <= TOL.
    """
    s = bc_star_two_layer(G, G, init, init, RULE, r=2.0, c=1.0)
    td = s["theta1"] - s["theta3"]
    bc = -s["theta2"] / td if abs(td) > TOL else np.nan
    return td, s["theta2"], bc


def network_summary(G):
    """Node centralities of G and their means over the nodes.

    Returns a dict with
        mean_degree       mean degree
        mean_closeness    mean of nx.closeness_centrality
        mean_betweenness  mean of nx.betweenness_centrality (normalized)
        clustering        mean of the local clustering coefficients
                          nx.clustering (equal to nx.average_clustering)
        deg_per_node, cl_per_node, bt_per_node, lc_per_node
                          the per-node values as arrays, in the node order
                          of G (0..5 for the atlas graphs)
    """
    deg = np.array([d for _, d in G.degree()])
    cl  = np.array(list(nx.closeness_centrality(G).values()))
    bt  = np.array(list(nx.betweenness_centrality(G).values()))
    lc  = np.array(list(nx.clustering(G).values()))
    return {
        "mean_degree":          float(deg.mean()),
        "mean_closeness":       float(cl.mean()),
        "mean_betweenness":     float(bt.mean()),
        "clustering":           float(lc.mean()),    # average local clustering
        "deg_per_node":         deg,
        "cl_per_node":          cl,
        "bt_per_node":          bt,
        "lc_per_node":          lc,
    }


def partial_corr(x, y, z):
    """First-order partial Pearson correlation of x and y, controlling for z.

    pcc = (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2) (1 - r_yz^2)), where r_ab is
    the Pearson correlation of a and b. The two-sided p-value comes from a
    t test with n - 3 degrees of freedom, t = pcc sqrt((n - 3) / (1 - pcc^2)),
    n = len(x).

    Returns
    -------
    (pcc, p); (nan, nan) if x or y is perfectly correlated with z.
    """
    rxy, _ = pearsonr(x, y)
    rxz, _ = pearsonr(x, z)
    ryz, _ = pearsonr(y, z)
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    if denom < 1e-15:
        return np.nan, np.nan
    pcc = (rxy - rxz * ryz) / denom
    # significance: t-stat with n - 3 df
    n = len(x)
    t = pcc * np.sqrt((n - 3) / max(1 - pcc**2, 1e-15))
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(abs(t), df=n - 3))
    return pcc, p


def main():
    """Run steps 0 to 4 of the module docstring; all outputs go to the working directory.

    Returns None.
    """
    graphs = all_connected_n6_graphs()
    print(f"# {len(graphs)} connected 6-node graphs")

    # ----- Per-graph quantities (r3_naoki_per_graph.csv) -----
    rows = []
    for i, G in enumerate(graphs):
        td_list, t2_list, bc_list = [], [], []
        for v in G.nodes():
            td, t2, bc = per_node_quantities(G, v)
            td_list.append(td); t2_list.append(t2); bc_list.append(bc)
        td_mean = float(np.mean(td_list))
        bc_arr = np.array(bc_list)
        finite = np.isfinite(bc_arr)
        # k_coop (n_init,C): number of nodes v with finite (b/c)*_single > 0.
        k_coop = int(((bc_arr > 0) & finite).sum())
        cents = network_summary(G)
        rows.append({
            "graph_idx": i,
            "theta_diff_mean": td_mean,
            "k_coop": k_coop,
            # the four means; the per-node arrays are not stored
            **{k: v for k, v in cents.items() if not isinstance(v, np.ndarray)},
        })
    df = pd.DataFrame(rows)
    df.to_csv("r3_naoki_per_graph.csv", index=False)
    print(f"# Saved per-graph data ({len(df)} rows)")

    # ====================================================================
    # Scatterplot matrix of five per-graph quantities (exploratory)
    # ====================================================================
    qty_cols = ["theta_diff_mean", "mean_degree", "mean_closeness",
                "mean_betweenness", "clustering"]
    qty_labels = [r"$\overline{\theta_1 - \theta_3}$",
                  r"$\langle k \rangle$",
                  "mean closeness",
                  "mean betweenness",
                  "clustering coefficient"]
    n = len(qty_cols)
    fig, axes = plt.subplots(n, n, figsize=(15, 14))
    for i, ci in enumerate(qty_cols):
        for j, cj in enumerate(qty_cols):
            ax = axes[i, j]
            xi = df[cj].values
            yi = df[ci].values
            if i == j:
                ax.hist(xi, bins=20, color="#1f77b4", alpha=0.75)
                ax.set_yticks([])
            else:
                ax.scatter(xi, yi, s=12, alpha=0.6, color="#1f77b4",
                            edgecolors="none")
                R, p = pearsonr(xi, yi)
                ax.text(0.05, 0.95, f"R = {R:+.3f}\np = {p:.1e}",
                         transform=ax.transAxes, va="top", ha="left",
                         fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="white", edgecolor="gray",
                                    alpha=0.8))
                # least-squares line (skipped if x is constant)
                if not np.allclose(xi, xi[0]):
                    m, b = np.polyfit(xi, yi, 1)
                    xs = np.linspace(xi.min(), xi.max(), 50)
                    ax.plot(xs, m * xs + b, color="#d62728", lw=1.2,
                            ls="--", alpha=0.85)
            if i == n - 1:
                ax.set_xlabel(qty_labels[j], fontsize=11)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(qty_labels[i], fontsize=11)
            else:
                if i != j:
                    ax.set_yticklabels([])
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.2)
    fig.suptitle(f"Pairwise scatterplots and Pearson R "
                 f"({len(df)} connected 6-node graphs, dB-dB, init averaged)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    # Exploratory scatterplot matrix, not part of S1 Text; Fig A is drawn by
    # r3_naoki_6x6_heatmap.py.
    fig.savefig("fig1_R3_scatter_matrix_5x5.png", dpi=180, bbox_inches="tight")
    print("# Saved fig1_R3_scatter_matrix_5x5.png")

    # ====================================================================
    # Partial correlations with the mean degree <k> partialled out (Text E.1)
    # ====================================================================
    print()
    print("# Partial correlations (with <k> partialed out)")
    print(f"  {'y':<24s}  {'X':<22s}  {'partial R':>10s}  {'p-value':>10s}")
    print("  " + "-" * 72)
    z = df["mean_degree"].values
    partial_rows = []
    for ylabel, ycol in [("mean theta_1 - theta_3", "theta_diff_mean"),
                          ("favorable init count k_coop", "k_coop")]:
        y = df[ycol].values
        for xname, xcol in [("clustering",       "clustering"),
                              ("mean closeness",  "mean_closeness"),
                              ("mean betweenness", "mean_betweenness")]:
            x = df[xcol].values
            pcc, p = partial_corr(y, x, z)
            print(f"  {ylabel:<24s}  {xname:<22s}  {pcc:+10.4f}  {p:10.2e}")
            partial_rows.append({"y": ylabel, "x": xname,
                                  "partial_R": pcc, "p_value": p})
    pdf = pd.DataFrame(partial_rows)
    pdf.to_csv("r3_partial_correlations.csv", index=False)
    with open("r3_partial_correlations.txt", "w") as f:
        f.write("Partial Pearson correlations with mean degree <k> partialed out\n")
        f.write(f"({len(df)} connected 6-node graphs, dB-dB)\n\n")
        f.write(f"{'y':<24s}  {'X':<22s}  {'partial R':>10s}  {'p-value':>10s}\n")
        f.write("-" * 72 + "\n")
        for r in partial_rows:
            f.write(f"{r['y']:<24s}  {r['x']:<22s}  "
                    f"{r['partial_R']:+10.4f}  {r['p_value']:10.2e}\n")
    print("# Saved r3_partial_correlations.{csv,txt}")

    # ====================================================================
    # Per-node centrality analysis on the k_coop = 6 graphs (Fig B)
    # ====================================================================
    coop_graphs = [(i, G) for (i, G) in enumerate(graphs)
                    if df.iloc[i]["k_coop"] == 6]
    print()
    print(f"# {len(coop_graphs)} networks with k_coop = 6 (all 6 inits "
          "give cooperation regime)")

    per_net_corrs = {"k_i": [], "closeness_i": [], "betweenness_i": [],
                     "local_C_i": []}
    per_net_rows = []
    for (gid, G) in coop_graphs:
        # node centralities
        deg_i = dict(G.degree())
        cl_i  = nx.closeness_centrality(G)
        bt_i  = nx.betweenness_centrality(G)
        lc_i  = nx.clustering(G)
        nodes = sorted(G.nodes())
        bc_per_node = []
        for v in nodes:
            _, _, bc = per_node_quantities(G, v)
            bc_per_node.append(bc)
        bc_per_node = np.array(bc_per_node)
        # guard: k_coop = 6 already implies six finite values
        if not np.all(np.isfinite(bc_per_node)):
            continue
        for label, lookup in [("k_i", deg_i), ("closeness_i", cl_i),
                                ("betweenness_i", bt_i), ("local_C_i", lc_i)]:
            x = np.array([lookup[v] for v in nodes])
            # centrality identical at all six nodes, so Pearson R is
            # undefined; skip
            if np.std(x) < 1e-12:
                continue
            R, p = pearsonr(x, bc_per_node)
            per_net_corrs[label].append(R)
            per_net_rows.append({"graph_idx": gid, "centrality": label,
                                  "R": R, "p": p})
    npn = len(coop_graphs)
    print()
    print(f"# Per-node centrality vs (b/c)*_single, "
          "Pearson R (mean ± std across networks)")
    print(f"  {'centrality':<18s}  {'# nets':>8s}  {'mean R':>10s}  {'std R':>10s}")
    print("  " + "-" * 52)
    summary_rows = []
    for label in ["k_i", "closeness_i", "betweenness_i", "local_C_i"]:
        arr = np.array(per_net_corrs[label])
        summary_rows.append({"centrality": label, "n_networks": len(arr),
                              "mean_R": arr.mean(), "std_R": arr.std()})
        print(f"  {label:<18s}  {len(arr):>8d}  {arr.mean():>+10.4f}  "
              f"{arr.std():>10.4f}")
    pd.DataFrame(per_net_rows).to_csv("r3_per_node_centrality_results.csv",
                                       index=False)
    pd.DataFrame(summary_rows).to_csv("r3_per_node_centrality_summary.csv",
                                       index=False)
    print("# Saved r3_per_node_centrality_*.csv")

    # ----- Fig B: box plot of the per-network correlations -----
    # (boxplot(labels=...) requires matplotlib < 3.11)
    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 6))
    labels = ["k_i", "closeness_i", "betweenness_i", "local_C_i"]
    label_pretty = ["degree", "closeness",
                     "betweenness",
                     "local clustering\ncoefficient"]
    box_data = [per_net_corrs[l] for l in labels]
    bp = ax2.boxplot(box_data, labels=label_pretty, patch_artist=True,
                      widths=0.6,
                      boxprops=dict(facecolor="#1f77b4", alpha=0.5),
                      medianprops=dict(color="black", lw=1.5))
    ax2.axhline(0, color="gray", lw=1, ls="--", alpha=0.6)
    ax2.set_ylim(-1.1, 1.1)
    # one point per network, jittered horizontally (seeded)
    rng = np.random.default_rng(0)
    for i, l in enumerate(labels):
        arr = np.array(per_net_corrs[l])
        ax2.scatter(np.ones(len(arr)) * (i + 1)
                     + 0.05 * rng.standard_normal(len(arr)),
                     arr, s=18, color="#d62728", alpha=0.6, edgecolors="none",
                     zorder=3)
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_ylabel(r"correlation between $(b/c)^{*}$ and node centrality",
                    fontsize=14)
    ax2.tick_params(axis="both", labelsize=13)
    ax2.grid(alpha=0.25, axis="y")
    fig2.tight_layout()
    fig2.savefig("fig5_R3_per_node_centrality.png", dpi=180,
                  bbox_inches="tight")
    print("# Saved fig5_R3_per_node_centrality.png")


if __name__ == "__main__":
    main()
