"""
r3_larger_networks_partB_extended.py: Pearson correlation between
|d(b/c)*/dr| and seven metrics of each layer of the larger two-layer
networks.  Produces the "larger" column of S1 Text, Table A.

Written for Reviewer 3, comment 2, of the first revision round (extended
Part B, requested by co-author N. Masuda on June 21).  Compared with
r3_larger_networks_partB.py, mean betweenness is dropped (it is not
comparable across networks), the network metrics are the same seven as used
for the six-node networks in Table A, and they are computed separately for
layer 1 and layer 2:
  mean degree, maximum degree, variance of degree, mean closeness,
  clustering coefficient, diameter, algebraic connectivity.
|slope| (= |phi_20 / (theta_1 - theta_3)|, mean over initial conditions) and
the cooperative filter are the same as in r3_larger_networks_partB.py, so the
11 cooperative two-layer networks and their mean |slope| are identical in
the two scripts.

SI item produced
----------------
S1 Text, Text E.1 ("Impact of the network structure"), Table A, column
"larger" (network layer 1): the Pearson correlation coefficient between
|d(b/c)*/dr| and seven metrics of the layer-1 network, each two-layer
network being one sample (n = 11: two-layer ER and BA networks with N = 15,
30, 50, 75, 100, one instance each, under the dB-dB rule, plus the LLF
network; VC7 is excluded because its layer 1 gives (b/c)* < 0 for every
initial condition).  Source of that column: the "layer 1" rows of
r3_part_B_extended_correlations.csv.  The "layer 2" rows of the same CSV
are computed but not shown in the SI (the SI explains why only layer-1
metrics are reported for the larger networks).

Quantity being correlated.  Under weak selection the cooperation condition
of the two-layer model (main-text Eq. (3)) reads
    c*theta_2 + b*(theta_1 - theta_3) - (r - 1)*phi_{2,0} > 0,
so  (b/c)* = (b/c)*_single + (r - 1)*phi_{2,0} / (c*(theta_1 - theta_3))
(main-text Eq. (5)) and d(b/c)*/dr = phi_{2,0} / (c*(theta_1 - theta_3)).
With c = 1 this is the "slope" computed by
r3_larger_networks_partB.fast_per_init_bilayer; its absolute value, averaged
over the N initial conditions (initial cooperator in layer 1 = initial
mutant in layer 2 = node v, v = 0..N-1), is the y variable
("mean_abs_slope") of every correlation in this script.

How to run
----------
    python r3_larger_networks_partB_extended.py                 # from this folder
    python <path to this folder>/r3_larger_networks_partB_extended.py   # from elsewhere
No environment variables and no command-line arguments.  Both output CSVs
are written into the current working directory.  The empirical edge lists
are located relative to r3_larger_networks_partB.py (data/empirical_networks/
in the same folder), so the script can be started from any directory.  Only
main() of THIS file runs; r3_larger_networks_partB (and, through it,
r3_larger_networks_partA) are imported for their functions and constants
only; their own main() is guarded by ``if __name__ == "__main__"`` and is
not executed.

Inputs
------
All twelve two-layer networks come from
r3_larger_networks_partB.collect_bilayers():
  * ER (N in SIZES = [15, 30, 50, 75, 100], p = ER_P = 0.1):
      layer 1 = get_network("ER", N, 0.1, seed=10*N),
      layer 2 = get_network("ER", N, 0.1, seed=10*N + 7),
    where np.random.default_rng(seed) draws the networkx seed; the two
    layers are then aligned by integer node label in collect_bilayers.
  * BA (m = BA_M = 2):
      layer 1 = ba_julia.ba_julia(N, 2, seed=10*N + 1),
      layer 2 = ba_julia.ba_julia(N, 2, seed=10*N + 8), then relabelled by
                ba_julia.shuffle_layer2(G2, seed=20260525 + 10*N).
  * Empirical: the rows with layer ids 1 and 2 of
    data/empirical_networks/edges.csv (VC7) and
    data/empirical_networks/law_edges.csv (LLF), read by
    r3_larger_networks_partB.load_VC7_bilayer and load_law_bilayer.
Update rule: dB-dB, hard-coded in r3_larger_networks_partB.fast_per_init_bilayer.

Outputs
-------
r3_part_B_extended_per_network.csv   12 rows, one per two-layer network
    (VC7 included): network, N, E1, E2, all_init_cooperative,
    mean_abs_slope, and L1_<prop> and L2_<prop> for the seven <prop>
    suffixes listed in PROPS.
r3_part_B_extended_correlations.csv  14 rows = 7 properties x {layer 1,
    layer 2}: property, layer, Pearson_R, p_value, n.  n is the number of
    cooperative two-layer networks for which both the property and
    mean_abs_slope are finite (graph_props records NaN when networkx cannot
    compute a metric).

Runtime
-------
About 4.2 min wall-clock on an Apple M1 laptop when run alone (about 8 min
while another job of similar size was running).  The time is dominated by
the sparse LU factorisations of the N^2 = 10 000-unknown gamma systems for
ER N=100 and BA N=100; the smaller networks take seconds.

Dependencies
------------
Python packages (versions used for the published table): numpy 2.1.3,
networkx 3.4.2, scipy 1.15.3, pandas 2.2.3.
Local modules (same folder): r3_larger_networks_partB.py (collect_bilayers,
fast_per_init_bilayer), which itself imports r3_larger_networks_partA.py
(get_network, SIZES, ER_P, BA_M), larger_networks_solver.py
(_transition_matrix_sparse, _compute_thetas, _compute_phis) and ba_julia.py
(ba_julia, shuffle_layer2).
"""
from __future__ import annotations
import warnings, scipy.sparse as sp
# fast_per_init_bilayer assigns a row of a CSR matrix (A_mat[-1, :] = pi1; the
# difference of two LIL matrices is CSR), for which scipy emits
# SparseEfficiencyWarning; silence only that warning class.
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import pearsonr

# Importing the module does NOT run its main() (guarded by __name__); it only
# provides the bilayer generator and the per-init solver.  Transitively this
# also imports r3_larger_networks_partA, larger_networks_solver and ba_julia.
from r3_larger_networks_partB import collect_bilayers, fast_per_init_bilayer


def graph_props(G: nx.Graph, tag: str) -> dict:
    """Seven network-level metrics of one layer, keyed as f"{tag}_<prop>".

    These are the same seven metrics used for the six-node networks in SI
    Table A (Text E.1), evaluated here on one layer of a larger two-layer
    network.

    Args:
        G:   undirected networkx graph of one layer (nodes labelled 0..N-1).
        tag: column prefix -- "L1" for layer 1 (donation-game layer) or
             "L2" for layer 2 (constant-selection layer).

    Returns:
        dict with keys
          {tag}_mean_degree      mean degree <k>
          {tag}_max_degree       maximum degree
          {tag}_var_degree       variance of the degree (population variance,
                                 numpy default ddof=0)
          {tag}_mean_closeness   mean of nx.closeness_centrality over nodes
          {tag}_clustering       mean local clustering coefficient
                                 (nx.clustering averaged over nodes)
          {tag}_diameter         nx.diameter; NaN if nx.diameter raises an
                                 exception
          {tag}_alg_connectivity algebraic connectivity (second-smallest
                                 Laplacian eigenvalue,
                                 nx.algebraic_connectivity with seed=0); NaN
                                 if networkx raises an exception
    """
    deg = np.array([d for _, d in G.degree()], dtype=float)
    cl = np.array(list(nx.closeness_centrality(G).values()))
    lc = np.array(list(nx.clustering(G).values()))
    # Record NaN if nx.diameter raises an exception.
    try:
        diam = float(nx.diameter(G))
    except Exception:
        diam = np.nan
    # seed=0 fixes the random start vector of networkx's default
    # 'tracemin_pcg' eigensolver, so the value is reproducible.  The guard
    # records NaN if networkx raises an exception (e.g. fewer than two nodes).
    try:
        alg = float(nx.algebraic_connectivity(G, seed=0))
    except Exception:
        alg = np.nan
    return {
        f"{tag}_mean_degree":   float(deg.mean()),
        f"{tag}_max_degree":    float(deg.max()),
        f"{tag}_var_degree":    float(deg.var()),
        f"{tag}_mean_closeness": float(cl.mean()),
        f"{tag}_clustering":    float(lc.mean()),
        f"{tag}_diameter":      diam,
        f"{tag}_alg_connectivity": alg,
    }


# Column suffixes produced by graph_props, in the row order of SI Table A
# (and of r3_part_B_extended_correlations.csv).
PROPS = ["mean_degree", "max_degree", "var_degree", "mean_closeness",
         "clustering", "diameter", "alg_connectivity"]
# Human-readable names written to the "property" column of the correlations
# CSV and used for the console table.
PRETTY = {
    "mean_degree": "mean degree <k>",
    "max_degree": "maximum degree",
    "var_degree": "variance of degree",
    "mean_closeness": "mean closeness",
    "clustering": "clustering coefficient",
    "diameter": "diameter",
    "alg_connectivity": "algebraic connectivity",
}


def main():
    """Build the per-bilayer table and the cross-network correlations.

    Mirrors r3_larger_networks_partB.main with the extended, per-layer metric
    set.  Steps:
      1. collect_bilayers() -> 12 (label, G1, G2) triples: ER and BA bilayers
         for N in SIZES, then "Empirical VC7" and "Empirical Law".
      2. For each bilayer, fast_per_init_bilayer(G1, G2) returns two length-N
         arrays indexed by the initial node v (initial cooperator in layer 1
         = initial mutant in layer 2 = v):
             bc[v] = (b/c)*_single = -theta_2 / (theta_1 - theta_3)
             sl[v] = |phi_{2,0} / (theta_1 - theta_3)| = |d(b/c)*/dr| (c = 1)
         under the dB-dB rule (NaN where |theta_1 - theta_3| <= 1e-12).
         A bilayer is "cooperative" iff bc[v] is finite and > 0 for every v;
         mean_abs_slope is the nanmean of sl over v.
      3. Append graph_props of G1 (L1_* columns) and G2 (L2_* columns) and
         write r3_part_B_extended_per_network.csv (all 12 bilayers).
      4. Keep only cooperative bilayers (11; VC7 is dropped) and, for each of
         the 7 properties and each layer, compute scipy.stats.pearsonr between
         mean_abs_slope and the property across bilayers; write
         r3_part_B_extended_correlations.csv.  The "layer 1" rows are the
         "larger" column of SI Table A.
      5. Print a property x {layer 1, layer 2} summary of R and p with three
         decimals.

    No arguments; no return value; writes the two CSVs into the cwd.
    """
    bilayers = collect_bilayers()
    print(f"# {len(bilayers)} bilayers collected", flush=True)
    rows = []
    for label, G1, G2 in bilayers:
        bc, sl = fast_per_init_bilayer(G1, G2)
        # Cooperative filter, identical to the one in
        # r3_larger_networks_partB.per_bilayer_stats: every one of the N
        # inits must give a finite, positive (b/c)*_single (the SI's
        # "layer-1 network has (b/c)* > 0 regardless of the initial
        # condition").  Requiring finite.sum() == N also rejects a network
        # for which any init returned NaN.
        finite = np.isfinite(bc)
        n_pos = int(((bc > 0) & finite).sum())
        all_coop = (n_pos == int(finite.sum())) and (int(finite.sum()) == G1.number_of_nodes())
        # N, E1, E2: number of nodes and numbers of edges in layers 1 and 2.
        rec = {"network": label, "N": G1.number_of_nodes(),
               "E1": G1.number_of_edges(), "E2": G2.number_of_edges(),
               "all_init_cooperative": all_coop,
               # nanmean skips inits whose slope is NaN (theta_1 ~ theta_3).
               "mean_abs_slope": float(np.nanmean(sl))}
        rec.update(graph_props(G1, "L1"))   # layer 1: donation game
        rec.update(graph_props(G2, "L2"))   # layer 2: constant selection
        rows.append(rec)
        print(f"# {label:18}: |slope|={rec['mean_abs_slope']:.4f} coop={all_coop}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("r3_part_B_extended_per_network.csv", index=False)

    # Sample set for the correlations: the cooperative bilayers only (11 of
    # 12 -- "Empirical VC7" is the one excluded).
    coop = df[df["all_init_cooperative"]].copy()
    y = coop["mean_abs_slope"].values
    print(f"\n# {len(coop)} cooperative bilayers\n")
    cor_rows = []
    for prop in PROPS:
        for layer in ["L1", "L2"]:
            col = f"{layer}_{prop}"
            x = coop[col].values
            # Drop networks where the property is NaN; `n` records how many
            # (property, slope) pairs were used.
            mask = np.isfinite(x) & np.isfinite(y)
            # Guard against degenerate inputs: fewer than three usable samples
            # or a constant property (pearsonr would divide by zero).
            if mask.sum() < 3 or np.ptp(x[mask]) == 0:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(x[mask], y[mask])
            cor_rows.append({"property": PRETTY[prop],
                             "layer": "layer 1" if layer == "L1" else "layer 2",
                             "Pearson_R": r, "p_value": p, "n": int(mask.sum())})
    out = pd.DataFrame(cor_rows)
    out.to_csv("r3_part_B_extended_correlations.csv", index=False)

    # pretty print as a property x {layer1,layer2} table
    print(f"{'property':26} {'layer 1 R':>10} {'(p)':>9}   {'layer 2 R':>10} {'(p)':>9}")
    for prop in PROPS:
        r1 = out[(out.property == PRETTY[prop]) & (out.layer == "layer 1")].iloc[0]
        r2 = out[(out.property == PRETTY[prop]) & (out.layer == "layer 2")].iloc[0]
        print(f"{PRETTY[prop]:26} {r1.Pearson_R:>+10.3f} {r1.p_value:>9.3f}   "
              f"{r2.Pearson_R:>+10.3f} {r2.p_value:>9.3f}")
    print(f"\n# n = {len(coop)} cooperative bilayers")
    print("# saved r3_part_B_extended_{per_network,correlations}.csv")


if __name__ == "__main__":
    main()
