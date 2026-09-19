"""
r3_larger_networks_partA.py: per-initial-condition (b/c)*_single versus node
centrality on larger single-layer networks.  Produces S1 Text, Table B.

Written for Reviewer 3, comment 2, of the first revision round (Part A of the
analysis plan proposed by co-author N. Masuda).  The linear systems of each
network are LU-factorised once and reused for all initial conditions.

SI item produced
----------------
* S1 Text, Text E.2 ("Impact of the initial condition"), Table B: for each of
  11 larger single-layer networks (ER and BA with N = 15, 30, 50, 75, 100, one
  instance each, plus layer 1 of the LLF network), the Pearson correlation
  coefficient across nodes v between (b/c)*_single obtained when v is the
  single initial cooperator and v's degree, closeness centrality, betweenness
  centrality and local clustering coefficient.  Source of Table B: the R_*
  columns of r3_part_A_table.csv.  The "N/A" cell (ER, N=15, local clus.)
  corresponds to R_clustering = NaN, which the script records when the local
  clustering coefficient is 0 on every node of a network.
* Supporting role: get_network, SIZES, ER_P, BA_M and RULE are imported by
  r3_larger_networks_partB.py (and through it by
  r3_larger_networks_partB_extended.py), which builds the two-layer networks
  of the "larger" column of S1 Text Table A with the same sizes, parameters
  and layer-1 seeds as used here.

How to run
----------
    python r3_larger_networks_partA.py                  # from this folder
    python <path to this folder>/r3_larger_networks_partA.py   # from elsewhere

No environment variables and no command-line arguments.  The two empirical
edge lists are located relative to this file (data/empirical_networks/ in the
same folder), so the script can be started from any directory; the three
CSVs are written to the current working directory.  Fully deterministic
(every random seed is fixed in the code).

Inputs
------
* Generated in the script (get_network / collect_networks):
    ER  G(N, p = 0.1) from nx.erdos_renyi_graph.  A numpy Generator created
        with np.random.default_rng(10*N) draws the networkx seed
        (rng.integers(0, 2**31 - 1)).
    BA  (N, m = 2) from ba_julia.ba_julia(N, 2, seed=10*N + 1): preferential
        attachment starting from a star centred at node m, the convention of
        the manuscript's Julia code (see ba_julia.py).
* data/empirical_networks/edges.csv      VC7 multilayer edge list; the rows
                                         with layer id 1 are used.  VC7 is
                                         excluded from Table B because its
                                         layer-1 network gives
                                         (b/c)*_single < 0 for every initial
                                         condition.
* data/empirical_networks/law_edges.csv  LLF multilayer edge list; the rows
                                         with the smallest layer id in the
                                         file (1) are used.

Method, for each network (5 ER + 5 BA instances with N in {15, 30, 50, 75,
100}, then layer 1 of the two empirical networks):

  1. Take layer 1 of the network.
  2. Compute (b/c)*_single for every node v as the single initial cooperator.
     ((b/c)*_single = -theta_2 / (theta_1 - theta_3), main-text Eq. (5) at
     r = 1; see fast_per_init_bc_singles for the recurrences solved.)
  3. If (b/c)*_single is finite and positive for ALL initial nodes, flag the
     network as "cooperative"; otherwise leave it out of the summary table
     (the plan was to avoid spite networks).
  4. Within each cooperative network, compute the Pearson R across nodes
     between (b/c)*_single and the node centralities (k_v, closeness_v,
     betweenness_v, clustering_v).
  5. Write the tables.

Outputs (written to the current working directory)
--------------------------------------------------
  r3_part_A_per_node.csv          per-(network, node) data: node, degree,
                                  closeness, betweenness, clustering,
                                  theta_diff (= theta_1 - theta_3), theta2,
                                  bc_single, network  (634 rows)
  r3_part_A_per_network.csv       one row per network (12 rows, VC7
                                  included): network, N (number of nodes),
                                  E (number of edges), n_finite,
                                  n_pos_bc_single, all_init_cooperative,
                                  bc_single_min/max/median, and R_* and p_*
                                  for the four centralities
  r3_part_A_table.csv             summary table, source of S1 Text Table B
                                  (first prepared for the response letter):
                                  the 11 cooperative networks with columns
                                  network, N, E, R_degree, R_closeness,
                                  R_betweenness, R_clustering

Runtime
-------
About 9.5 min wall-clock on an Apple M1, single process.  Almost all of it
is spent on the two N = 100 instances (roughly 5.5 min for ER and 3 min for
BA), whose beta_ij systems have N^2 = 10,000 unknowns and are solved with a
sparse LU factorisation (scipy.sparse.linalg.splu).

Dependencies
------------
Python 3 with numpy 2.1.3, networkx 3.4.2, scipy 1.15.3, pandas 2.2.3 (the
versions used to produce the published tables).  Local modules in the same
folder:
  larger_networks_solver.py   _transition_matrix_sparse (random-walk matrix
                              P and reproductive values pi), _compute_thetas
  ba_julia.py                 ba_julia (imported inside get_network)
"""
from __future__ import annotations
import os, time
import warnings, scipy.sparse as sp
# The row assignment A_mat[-1, :] = pi in fast_per_init_bc_singles acts on a
# CSR matrix (the difference of two LIL matrices is CSR), for which scipy
# prints a SparseEfficiencyWarning; the warning does not affect the results.
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse.linalg as spla
from scipy.stats import pearsonr

from larger_networks_solver import (
    _transition_matrix_sparse, _compute_thetas,
)

# ---- Match fig4_style_random.py conventions exactly ----
# These constants are also imported by r3_larger_networks_partB.py, so the
# two-layer networks behind the "larger" column of S1 Text Table A use the
# same sizes, parameters and layer-1 seeds as the single-layer networks here.
SIZES = [15, 30, 50, 75, 100]   # network sizes N (also used in the labels)
ER_P  = 0.1                     # ER edge probability p
BA_M  = 2                       # BA: edges brought by each new node (m)
RULE  = "dB-dB"                 # not used in this script's computations: the
                                # layer-1 beta recurrences solved below are the
                                # dB ones, and layer 1 uses dB under both dB-dB
                                # and dB-Bd, so (b/c)*_single is rule-independent
R_DUMMY = 2.0   # bc_star_single doesn't depend on r; use a fixed dummy.
                # (Not referenced anywhere in this script.)


def get_network(model: str, N: int, param, seed: int) -> nx.Graph:
    """Single-layer ER or BA network for the per-initial-condition analysis.

    Same construction as fig4_style_random.get_network.  No layer-2 shuffle
    is applied here, because Part A analyses single-layer networks.

    Args:
        model: "ER" or "BA".
        N:     number of nodes passed to the graph generator.
        param: for "ER" the edge probability p (float); for "BA" the number
               of edges m attached by each new node (cast to int).
        seed:  integer seed.  For "ER", a numpy Generator created with
               np.random.default_rng(seed) draws the networkx seed
               (rng.integers(0, 2**31 - 1)); for "BA" the seed is passed to
               networkx directly.

    Returns:
        Undirected nx.Graph with integer node labels 0..n-1.
        ER: built from nx.erdos_renyi_graph(N, p, seed=<drawn seed>).
        BA: ba_julia.ba_julia(N, m, seed), i.e. preferential attachment
        starting from a star centred at node m (the convention of the
        manuscript's Julia code; see ba_julia.py).
    """
    if model == "ER":
        rng = np.random.default_rng(seed)
        G = nx.erdos_renyi_graph(N, param, seed=int(rng.integers(0, 2**31 - 1)))
        if not nx.is_connected(G):
            cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(cc).copy()
            G = nx.convert_node_labels_to_integers(G)
        return G
    elif model == "BA":
        from ba_julia import ba_julia
        return ba_julia(N, int(param), seed=seed)
    raise ValueError(model)


def load_VC7_layer1() -> nx.Graph | None:
    """Layer 1 of the empirical VC7 network as a single-layer simple graph.

    Reads data/empirical_networks/edges.csv in the folder of this script
    (columns source, target, weight, layer; the leading "# ..." header line
    is skipped), keeps the rows with layer id 1, drops self-loops, merges
    duplicate and reciprocal rows into unordered pairs, and builds an
    undirected, unweighted simple graph.  Nodes are relabelled 0..n-1 in
    insertion order, i.e. the order in which they first appear when
    the de-duplicated edge set is iterated.  This order is deterministic, but
    the labels do not coincide with the node ids in the file.

    Returns:
        nx.Graph, or None when the file is not found.  In that case the VC7
        row is absent from every output CSV and no error is raised.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "empirical_networks", "edges.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, comment="#", names=["i", "j", "w", "layer"])
    sub = df[df["layer"] == 1]          # rows with layer id 1 only
    pairs = set()
    for _, row in sub.iterrows():
        a, b = int(row["i"]), int(row["j"])
        if a == b: continue             # drop self-loops
        pairs.add((min(a, b), max(a, b)))   # unordered pair -> undirected, deduplicated
    G = nx.Graph()
    G.add_edges_from(pairs)
    if not nx.is_connected(G):
        cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(cc).copy()
    G = nx.convert_node_labels_to_integers(G)   # labels 0..n-1 in insertion order
    return G


def load_law_layer1() -> nx.Graph | None:
    """Layer 1 of the empirical LLF (law firm) network as a single-layer simple graph.

    Same processing as load_VC7_layer1 but for
    data/empirical_networks/law_edges.csv in the folder of this script,
    keeping the rows whose layer id is the smallest one present in the file
    (1).  Nodes are relabelled 0..n-1 in insertion order, so the labels do
    not coincide with the node ids in the file.

    Returns:
        nx.Graph, or None when the file is not found.  In that case the LLF
        row is absent from every output CSV and no error is raised.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "empirical_networks", "law_edges.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, comment="#", names=["i", "j", "w", "layer"])
    sub = df[df["layer"] == df["layer"].min()]   # smallest layer id in the file (1)
    pairs = set()
    for _, row in sub.iterrows():
        a, b = int(row["i"]), int(row["j"])
        if a == b: continue             # drop self-loops
        pairs.add((min(a, b), max(a, b)))   # unordered pair -> undirected, deduplicated
    G = nx.Graph()
    G.add_edges_from(pairs)
    if not nx.is_connected(G):
        cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(cc).copy()
    G = nx.convert_node_labels_to_integers(G)   # labels 0..n-1 in insertion order
    return G


def collect_networks() -> list[tuple[str, nx.Graph]]:
    """Assemble the ordered list of (label, graph) pairs analysed by main().

    For each N in SIZES: the ER instance with seed 10*N + 0, then the BA
    instance with seed 10*N + 1.  These are exactly the layer-1 ("G1") seeds
    used by fig4_style_random.py and r3_larger_networks_partB.py (their layer
    2 uses seeds 10*N + 7 and 10*N + 8), hence the "(G1)" tag in the labels.
    Layer 1 of VC7 and of LLF are appended afterwards, each only if its edge
    file was found.  The row order of all three output CSVs follows this
    list.

    Returns:
        list of (label, nx.Graph); 12 entries when both empirical files are
        present (10 ER/BA + VC7 + LLF).
    """
    networks = []
    for N in SIZES:
        G_er = get_network("ER", N, ER_P, seed=10 * N + 0)
        networks.append((f"ER N={N} p={ER_P} (G1)", G_er))
        G_ba = get_network("BA", N, BA_M, seed=10 * N + 1)
        networks.append((f"BA N={N} m={BA_M} (G1)", G_ba))
    G_vc7 = load_VC7_layer1()
    if G_vc7 is not None:
        networks.append(("Empirical VC7 (layer 1)", G_vc7))
    G_law = load_law_layer1()
    if G_law is not None:
        networks.append(("Empirical Law (layer 1)", G_law))
    return networks


def fast_per_init_bc_singles(G: nx.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (b/c)*_single for each init in G. Returns (bc_singles, theta_diffs, theta2s).

    LU-factors the beta and beta_ij systems once per G, then back-subs per init.

    Equations solved (S1 Text, Text B, dB rule in layer 1; single layer,
    i.e. r = 1):
      * P = D^-1 A is the random-walk matrix p^[1]_ij and pi_i = k_i / sum_l k_l
        the reproductive value (RV) of node i under dB, both from
        larger_networks_solver._transition_matrix_sparse.  xi is the indicator
        of the initial cooperator and xi_hat = sum_i pi_i xi_i its RV-weighted
        frequency (= pi[v] for a single initial cooperator on v).
      * Diagonal terms, S1 Text Eq. (S52):
            beta_ii = N (xi_i - xi_hat) + sum_k p_ik beta_kk,
        i.e. (I - P) beta_diag = N (xi - xi_hat).  The last row of this
        system is replaced by the normalisation sum_i pi_i beta_ii = 0
        (S1 Text Eq. (S37)), which selects the solution used in the SI.
      * Off-diagonal terms, S1 Text Eq. (S54):
            beta_ij = (N/2)(xi_i xi_j - xi_hat)
                      + (1/2) sum_k p_ik beta_kj + (1/2) sum_k p_jk beta_ik,
        written for the N^2-vector vec(beta) with index (i, j) -> i*N + j as
            (I - 1/2 (P kron I) - 1/2 (I kron P)) vec(beta)
                = (N/2) vec(xi xi^T - xi_hat).
        (P kron I) acts on the first index (sum_k P[i,k] beta[k,j]) and
        (I kron P) on the second (sum_k P[j,k] beta[i,k]).  The N rows with
        i = j are replaced by identity rows whose right-hand side is beta_ii
        from the diagonal system, so both systems are solved consistently.
      * theta_n = sum_ij pi_i (P^n)_ij beta_ij for n = 1, 2, 3,
        S1 Text Eq. (S47), via larger_networks_solver._compute_thetas.
      * The cooperation condition c theta_2 + b (theta_1 - theta_3)
        - (r - 1) phi_20 > 0 (main-text Eq. (3), S1 Text Eq. (S50)) at r = 1
        gives the single-layer threshold
        (b/c)*_single = -theta_2 / (theta_1 - theta_3) (main-text Eq. (5)
        with r = 1).

    Args:
        G: undirected graph with node labels 0..N-1.  Matrix index v is node
           v (both here and in _transition_matrix_sparse the index follows
           sorted(G.nodes())).

    Returns:
        (bc_singles, theta_diffs, theta2s): float arrays of length N.
        bc_singles[v] is (b/c)*_single with the single initial cooperator on
        node v, theta_diffs[v] = theta_1 - theta_3 and theta2s[v] = theta_2 for
        that initial condition.  bc_singles[v] is NaN when
        |theta_1 - theta_3| <= 1e-12 (does not occur for the published
        networks: n_finite == N in every row of r3_part_A_per_network.csv).
    """
    P, pi = _transition_matrix_sparse(G)
    N = P.shape[0]
    nn = N * N

    # Beta_i system: (I - P) beta = N*(xi - xi_hat); last row replaced by sum pi*beta = 0.
    A_mat = sp.eye(N, format="lil") - P.tolil()
    A_mat[-1, :] = pi                       # constraint row; its rhs is zeroed per init below
    A_lu = spla.splu(A_mat.tocsc())         # factorise once, reuse for all N inits

    # Beta_ij system: (I - 0.5(P x I) - 0.5(I x P)) beta = (N/2)(outer(xi,xi) - xi_hat)
    # with N constraint rows replaced by identity (beta_ii = beta_i[i]).
    I_N = sp.eye(N, format="csr")
    M_ij = (sp.eye(nn, format="csr")
             - 0.5 * sp.kron(P, I_N)        # (P x I): sums over the first index i
             - 0.5 * sp.kron(I_N, P))       # (I x P): sums over the second index j
    M_ij = M_ij.tolil()
    diag_rows = [i * N + i for i in range(N)]   # vector positions of the (i, i) entries
    for r in diag_rows:
        M_ij.rows[r] = [r]                  # row (i,i) becomes  1 * beta_ii = rhs[(i,i)]
        M_ij.data[r] = [1.0]
    M_ij_lu = spla.splu(M_ij.tocsc())       # factorise once (N^2 unknowns), reuse per init

    bc_list, td_list, t2_list = [], [], []
    for v in range(N):
        xi = np.zeros(N); xi[v] = 1.0       # single initial cooperator on node v
        xi_hat = pi[v]                      # = pi @ xi (RV-weighted initial frequency)

        # Beta_i
        rhs_i = N * (xi - xi_hat); rhs_i[-1] = 0.0   # rhs[-1] = 0 matches the constraint row
        beta_i = A_lu.solve(rhs_i)

        # Beta_ij
        src = (N / 2.0) * (np.outer(xi, xi) - xi_hat)
        rhs = src.ravel()                   # row-major: (i, j) -> i*N + j
        for i in range(N):
            rhs[i * N + i] = beta_i[i]      # pin the diagonal to the beta_ii solution
        x = M_ij_lu.solve(rhs)
        beta_full = x.reshape(N, N)         # beta_full[i, j] = beta_ij

        theta1, theta2, theta3 = _compute_thetas(P, pi, beta_full)
        td = theta1 - theta3
        # (b/c)*_single = -theta_2 / (theta_1 - theta_3); NaN if the denominator vanishes
        bc = (-theta2 / td) if abs(td) > 1e-12 else np.nan
        bc_list.append(bc); td_list.append(td); t2_list.append(theta2)
    return (np.array(bc_list), np.array(td_list), np.array(t2_list))


def per_node_quantities(G: nx.Graph) -> pd.DataFrame:
    """Per-node centralities and per-init (b/c)*_single for one network.

    Centralities use the networkx defaults: degree k_v; closeness centrality
    (nx.closeness_centrality); betweenness centrality
    (nx.betweenness_centrality, exact and normalised); local clustering
    coefficient (nx.clustering, which assigns 0 to nodes of degree < 2, as
    stated in S1 Text, Text E.2).

    Args:
        G: graph with integer labels 0..N-1 (as returned by get_network /
           load_*_layer1).

    Returns:
        DataFrame with one row per node in sorted node order -- the same order
        as the arrays from fast_per_init_bc_singles -- and columns node,
        degree, closeness, betweenness, clustering, theta_diff
        (= theta_1 - theta_3), theta2, bc_single.  main() adds the "network"
        column before writing r3_part_A_per_node.csv.
    """
    nodes = sorted(G.nodes())               # same ordering as the P matrix rows
    deg = dict(G.degree())
    cl  = nx.closeness_centrality(G)
    bt  = nx.betweenness_centrality(G)
    lc  = nx.clustering(G)
    bc, td, t2 = fast_per_init_bc_singles(G)
    rows = []
    for i, v in enumerate(nodes):
        rows.append({
            "node": v,
            "degree": deg[v],
            "closeness": cl[v],
            "betweenness": bt[v],
            "clustering": lc[v],
            "theta_diff": td[i],
            "theta2": t2[i],
            "bc_single": bc[i],
        })
    return pd.DataFrame(rows)


def per_network_correlations(per_node: pd.DataFrame) -> dict:
    """Pearson R between (b/c)*_single and each centrality across the nodes of one network.

    The nodes of the network are the samples (S1 Text, Text E.2, Table B:
    one value per network and centrality).

    Args:
        per_node: output of per_node_quantities for one network.

    Returns:
        dict with keys R_degree, p_degree, R_closeness, p_closeness,
        R_betweenness, p_betweenness, R_clustering, p_clustering from
        scipy.stats.pearsonr.  R and p are NaN when either variable is
        constant across nodes (std < 1e-12).  This happens for the local
        clustering coefficient of the ER N=15 instance, which is
        triangle-free (all clustering values are 0), and gives the "N/A"
        cell of Table B.  The p-values are stored in
        r3_part_A_per_network.csv but are not reported in the SI.
    """
    out = {}
    y = per_node["bc_single"].values
    for col in ["degree", "closeness", "betweenness", "clustering"]:
        x = per_node[col].values
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            # pearsonr is undefined for a constant input; record NaN instead
            r, p = np.nan, np.nan
        else:
            r, p = pearsonr(x, y)
        out[f"R_{col}"] = r
        out[f"p_{col}"] = p
    return out


def main():
    """Run the whole analysis and write the three CSV files (see module docstring).

    A network is flagged all_init_cooperative when every one of its nodes,
    taken as the single initial cooperator, gives a finite and positive
    (b/c)*_single (n_pos == n_finite == number of nodes).  Only those networks
    are written to r3_part_A_table.csv (source of S1 Text Table B); the
    per-node and per-network CSVs contain every network (12 when both
    empirical files are present; VC7 is the one that is not flagged).  The
    console shows the same statistics with 3 or 4 decimals.
    """
    networks = collect_networks()
    print(f"# {len(networks)} networks collected", flush=True)

    per_node_rows = []
    per_net_rows = []
    for label, G in networks:
        actual_N = G.number_of_nodes()      # number of nodes of the graph analysed
        E = G.number_of_edges()
        t0 = time.time()
        df_node = per_node_quantities(G)
        dt = time.time() - t0
        df_node["network"] = label
        per_node_rows.append(df_node)

        finite = np.isfinite(df_node["bc_single"].values)
        n_pos = ((df_node["bc_single"].values > 0) & finite).sum()
        n_finite = int(finite.sum())
        # "cooperative" network <=> all actual_N inits give a finite, positive (b/c)*_single
        all_positive = (n_pos == n_finite) and (n_finite == actual_N)
        cor = per_network_correlations(df_node)
        per_net_rows.append({
            "network": label, "N": actual_N, "E": E,
            "n_finite": n_finite,
            "n_pos_bc_single": n_pos,
            "all_init_cooperative": all_positive,
            "bc_single_min": df_node["bc_single"].min(),
            "bc_single_max": df_node["bc_single"].max(),
            "bc_single_median": df_node["bc_single"].median(),
            **cor,
        })
        print(f"\n# {label}  N={actual_N}, E={E}  ({dt:.1f}s)", flush=True)
        print(f"  bc_single: min={df_node['bc_single'].min():+.3f}, "
              f"max={df_node['bc_single'].max():+.3f}, "
              f"median={df_node['bc_single'].median():+.3f},  "
              f"n_pos={n_pos}/{n_finite}  "
              f"all_pos={'yes' if all_positive else 'NO'}", flush=True)
        print(f"  Pearson R(b/c, k_i)         = {cor['R_degree']:+.4f}  (p={cor['p_degree']:.2e})", flush=True)
        print(f"  Pearson R(b/c, closeness_i) = {cor['R_closeness']:+.4f}  (p={cor['p_closeness']:.2e})", flush=True)
        print(f"  Pearson R(b/c, between_i)   = {cor['R_betweenness']:+.4f}  (p={cor['p_betweenness']:.2e})", flush=True)
        print(f"  Pearson R(b/c, clust_i)     = {cor['R_clustering']:+.4f}  (p={cor['p_clustering']:.2e})", flush=True)

    pd.concat(per_node_rows, ignore_index=True).to_csv("r3_part_A_per_node.csv", index=False)
    df_net = pd.DataFrame(per_net_rows)
    df_net.to_csv("r3_part_A_per_network.csv", index=False)

    coop = df_net[df_net["all_init_cooperative"]].copy()
    print(f"\n# {len(coop)} of {len(df_net)} networks are 'cooperative' "
          f"(all (b/c)*_single > 0)", flush=True)
    # Summary table for the cooperative networks (11 rows); source of S1 Text Table B
    table_cols = ["network", "N", "E",
                   "R_degree", "R_closeness", "R_betweenness", "R_clustering"]
    coop[table_cols].to_csv("r3_part_A_table.csv", index=False)
    print("\n# Final table (cooperative networks only):")
    print(coop[table_cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}" if isinstance(v, float) else v))


if __name__ == "__main__":
    main()
