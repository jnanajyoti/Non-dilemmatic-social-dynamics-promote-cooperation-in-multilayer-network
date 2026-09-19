"""
r3_larger_networks_partB.py: |d(b/c)*/dr| versus layer-1 network metrics on
larger two-layer networks (four-metric version), and the two-layer network
construction and solver used by r3_larger_networks_partB_extended.py.

SI item supported
-----------------
S1 Text, Text E.1 ("Impact of the network structure"), Table A, column
"larger": the Pearson correlation coefficient between |d(b/c)*/dr| and
metrics of the layer-1 network over the 11 "cooperative" larger two-layer
networks (two-layer ER and BA networks with N = 15, 30, 50, 75, 100 under
the dB-dB rule, plus the LLF network; the VC7 network is excluded because
its layer 1 gives (b/c)* < 0 for every initial condition).

Role of this file
-----------------
* The published 7-metric table (mean degree, maximum degree, variance of
  degree, mean closeness, clustering coefficient, diameter, algebraic
  connectivity) is written by `r3_larger_networks_partB_extended.py`, which
  IMPORTS from this module the two-layer network construction
  (`collect_bilayers`, `load_VC7_bilayer`, `load_law_bilayer`) and the
  per-initial-condition solver (`fast_per_init_bilayer`).  Importing this
  module does not execute `main()` (guarded by `__name__`).
* Running this file directly produces the earlier 4-metric version of the
  same analysis (mean degree, mean closeness, mean betweenness, clustering
  coefficient).  Its CSVs are not printed in the SI, but the mean-degree,
  mean-closeness and clustering rows of r3_part_B_correlations.csv are
  identical (to all digits) to the corresponding layer-1 rows of the
  extended table, so this script doubles as a consistency check.

How to run
----------
    python r3_larger_networks_partB.py                  # from this folder
    python <path to this folder>/r3_larger_networks_partB.py   # from elsewhere
No environment variables and no command-line arguments.  The empirical edge
lists are located relative to this file (data/empirical_networks/ in the
same folder); the output CSVs are written to the current working directory.
The published run used Python 3.13.9 (Anaconda) with numpy 2.1.3,
networkx 3.4.2, scipy 1.15.3, pandas 2.2.3.

Inputs
------
* Generated in the script (see `collect_bilayers`), for N in
  r3_larger_networks_partA.SIZES = [15, 30, 50, 75, 100]:
    ER two-layer networks: layer 1 = get_network("ER", N, 0.1, seed=10*N),
      layer 2 = get_network("ER", N, 0.1, seed=10*N + 7).  For each layer a
      numpy Generator created with np.random.default_rng(seed) draws the
      networkx seed (see r3_larger_networks_partA.get_network).  The two
      layers are then aligned by integer node label (see collect_bilayers).
    BA two-layer networks: layer 1 = ba_julia(N, 2, seed=10*N + 1),
      layer 2 = ba_julia(N, 2, seed=10*N + 8) with node labels permuted by
      ba_julia.shuffle_layer2(seed=20260525 + 10*N).
* data/empirical_networks/edges.csv      VC7; rows with layer ids 1 and 2
* data/empirical_networks/law_edges.csv  LLF; rows with layer ids 1 and 2
  Both files: columns (source, target, weight, layer); the first line
  "# source, target, weight, layer" is a comment.  If a file is missing, the
  loader returns (None, None) and that two-layer network is left out of the
  outputs without an error.

Outputs (written to the current working directory)
--------------------------------------------------
* r3_part_B_per_network.csv   one row per two-layer network (12 rows, VC7
                              included): network, N, E1, E2,
                              all_init_cooperative, n_pos_bc_single,
                              mean_abs_slope, median_abs_slope, mean_degree,
                              mean_closeness, mean_betweenness, clustering
                              (the four metrics are those of LAYER 1)
* r3_part_B_correlations.csv  property, Pearson_R, p_value, n: Pearson R
                              between mean |slope| and each of the four
                              layer-1 metrics across the cooperative
                              two-layer networks (n = 11)

Runtime
-------
About 7 min wall-clock on an Apple M1 (about 18 min CPU) for all 12
two-layer networks; dominated by the N = 100 ER and BA cases, whose gamma
system has N^2 = 10,000 unknowns (one sparse LU factorisation per network,
then one back-substitution per initial condition).  No Monte Carlo
simulation is involved.

Dependencies
------------
numpy, networkx, scipy (sparse, sparse.linalg, stats), pandas; local
modules larger_networks_solver.py (`_transition_matrix_sparse`,
`_compute_thetas`, `_compute_phis`), r3_larger_networks_partA.py
(`get_network`, `SIZES`, `ER_P`, `BA_M`, `RULE`) and ba_julia.py
(`ba_julia` through partA.get_network, `shuffle_layer2`).

Quantities computed (S1 Text, Text B, dB-dB rule; c = 1 throughout)
-------------------------------------------------------------------
For every initial condition v (single cooperator on node v of layer 1 AND
single mutant on the same node label v of layer 2):
    (b/c)*_single = -theta_2 / (theta_1 - theta_3)      [main-text Eq. (5) at r = 1]
    slope         = d(b/c)*/dr = phi_{2,0} / (theta_1 - theta_3)
                    [from main-text Eq. (5): (b/c)* = (b/c)*_single
                              + (r - 1) phi_{2,0} / (c (theta_1 - theta_3))]
theta_n and phi_{n,m} are defined in S1 Text Eqs. (S47) and (S48) and
obtained from the beta_ij / gamma_ij recurrences (Eqs. (S52), (S54) and
(S56)) with the normalisations sum_i pi_i beta_ii = 0 and
sum_i pi_i gamma_ii = 0 (Eqs. (S37) and (S38)).  The per-network
|d(b/c)*/dr| entering the correlation is the mean over v of |slope|.

Notes from the original header
------------------------------
Written for Reviewer 3, comment 2, of the first revision round (Part B of
the analysis plan proposed by co-author N. Masuda).  Fast version: the
two-layer gamma system is LU-factorised once per (G1, G2) and reused for all
initial conditions.

Plan, from N. Masuda's email:
  > For coupling ... select large cooperative networks (not spite networks)
  > we are using, measure |slope| and measure the correlation between |slope|
  > and <k>, mean closeness, mean betweenness (just as a test), and
  > clustering coefficient.

For each large two-layer network (G1, G2) used in the paper:
  1. Compute per-init (b/c)*_single, slope = phi_20 / (theta_1 - theta_3).
  2. Keep the networks where (b/c)*_single > 0 for ALL inits (cooperative).
  3. For each cooperative network, compute the mean |slope| over inits.
  4. Compute G1's network-level properties:
        <k>, mean closeness, mean betweenness, clustering coefficient
  5. Cross-network Pearson R between mean |slope| and each property.

Outputs:
  r3_part_B_per_network.csv     mean |slope| and properties per network
  r3_part_B_correlations.csv    cross-network correlations (4-metric version;
                                the SI prints the 7-metric table written by
                                r3_larger_networks_partB_extended.py)
"""
from __future__ import annotations
import os, time
import warnings, scipy.sparse as sp
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse.linalg as spla
from scipy.stats import pearsonr

from larger_networks_solver import (
    _transition_matrix_sparse, _compute_thetas, _compute_phis,
)
# NOTE: load_VC7_layer1, load_law_layer1 and RULE are imported but not used
# in this module.  The empirical two-layer networks are built by the two
# loaders below, and `fast_per_init_bilayer` always solves the dB-dB gamma
# recurrence (RULE is "dB-dB" in r3_larger_networks_partA.py) instead of
# branching on RULE.
from r3_larger_networks_partA import (
    get_network, load_VC7_layer1, load_law_layer1,
    SIZES, ER_P, BA_M, RULE,
)


def load_VC7_bilayer():
    """Two-layer VC7 network from data/empirical_networks/edges.csv.

    Layer convention: the edge list (in the folder of this script) has
    columns (source, target, weight, layer) and a "#" comment header.  The
    rows with the two SMALLEST layer ids in the file (1 and 2) become layer
    1 (donation game) and layer 2 (constant selection), in that order.
    Weights are ignored, self-loops are dropped and each unordered pair is
    kept once, i.e. both layers are simple undirected unweighted graphs.

    Node alignment: both layers receive the UNION of the two node sets.  The
    original node ids are then relabelled 0..n-1 in increasing order using
    the SAME map for both layers, so node v of G1 and node v of G2 are the
    same individual.

    Returns:
        (G1, G2): nx.Graph pair on the common node set 0..n-1, or
                  (None, None) if the file is not found; `collect_bilayers`
                  then omits VC7 without an error.
    """
    # Edge list stored in data/empirical_networks/ next to this file.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "empirical_networks", "edges.csv")
    if not os.path.exists(path):
        return None, None   # guard: missing file -> network skipped, no error
    df = pd.read_csv(path, comment="#", names=["i", "j", "w", "layer"])
    layers = sorted(df["layer"].unique())[:2]   # the two smallest layer ids (1 and 2)
    Gs = []
    for lid in layers:
        sub = df[df["layer"] == lid]
        pairs = set()
        for _, row in sub.iterrows():
            a, b = int(row["i"]), int(row["j"])
            if a == b: continue                       # drop self-loops
            pairs.add((min(a, b), max(a, b)))         # undirected, de-duplicated
        G = nx.Graph(); G.add_edges_from(pairs); Gs.append(G)
    G1, G2 = Gs
    # Give both layers the union of the two node sets.
    nodes = sorted(set(G1.nodes()) | set(G2.nodes()))
    G1.add_nodes_from(nodes); G2.add_nodes_from(nodes)
    common = set(G1.nodes()) & set(G2.nodes())   # == union after the line above
    G1 = G1.subgraph(common).copy(); G2 = G2.subgraph(common).copy()
    if not nx.is_connected(G1):
        cc = max(nx.connected_components(G1), key=len)
        G1 = G1.subgraph(cc).copy(); G2 = G2.subgraph(cc).copy()
    # Same relabelling map for both layers: original id -> rank among sorted ids.
    map_nodes = {v: i for i, v in enumerate(sorted(G1.nodes()))}
    G1 = nx.relabel_nodes(G1, map_nodes); G2 = nx.relabel_nodes(G2, map_nodes)
    return G1, G2


def load_law_bilayer():
    """Two-layer LLF (law-firm) network from data/empirical_networks/law_edges.csv.

    Same file format, layer convention (rows with layer ids 1 and 2 become
    layers 1 and 2), edge cleaning and node alignment as `load_VC7_bilayer`.

    Returns:
        (G1, G2) on the common node set 0..n-1, or (None, None) if the file
        is not found; `collect_bilayers` then omits LLF without an error.
    """
    # Edge list stored in data/empirical_networks/ next to this file.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "empirical_networks", "law_edges.csv")
    if not os.path.exists(path):
        return None, None   # guard: missing file -> network skipped, no error
    df = pd.read_csv(path, comment="#", names=["i", "j", "w", "layer"])
    layers = sorted(df["layer"].unique())[:2]   # the two smallest layer ids (1 and 2)
    Gs = []
    for lid in layers:
        sub = df[df["layer"] == lid]
        pairs = set()
        for _, row in sub.iterrows():
            a, b = int(row["i"]), int(row["j"])
            if a == b: continue                       # drop self-loops
            pairs.add((min(a, b), max(a, b)))         # undirected, de-duplicated
        G = nx.Graph(); G.add_edges_from(pairs); Gs.append(G)
    G1, G2 = Gs
    # Give both layers the union of the two node sets.
    nodes = sorted(set(G1.nodes()) | set(G2.nodes()))
    G1.add_nodes_from(nodes); G2.add_nodes_from(nodes)
    common = set(G1.nodes()) & set(G2.nodes())   # == union after the line above
    G1 = G1.subgraph(common).copy(); G2 = G2.subgraph(common).copy()
    if not nx.is_connected(G1):
        cc = max(nx.connected_components(G1), key=len)
        G1 = G1.subgraph(cc).copy(); G2 = G2.subgraph(cc).copy()
    # Same relabelling map for both layers: original id -> rank among sorted ids.
    map_nodes = {v: i for i, v in enumerate(sorted(G1.nodes()))}
    G1 = nx.relabel_nodes(G1, map_nodes); G2 = nx.relabel_nodes(G2, map_nodes)
    return G1, G2


def collect_bilayers() -> list[tuple[str, nx.Graph, nx.Graph]]:
    """Build the list of (label, G1, G2) two-layer networks analysed.

    Order (12 entries when both empirical files are present): for each N in
    SIZES = [15, 30, 50, 75, 100] one ER pair, then one BA pair; followed by
    ("Empirical VC7", ...) and ("Empirical Law", ...) (LLF).  The label is
    written to the CSV "network" column.

    ER: layer 1 = r3_larger_networks_partA.get_network("ER", N, ER_P = 0.1,
    seed=10*N), layer 2 = get_network("ER", N, 0.1, seed=10*N + 7) (see
    get_network for the seeding; each layer has integer labels starting at
    0).  The two layers are aligned by INTEGER LABEL, keeping only the
    labels present in both, and each layer is relabelled 0..n-1 again.  The
    two layers are generated independently, so the layer-2 node placed
    under label v is unrelated to the layer-1 node v (an independent second
    layer).

    BA: layer 1 = ba_julia(N, BA_M = 2, seed=10*N + 1), layer 2 =
    ba_julia(N, 2, seed=10*N + 8) (preferential attachment from a star
    initial graph, see ba_julia.py).  Both have node set 0..N-1, so the
    intersection step keeps every node.  Layer 2's labels are then permuted
    uniformly at random with ba_julia.shuffle_layer2(seed=20260525 + 10*N)
    (the manuscript's Julia protocol for pairing two BA layers).  The
    returned permutation is not used because every initial condition places
    the cooperator and the mutant on the same LABEL v of the (already
    shuffled) layers.

    Returns:
        list of (label, G1, G2); G1 and G2 always share the node set
        0..n-1, which `fast_per_init_bilayer` relies on.
    """
    out = []
    for N in SIZES:
        G1 = get_network("ER", N, ER_P, seed=10 * N + 0)
        G2 = get_network("ER", N, ER_P, seed=10 * N + 7)
        # Align by integer label: keep the labels present in both layers.
        common = set(G1.nodes()) & set(G2.nodes())
        G1 = nx.convert_node_labels_to_integers(G1.subgraph(common).copy())
        G2 = nx.convert_node_labels_to_integers(G2.subgraph(common).copy())
        out.append((f"ER N={N} p={ER_P}", G1, G2))
        G1 = get_network("BA", N, BA_M, seed=10 * N + 1)
        G2 = get_network("BA", N, BA_M, seed=10 * N + 8)
        common = set(G1.nodes()) & set(G2.nodes())   # = all N nodes for BA
        G1 = nx.convert_node_labels_to_integers(G1.subgraph(common).copy())
        G2 = nx.convert_node_labels_to_integers(G2.subgraph(common).copy())
        # Julia-style layer-2 shuffle (random relabelling of layer 2) to break
        # the spurious correlation between a node's degrees in the two BA
        # layers.  Deterministic seed per N.
        from ba_julia import shuffle_layer2
        G2, _perm = shuffle_layer2(G2, seed=20260525 + 10 * N)   # _perm unused (init_M = init_C = label v)
        out.append((f"BA N={N} m={BA_M}", G1, G2))
    # Empirical networks are appended only if their edge lists were found.
    G1, G2 = load_VC7_bilayer()
    if G1 is not None: out.append(("Empirical VC7", G1, G2))
    G1, G2 = load_law_bilayer()
    if G1 is not None: out.append(("Empirical Law", G1, G2))
    return out


def fast_per_init_bilayer(G1: nx.Graph, G2: nx.Graph
                            ) -> tuple[np.ndarray, np.ndarray]:
    """For each init (init_C = init_M = v), compute (bc_single, |slope|).
    LU-factors the per-network linear systems once and back-subs per init.
    Assumes dB-dB rule (matches RULE constant).

    Args:
        G1, G2: layer-1 and layer-2 graphs on the SAME node set 0..N-1
                (node v of G1 and node v of G2 are one individual; see
                `collect_bilayers`).

    Returns:
        (bc_singles, abs_slopes): two length-N float arrays indexed by the
        initial node v (cooperator on v in layer 1, mutant on v in layer 2):
            bc_singles[v] = (b/c)*_single = -theta_2 / (theta_1 - theta_3)
            abs_slopes[v] = |phi_{2,0} / (theta_1 - theta_3)| = |d(b/c)*/dr| (c = 1)
        Both are NaN when |theta_1 - theta_3| <= 1e-12.  The division is
        performed whatever the sign of theta_1 - theta_3.

    Implementation (S1 Text, Text B; every N-by-N unknown matrix is stored
    as a length-N^2 vector with index (i, j) -> i*N + j, i.e. row-major
    `ravel()` / `reshape(N, N)`; equation numbers refer to S1 Text):
      beta_ii  : Eq. (S52), (I - P1) beta = N (xi1 - xi1_hat),
                 last row replaced by the normalisation
                 sum_i pi1_i beta_ii = 0 (Eq. (S37)).
      beta_ij  : Eq. (S54),
                 (I - 1/2 P1 (x) I - 1/2 I (x) P1) beta = (N/2)(xi1_i xi1_j - xi1_hat)
                 for i != j, with the N diagonal rows (i, i) replaced by the
                 identity so that beta_ii equals the solution of the first system.
      gamma_ij : Eq. (S56) (dB-dB),
                 (I - c1 P1 (x) P2 - c2 P1 (x) I - c2 I (x) P2) gamma
                     = a (xi1_i xi2_j - xi1_hat xi2_hat),
                 a = N^2/(2N-1), c1 = 1/(2N-1), c2 = (N-1)/(2N-1), last row
                 replaced by sum_i pi1_i gamma_ii = 0 (Eq. (S38)).
      theta_n  = sum_ij pi1_i (P1^n)_ij beta_ij,  n = 1, 2, 3   (Eq. (S47))
      phi_{2,0} = sum_ij pi1_i (P1^2)_ij gamma_ij              (Eq. (S48))
    Here P_k is the row-stochastic random-walk matrix of layer k
    (P[i,j] = A[i,j]/k_i) and pi_k = k_i / sum_l k_l its stationary
    distribution (= reproductive value under dB); xi_hat = pi . xi.
    The three coefficient matrices do not depend on v, so each is
    factorised once with SuperLU (scipy.sparse.linalg.splu) and only the
    right-hand sides change per init.  These are the same linear systems
    as larger_networks_solver._solve_beta_sparse /
    _solve_gamma_dBdB_sparse, which instead re-solve for every init.
    """
    # Nodes are sorted inside _transition_matrix_sparse, so matrix index == node label.
    P1, pi1 = _transition_matrix_sparse(G1)
    P2, pi2 = _transition_matrix_sparse(G2)
    N = P1.shape[0]
    nn = N * N

    # ---- Beta_i and Beta_ij systems on G1 (single-layer thetas) ----
    # (I - P1) beta_i = N (xi1 - xi1_hat); the last equation is replaced by
    # sum_i pi1_i beta_i = 0 (the rhs entry is zeroed per init below).
    A_mat = sp.eye(N, format="lil") - P1.tolil()
    A_mat[-1, :] = pi1
    A_lu = spla.splu(A_mat.tocsc())

    # Kronecker index convention for the vectorised (i, j) -> i*N + j unknowns:
    #   kron(P1, I_N)[(i,j),(k,l)] = P1[i,k] delta_jl  -> sum_k P1[i,k] beta[k,j]
    #   kron(I_N, P1)[(i,j),(k,l)] = delta_ik P1[j,l]  -> sum_l P1[j,l] beta[i,l]
    I_N = sp.eye(N, format="csr")
    M_ij = (sp.eye(nn, format="csr")
             - 0.5 * sp.kron(P1, I_N)
             - 0.5 * sp.kron(I_N, P1))
    M_ij = M_ij.tolil()
    # Diagonal unknowns (i, i) are fixed to beta_i: replace their rows by identity.
    for i in range(N):
        r = i * N + i
        M_ij.rows[r] = [r]
        M_ij.data[r] = [1.0]
    M_ij_lu = spla.splu(M_ij.tocsc())

    # ---- Gamma system (bilayer perturbation) for dB-dB ----
    # Coefficients of S1 Text Eq. (S56).
    a_const = (N * N) / (2 * N - 1.0)
    c1 = 1.0 / (2 * N - 1.0)
    c2 = (N - 1.0) / (2 * N - 1.0)
    # kron(P1, P2)[(i,j),(k1,k2)] = P1[i,k1] P2[j,k2] -> sum_{k1,k2} P1[i,k1] P2[j,k2] gamma[k1,k2]
    M_g = (sp.eye(nn, format="csr")
           - c1 * sp.kron(P1, P2)
           - c2 * sp.kron(P1, I_N)
           - c2 * sp.kron(I_N, P2))
    M_g = M_g.tolil()
    # Replace the last row (unknown (N-1, N-1)) by the normalisation
    # sum_i pi1_i gamma_ii = 0 (S1 Text Eq. (S38)); its rhs is zeroed per init.
    constraint_row = np.zeros(nn)
    for i in range(N):
        constraint_row[i * N + i] = pi1[i]
    M_g.rows[-1] = list(np.flatnonzero(constraint_row))
    M_g.data[-1] = [constraint_row[k] for k in M_g.rows[-1]]
    M_g_lu = spla.splu(M_g.tocsc())

    bc_singles = np.empty(N)
    abs_slopes = np.empty(N)
    for v in range(N):
        # One-hot initial conditions: cooperator on v (layer 1), mutant on v (layer 2).
        xi1 = np.zeros(N); xi1[v] = 1.0
        xi2 = np.zeros(N); xi2[v] = 1.0
        xi1_hat = pi1[v]   # = pi1 . xi1 (RV-weighted initial cooperator frequency)
        xi2_hat = pi2[v]   # = pi2 . xi2

        # ---- Beta_i ----
        rhs_i = N * (xi1 - xi1_hat); rhs_i[-1] = 0.0   # last entry: constraint row
        beta_i = A_lu.solve(rhs_i)
        # ---- Beta_ij ----
        # xi1_hat is subtracted from EVERY (i, j) entry (broadcast), as in
        # S1 Text Eq. (S54).
        src = (N / 2.0) * (np.outer(xi1, xi1) - xi1_hat)
        rhs = src.ravel()
        for i in range(N):
            rhs[i * N + i] = beta_i[i]   # diagonal rows were replaced by identity
        beta_full = M_ij_lu.solve(rhs).reshape(N, N)
        theta1, theta2, theta3 = _compute_thetas(P1, pi1, beta_full)

        # ---- Gamma ----
        src_g = a_const * (np.outer(xi1, xi2) - xi1_hat * xi2_hat)
        rhs_g = src_g.ravel()
        rhs_g[-1] = 0.0                  # last entry: constraint row
        gamma = M_g_lu.solve(rhs_g).reshape(N, N)
        phi01, phi20, phi21 = _compute_phis(P1, P2, pi1, gamma)   # only phi20 is used

        # (b/c)*_single = -theta2/(theta1-theta3); d(b/c)*/dr = phi20/(theta1-theta3) at c = 1.
        td = theta1 - theta3
        bc_s = (-theta2 / td) if abs(td) > 1e-12 else np.nan
        sl = (phi20 / td) if abs(td) > 1e-12 else np.nan
        bc_singles[v] = bc_s
        abs_slopes[v] = abs(sl) if np.isfinite(sl) else np.nan
    return bc_singles, abs_slopes


def per_bilayer_stats(G1: nx.Graph, G2: nx.Graph, label: str) -> dict:
    """One row of r3_part_B_per_network.csv for the two-layer network (G1, G2).

    Args:
        G1, G2: aligned layers on the node set 0..N-1 (see `collect_bilayers`).
        label:  row name ("network" column).

    Returns a dict with
        N, E1, E2             node count and edge counts of layers 1 and 2
        all_init_cooperative  True iff (b/c)*_single is finite and > 0 for
                              EVERY initial node v -- the SI criterion for a
                              "cooperative" network, i.e. inclusion in Table A;
                              False as soon as one init is <= 0 or NaN
        n_pos_bc_single       number of inits with (b/c)*_single > 0
        mean_abs_slope        mean over v of |d(b/c)*/dr| (NaNs ignored); this
                              is the per-network quantity correlated with the
                              network metrics
        median_abs_slope      median over v of |d(b/c)*/dr|
        mean_degree, mean_closeness, mean_betweenness, clustering
                              metrics of LAYER 1 ONLY (G2 enters only through
                              the gamma system): mean degree <k>, mean
                              closeness centrality (nx.closeness_centrality
                              with default settings), mean normalised
                              betweenness centrality, mean local clustering
                              coefficient (= nx.average_clustering).
    """
    nodes = sorted(G1.nodes())
    bc, sl = fast_per_init_bilayer(G1, G2)
    finite_bc = np.isfinite(bc)
    n_pos_bc = int(((bc > 0) & finite_bc).sum())
    # Cooperative <=> every one of the N inits is finite AND positive.
    all_coop = (n_pos_bc == int(finite_bc.sum())) and (int(finite_bc.sum()) == len(nodes))

    # Layer-1 network metrics (only means are taken, so node order is irrelevant).
    deg = np.array([d for _, d in G1.degree()], dtype=float)
    cl  = np.array(list(nx.closeness_centrality(G1).values()))
    bt  = np.array(list(nx.betweenness_centrality(G1).values()))
    lc  = np.array(list(nx.clustering(G1).values()))
    return {
        "network": label,
        "N": G1.number_of_nodes(),
        "E1": G1.number_of_edges(),
        "E2": G2.number_of_edges(),
        "all_init_cooperative": all_coop,
        "n_pos_bc_single": n_pos_bc,
        "mean_abs_slope": float(np.nanmean(sl)),
        "median_abs_slope": float(np.nanmedian(sl)),
        "mean_degree":      float(deg.mean()),
        "mean_closeness":   float(cl.mean()),
        "mean_betweenness": float(bt.mean()),
        "clustering":       float(lc.mean()),
    }


def main():
    """Run the 4-metric analysis and write both CSVs into the current directory.

    Steps: build the 12 two-layer networks (`collect_bilayers`); compute
    one `per_bilayer_stats` row each, printing progress and timing; save
    r3_part_B_per_network.csv; keep the rows with all_init_cooperative ==
    True (11 of 12 -- VC7 is dropped); if at least three remain, compute
    scipy.stats.pearsonr between mean |slope| and each of the four layer-1
    metrics across those networks and save r3_part_B_correlations.csv.
    """
    bilayers = collect_bilayers()
    print(f"# {len(bilayers)} bilayers collected", flush=True)
    rows = []
    for label, G1, G2 in bilayers:
        t0 = time.time()
        st = per_bilayer_stats(G1, G2, label)
        dt = time.time() - t0
        print(f"\n# {label}: N={G1.number_of_nodes()}, "
              f"E1={G1.number_of_edges()}, E2={G2.number_of_edges()}  ({dt:.1f}s)", flush=True)
        print(f"  mean |slope|={st['mean_abs_slope']:.4f}, "
              f"all_init_cooperative={st['all_init_cooperative']}", flush=True)
        rows.append(st)
    df = pd.DataFrame(rows)
    df.to_csv("r3_part_B_per_network.csv", index=False)
    print(f"\n# Saved r3_part_B_per_network.csv ({len(df)} bilayers)", flush=True)

    # Cross-network correlation uses only the cooperative two-layer networks
    # (each network is one sample; y = mean |d(b/c)*/dr|, x = layer-1 metric).
    coop = df[df["all_init_cooperative"]].copy()
    print(f"# {len(coop)} of {len(df)} are cooperative bilayers", flush=True)
    if len(coop) >= 3:   # guard: pearsonr needs at least 3 samples
        cor_rows = []
        y = coop["mean_abs_slope"].values
        for col in ["mean_degree", "mean_closeness", "mean_betweenness", "clustering"]:
            x = coop[col].values
            r, p = pearsonr(x, y)
            cor_rows.append({"property": col, "Pearson_R": r, "p_value": p,
                              "n": len(coop)})
        out = pd.DataFrame(cor_rows)
        out.to_csv("r3_part_B_correlations.csv", index=False)
        print("\n# Cross-network correlations (mean |slope| vs G1 property):")
        print(out.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    else:
        print(f"\n# Too few cooperative bilayers ({len(coop)}) to compute correlations.")


if __name__ == "__main__":
    main()
