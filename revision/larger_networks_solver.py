"""
Sparse solver for the weak-selection quantities of the two-layer model.

Model
-----
Layer 1 carries the donation game (cooperator or defector; benefit b, cost c).
Layer 2 carries constant selection (mutant with fitness r, resident with
fitness 1). The same individual occupies node i in both layers. The updating
rule is 'dB-dB' (dB in both layers) or 'dB-Bd' (dB in layer 1, Bd in layer 2).
The initial state has a single cooperator in layer 1 and a single mutant in
layer 2.

Given the two layers G1 and G2, the rule, the node of the initial cooperator
and the node of the initial mutant, the module computes

    theta_n    for n = 1, 2, 3                    (S1 Text, Eq (S47))
    phi_{n,m}  for (n,m) = (0,1), (2,0), (2,1)    (S1 Text, Eq (S48))

Under weak selection the cooperator is favored when

    c theta_2 + b (theta_1 - theta_3) - (r - 1) phi_{2,0} > 0

(S1 Text, Eq (S50); Eq (3) of the main text), which gives the thresholds

    (b/c)*_single = -theta_2 / (theta_1 - theta_3)
    (b/c)*_two    = (b/c)*_single + (r - 1) phi_{2,0} / (c (theta_1 - theta_3)).

Conventions
-----------
* Nodes are taken in sorted order: row, column and vector entry k refer to
  the k-th node. G1 and G2 must have the same node set.
* P[i, j] = w_ij / s_i with s_i = sum_j w_ij (edge attribute 'weight', 1 if
  absent). pi_i = s_i / sum_l s_l is the reproductive value under the dB rule.
* The pair quantities beta_ij and gamma_ij are solved as vectors of length
  N^2, with the entry for (i, j) at position i*N + j. For gamma, i is a
  layer-1 node and j is a layer-2 node.
* Every linear system is solved with scipy.sparse.linalg.spsolve. The pair
  systems have N^2 unknowns, so time and memory grow quickly with N.

Functions
---------
bc_star_two_layer          public entry point: theta, phi and both thresholds
                           for one two-layer network and initial condition
_transition_matrix_sparse  P and pi of one layer
_solve_beta_sparse         beta_ij of layer 1 (dB rule)
_solve_gamma_dBdB_sparse   gamma_ij under dB-dB
_solve_gamma_dBBd_sparse   gamma_ij under dB-Bd
_compute_thetas            theta_1, theta_2, theta_3
_compute_phis              phi_{0,1}, phi_{2,0}, phi_{2,1}

Used by
-------
bc_star_two_layer:
    r3_topology_correlation_n6_exhaustive.py  (Table A, six-node columns)
    r3_naoki_revisions.py                      (Fig A, Fig B, Text E.1)
    r3_strategy_coupling_ic.py                 (Fig C)
    fig4_style_random.py                       (Fig H, dB-dB and dB-Bd)
    point_b_mc_overlay.py                      (Fig I)
_transition_matrix_sparse and _compute_thetas:
    r3_larger_networks_partA.py (Table B) and r3_larger_networks_partB.py
    (its fast_per_init_bilayer, which r3_larger_networks_partB_extended.py
    calls for the "larger" column of Table A)
_compute_phis:
    r3_larger_networks_partB.py (fast_per_init_bilayer)

Running this file
-----------------
The module is a library; nothing reads files and nothing is written. Running
it directly,

    python larger_networks_solver.py

prints a quick check of bc_star_two_layer on the ring with N = 10 (dB-dB,
r = 2, cooperator on node 0, mutant on node 1; (b/c)*_single = 2.6667 and
(b/c)*_two = 2.5668) and then a timing demonstration on random graphs with
N = 50, 100 and 200. The N = 200 case takes more than 8 minutes on an Apple
M1 laptop. Neither output is used for any S1 Text item.

Dependencies: numpy, networkx, scipy.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, lgmres


def _transition_matrix_sparse(G: nx.Graph) -> tuple[sp.csr_matrix, np.ndarray]:
    """Random-walk matrix and dB reproductive values of one layer.

    Parameters
    ----------
    G : undirected networkx graph. Nodes are taken in sorted order, and the
        edge attribute 'weight' is used as w_ij (1 if absent).

    Returns
    -------
    P  : scipy.sparse CSR matrix, N x N, P[i, j] = w_ij / s_i with
         s_i = sum_j w_ij (the degree for an unweighted graph).
    pi : ndarray of length N, pi_i = s_i / sum_l s_l. This is the stationary
         distribution of P and the reproductive value of node i under the dB
         rule (defined in S1 Text, Text A, Eqs (S4) and (S5)).
    """
    nodes = sorted(G.nodes())
    N = len(nodes)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_safe = np.where(deg > 0, deg, 1.0)
    # P[i,j] = A[i,j] / deg[i]
    invdeg_diag = sp.diags(1.0 / deg_safe)
    P = invdeg_diag @ A
    pi = deg / deg.sum()
    return P.tocsr(), pi


def _solve_beta_sparse(P: sp.csr_matrix, pi: np.ndarray, xi1: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """beta_ij of layer 1 under the dB rule (S1 Text, Text B).

    Parameters
    ----------
    P   : CSR random-walk matrix of layer 1 (from _transition_matrix_sparse).
    pi  : dB reproductive values of layer 1.
    xi1 : length-N 0/1 vector, the initial state of layer 1 (1 = cooperator).

    The solution has two steps, with xi_hat = sum_i pi_i xi1_i.

    1. Diagonal, Eq (S52):
           beta_ii = N (xi1_i - xi_hat) + sum_k P[i,k] beta_kk.
       The equation of the last node is replaced by the normalization
       sum_i pi_i beta_ii = 0, Eq (S37).
    2. Off-diagonal, Eq (S54), for i != j:
           beta_ij = (N/2) (xi1_i xi1_j - xi_hat)
                     + (1/2) sum_k P[i,k] beta_kj + (1/2) sum_k P[j,k] beta_ik.
       All N^2 unknowns are solved together, with the N diagonal entries
       fixed to the values from step 1.

    Returns
    -------
    beta_i    : ndarray of length N, beta_ii from step 1.
    beta_full : dense N x N ndarray, beta_full[i, j] = beta_ij.
    """
    N = P.shape[0]
    xi_hat = pi @ xi1

    # ---- beta_i: (I - P) beta = N*(xi - xi_hat); replace last row with sum pi_i beta_i = 0
    A_mat = sp.eye(N, format="lil") - P.tolil()
    rhs = N * (xi1 - xi_hat)
    A_mat[-1, :] = pi
    rhs[-1] = 0.0
    beta_i = spsolve(A_mat.tocsr(), rhs)

    # ---- beta_ij (off-diagonal):
    # beta_ij - 0.5 sum_k P[i,k] beta_kj - 0.5 sum_k P[j,k] beta_ik = (N/2)(xi_i xi_j - xi_hat)
    # Variables index: (i,j) -> i*N + j. Diagonal entries beta_ii = beta_i (constraint).
    nn = N * N
    # Build (I - 0.5 (P kron I) - 0.5 (I kron P)). In row i*N + j,
    # kron(P, I) picks P[i,k] beta_kj and kron(I, P) picks P[j,k] beta_ik.
    I_N = sp.eye(N, format="csr")
    M = sp.eye(nn, format="csr") - 0.5 * sp.kron(P, I_N) - 0.5 * sp.kron(I_N, P)
    # Right-hand side
    rhs = np.zeros(nn)
    src = (N / 2.0) * (np.outer(xi1, xi1) - xi_hat)
    rhs[:] = src.ravel()

    # Constraint: beta_ii = beta_i. Replace those N rows by the identity row
    # (edited directly in the LIL row lists).
    M = M.tolil()
    for i in range(N):
        r = i * N + i
        M.rows[r] = [r]
        M.data[r] = [1.0]
        rhs[r] = beta_i[i]
    M = M.tocsr()

    x = spsolve(M, rhs)
    beta_full = x.reshape(N, N)
    return beta_i, beta_full


def _solve_gamma_dBdB_sparse(P1: sp.csr_matrix, P2: sp.csr_matrix,
                              pi1: np.ndarray, pi2: np.ndarray,
                              xi1: np.ndarray, xi2: np.ndarray) -> np.ndarray:
    """gamma_ij under the dB-dB rule (S1 Text, Text B, Eq (S56)).

    Parameters
    ----------
    P1, P2   : CSR random-walk matrices of layers 1 and 2.
    pi1, pi2 : dB reproductive values of layers 1 and 2.
    xi1      : length-N 0/1 vector, 1 on the node of the initial cooperator
               (layer 1).
    xi2      : length-N 0/1 vector, 1 on the node of the initial mutant
               (layer 2).

    Solves, for all (i, j), with i a layer-1 node and j a layer-2 node,

        gamma_ij = N^2/(2N-1) (xi1_i xi2_j - xi1_hat xi2_hat)
                   + 1/(2N-1)     sum_{k1,k2} P1[i,k1] P2[j,k2] gamma_{k1 k2}
                   + (N-1)/(2N-1) sum_k1 P1[i,k1] gamma_{k1 j}
                   + (N-1)/(2N-1) sum_k2 P2[j,k2] gamma_{i k2},

    where xi1_hat = pi1 . xi1 and xi2_hat = pi2 . xi2 (dB reproductive
    values in both layers). The equation for (i, j) = (last node, last node)
    is replaced by the normalization sum_i pi1_i gamma_ii = 0, Eq (S38).

    Returns
    -------
    gamma : dense N x N ndarray, gamma[i, j] = gamma_ij.
    """
    N = P1.shape[0]
    xi1_hat = pi1 @ xi1
    xi2_hat = pi2 @ xi2

    a = (N * N) / (2 * N - 1.0)
    c1 = 1.0 / (2 * N - 1.0)
    c2 = (N - 1.0) / (2 * N - 1.0)

    nn = N * N
    I_N = sp.eye(N, format="csr")
    # In row i*N + j: kron(P1, P2) has P1[i,k1] P2[j,k2] in column k1*N + k2,
    # kron(P1, I) has P1[i,k1] in column k1*N + j, and kron(I, P2) has
    # P2[j,k2] in column i*N + k2.
    M = (sp.eye(nn, format="csr")
         - c1 * sp.kron(P1, P2)
         - c2 * sp.kron(P1, I_N)
         - c2 * sp.kron(I_N, P2))

    src = a * (np.outer(xi1, xi2) - xi1_hat * xi2_hat)
    rhs = src.ravel()

    # Constraint: sum_i pi1_i gamma_ii = 0. Replace last row of M.
    M = M.tolil()
    constraint_row = np.zeros(nn)
    for i in range(N):
        constraint_row[i * N + i] = pi1[i]
    M.rows[-1] = list(np.flatnonzero(constraint_row))
    M.data[-1] = [constraint_row[k] for k in M.rows[-1]]
    rhs[-1] = 0.0
    M = M.tocsr()

    x = spsolve(M, rhs)
    gamma = x.reshape(N, N)
    return gamma


def _solve_gamma_dBBd_sparse(P1: sp.csr_matrix, P2: sp.csr_matrix,
                              pi1: np.ndarray, pi2_dB: np.ndarray,
                              xi1: np.ndarray, xi2: np.ndarray) -> np.ndarray:
    """gamma_ij under the dB-Bd rule (dB in layer 1, Bd in layer 2; S1 Text, Text C).

    Parameters
    ----------
    P1, P2   : CSR random-walk matrices of layers 1 and 2.
    pi1      : dB reproductive values of layer 1.
    pi2_dB   : dB reproductive values of layer 2, s_i / sum_l s_l. They are
               converted here into the Bd reproductive values of layer 2.
    xi1      : length-N 0/1 vector, 1 on the node of the initial cooperator
               (layer 1).
    xi2      : length-N 0/1 vector, 1 on the node of the initial mutant
               (layer 2).

    Solves, for all (i, j), with i a layer-1 node and j a layer-2 node,

        gamma_ij = xi1_i xi2_j - xi1_hat xi2_hat
                   + (1/N^2) sum_{k1,k2} P1[i,k1] P2[k2,j] gamma_{k1 k2}
                   + (1/N) (1 - t_j/N)     sum_k1 P1[i,k1] gamma_{k1 j}
                   + (1/N) (1 - 1/N)       sum_k2 P2[k2,j] gamma_{i k2}
                   + (1 - 1/N) (1 - t_j/N) gamma_ij,

    where

        t_j     = sum_k P2[k, j]  (the array s_col below),
        xi1_hat = pi1 . xi1       (dB reproductive values of layer 1),
        xi2_hat = pi2_Bd . xi2,   pi2_Bd_i = (1/s_i) / sum_l (1/s_l)
                  (Bd reproductive values of layer 2, S1 Text Eq (S65)),
                  obtained as pi2_Bd proportional to 1 / pi2_dB.

    The equation for (i, j) = (last node, last node) is replaced by the
    normalization sum_i pi1_i gamma_ii = 0, Eq (S38).

    Returns
    -------
    gamma : dense N x N ndarray, gamma[i, j] = gamma_ij.
    """
    N = P1.shape[0]
    xi1_hat = pi1 @ xi1

    # Bd reproductive values of layer 2: pi2_Bd_i proportional to 1/s_i,
    # i.e. to 1/pi2_dB_i.
    inv_pi2 = np.where(pi2_dB > 1e-15, 1.0 / pi2_dB, 0.0)
    pi2_Bd = inv_pi2 / inv_pi2.sum()
    xi2_hat = float(pi2_Bd @ xi2)

    nn = N * N
    I_N = sp.eye(N, format="csr")
    P2_T = P2.T

    s_col = np.asarray(P2.sum(axis=0)).ravel()        # t_j = sum_k P2[k, j] (column sums)
    one_minus_sj_over_N = (1.0 - s_col / N)
    inv_N = 1.0 / N

    # Coefficients of the three gamma terms on the right-hand side:
    #   layer-1 sum  sum_k1 P1[i,k1] gamma_{k1 j} :  (1/N)(1 - t_j/N)
    #   layer-2 sum  sum_k2 P2[k2,j] gamma_{i k2} :  (1/N)(1 - 1/N)
    #   gamma_ij itself                            :  (1 - 1/N)(1 - t_j/N)
    fac_L1 = inv_N * one_minus_sj_over_N             # shape (N,) for column j
    fac_L2_const = inv_N * (1.0 - inv_N)             # scalar
    fac_self = (1.0 - inv_N) * one_minus_sj_over_N   # shape (N,) for column j

    # j of the unknown stored at position i*N + j, used to expand the
    # j-dependent coefficients to length N^2.
    j_index = np.tile(np.arange(N), N)
    fac_L1_full = fac_L1[j_index]
    fac_self_full = fac_self[j_index]

    # In row i*N + j: kron(P1, P2.T) has P1[i,k1] P2[k2,j] in column k1*N + k2.
    M_tensor = sp.kron(P1, P2_T) * (1.0 / (N * N))
    M_L1_block = sp.kron(P1, I_N)
    M_L1 = sp.diags(fac_L1_full) @ M_L1_block        # j-dependent on the L1 term
    M_L2 = sp.kron(I_N, P2_T) * fac_L2_const         # constant on the L2 term
    M_self = sp.diags(fac_self_full)

    M = sp.eye(nn, format="csr") - (M_tensor + M_L1 + M_L2 + M_self)

    src = np.outer(xi1, xi2) - xi1_hat * xi2_hat
    rhs = src.ravel()

    # Constraint: sum_i pi1_i gamma_ii = 0. Replace last row of M.
    M = M.tolil()
    constraint_row = np.zeros(nn)
    for i in range(N):
        constraint_row[i * N + i] = pi1[i]
    M.rows[-1] = list(np.flatnonzero(constraint_row))
    M.data[-1] = [constraint_row[k] for k in M.rows[-1]]
    rhs[-1] = 0.0
    M = M.tocsr()

    x = spsolve(M, rhs)
    gamma = x.reshape(N, N)
    return gamma


def _compute_thetas(P1: sp.csr_matrix, pi1: np.ndarray,
                     beta_full: np.ndarray) -> tuple[float, float, float]:
    """theta_n = sum_{i,j} pi1_i (P1^n)[i,j] beta_ij for n = 1, 2, 3 (S1 Text, Eq (S47)).

    Parameters
    ----------
    P1        : CSR random-walk matrix of layer 1.
    pi1       : dB reproductive values of layer 1.
    beta_full : dense N x N array of beta_ij (from _solve_beta_sparse).

    (P1^n)[i,j] is the probability that an n-step random walk in layer 1
    goes from i to j. The matrix powers are formed as dense arrays.

    Returns
    -------
    (theta1, theta2, theta3) as floats.
    """
    M1 = P1.toarray()
    M2 = M1 @ M1
    M3 = M2 @ M1
    theta1 = float(np.sum(pi1[:, None] * M1 * beta_full))
    theta2 = float(np.sum(pi1[:, None] * M2 * beta_full))
    theta3 = float(np.sum(pi1[:, None] * M3 * beta_full))
    return theta1, theta2, theta3


def _compute_phis(P1: sp.csr_matrix, P2: sp.csr_matrix, pi1: np.ndarray,
                   gamma: np.ndarray) -> tuple[float, float, float]:
    """phi_{n,m} = sum_{i,j} pi1_i (P1^n P2^m)[i,j] gamma_ij (S1 Text, Eq (S48)).

    Parameters
    ----------
    P1, P2 : CSR random-walk matrices of layers 1 and 2.
    pi1    : dB reproductive values of layer 1.
    gamma  : dense N x N array of gamma_ij, i a layer-1 node and j a layer-2
             node (from _solve_gamma_dBdB_sparse or _solve_gamma_dBBd_sparse).

    (P1^n P2^m)[i,j] is the probability that a random walk goes from i to j
    by n steps in layer 1 followed by m steps in layer 2. The matrix products
    are formed as dense arrays.

    Returns
    -------
    (phi01, phi20, phi21) as floats, for (n,m) = (0,1), (2,0), (2,1).
    phi20 enters the cooperation condition. phi01 and phi21 are used by
    fig4_style_random.py (Fig H) in its condition for the mutant to be
    favored.
    """
    M1 = P1.toarray()
    M2 = M1 @ M1
    P2m = P2.toarray()
    phi01 = float(np.sum(pi1[:, None] * P2m * gamma))
    phi20 = float(np.sum(pi1[:, None] * M2 * gamma))
    phi21 = float(np.sum(pi1[:, None] * (M2 @ P2m) * gamma))
    return phi01, phi20, phi21


def bc_star_two_layer(G1: nx.Graph, G2: nx.Graph,
                       init_C: int, init_M: int,
                       rule: str = "dB-dB",
                       r: float = 1.0,
                       c: float = 1.0) -> dict:
    """theta, phi and (b/c)* for one two-layer network and one initial condition.

    Parameters
    ----------
    G1     : networkx graph of layer 1 (donation game).
    G2     : networkx graph of layer 2 (constant selection), with the same
             node set as G1. Nodes are taken in sorted order.
    init_C : position, in sorted node order, of the node of the single
             initial cooperator in layer 1. For graphs with nodes 0..N-1
             this is the node label.
    init_M : position of the node of the single initial mutant in layer 2.
    rule   : 'dB-dB' or 'dB-Bd' (dB in layer 1, Bd in layer 2). Only gamma,
             and therefore phi, depends on the rule; theta is the same for
             both rules.
    r      : mutant fitness in layer 2 (resident fitness 1). Enters only
             bc_star_two.
    c      : cost of cooperation. Enters only bc_star_two.

    Returns
    -------
    dict with keys
        N               number of nodes
        theta1, theta2, theta3
        phi01, phi20, phi21
        bc_star_single  -theta2 / (theta1 - theta3)
        bc_star_two     bc_star_single + (r - 1) phi20 / (c (theta1 - theta3))
                        (bc_star_single and bc_star_two are inf if
                        |theta1 - theta3| < 1e-14)
        pi_init         pi1[init_C], the layer-1 dB reproductive value of the
                        cooperator's node, which is the cooperator's fixation
                        probability at selection strength delta = 0 (used by
                        point_b_mc_overlay.py)
        denom_sign      'positive' if theta1 - theta3 > 0, else 'negative'

    Cooperation is favored for b/c above (b/c)* when theta1 - theta3 > 0 and
    for b/c below (b/c)* when theta1 - theta3 < 0.
    """
    P1, pi1 = _transition_matrix_sparse(G1)
    P2, pi2 = _transition_matrix_sparse(G2)
    N = P1.shape[0]

    # Initial states: single cooperator in layer 1, single mutant in layer 2.
    xi1 = np.zeros(N); xi1[init_C] = 1.0
    xi2 = np.zeros(N); xi2[init_M] = 1.0

    # beta (hence theta) depends on layer 1 only; gamma depends on both layers
    # and on the rule.
    beta_i, beta_full = _solve_beta_sparse(P1, pi1, xi1)
    if rule == "dB-dB":
        gamma = _solve_gamma_dBdB_sparse(P1, P2, pi1, pi2, xi1, xi2)
    elif rule == "dB-Bd":
        gamma = _solve_gamma_dBBd_sparse(P1, P2, pi1, pi2, xi1, xi2)
    else:
        raise ValueError(rule)

    theta1, theta2, theta3 = _compute_thetas(P1, pi1, beta_full)
    phi01, phi20, phi21 = _compute_phis(P1, P2, pi1, gamma)

    denom = theta1 - theta3
    if abs(denom) < 1e-14:
        bc_single = float("inf")
        bc_two = float("inf")
    else:
        bc_single = -theta2 / denom
        bc_two = -theta2 / denom + (r - 1.0) * phi20 / (c * denom)

    return {
        "N": N,
        "theta1": theta1, "theta2": theta2, "theta3": theta3,
        "phi01": phi01, "phi20": phi20, "phi21": phi21,
        "bc_star_single": bc_single, "bc_star_two": bc_two,
        "pi_init": pi1[init_C],
        "denom_sign": "positive" if denom > 0 else "negative",
    }


if __name__ == "__main__":
    # Quick check on the ring with N = 10, then a timing demonstration on
    # random graphs. Not used for any S1 Text item.
    import time
    print("=== Sanity: ring N=10 ===")
    G = nx.cycle_graph(10)
    t0 = time.time()
    res = bc_star_two_layer(G, G, 0, 1, "dB-dB", r=2.0)
    print(f"  (b/c)*_single = {res['bc_star_single']:.4f} (expected 8/3 = 2.6667)")
    print(f"  (b/c)*_two    = {res['bc_star_two']:.4f} (expected 2.57)")
    print(f"  time = {time.time()-t0:.2f}s")

    print("\n=== Scaling check: ER N=50, 100, 200 ===")
    rng = np.random.default_rng(0)
    for N in [50, 100, 200]:
        G = nx.erdos_renyi_graph(N, p=4.0/(N-1), seed=0)  # avg degree ~4
        t0 = time.time()
        res = bc_star_two_layer(G, G, 0, 1, "dB-dB", r=2.0)
        print(f"  N={N}: (b/c)*_single = {res['bc_star_single']:.4f}, "
              f"(b/c)*_two = {res['bc_star_two']:.4f}, "
              f"time = {time.time()-t0:.2f}s")
