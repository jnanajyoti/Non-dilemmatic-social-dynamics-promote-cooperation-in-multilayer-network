"""
Dense NumPy solver for the weak-selection quantities of the two-layer model,
and a pure-NumPy reference simulator of the same model.

Model
-----
Layer 1 carries the donation game (x1_i = 1 cooperator, 0 defector; benefit
b, cost c) with the pf goods scheme of the main text,
    u1_i = -c x1_i + b sum_j (w_ij / s_i) x1_j,   s_i = sum_j w_ij.
Layer 2 carries constant selection (x2_i = 1 mutant, 0 resident),
    u2_i = x2_i (r - 1) + 1.
The same individual occupies node i in both layers. The updating rule is
'dB-dB' (dB in both layers) or 'dB-Bd' (dB in layer 1, Bd in layer 2). The
initial state has one cooperator in layer 1, on node initial_C_node, and
one mutant in layer 2, on node initial_M_node; the two may be the same node
or different nodes.

Quantities (S1 Text, Texts A to C)
----------------------------------
    theta_n    = sum_{i,j} pi1_i (P1^n)[i,j] beta_ij         n = 1, 2, 3, Eq (S47)
    phi_{n,m}  = sum_{i,j} pi1_i (P1^n P2^m)[i,j] gamma_ij   Eq (S48)
Cooperation is favored under weak selection when
    c theta_2 + b (theta_1 - theta_3) - (r - 1) phi_{2,0} > 0
(Eq (3) of the main text, Eq (S50) of S1 Text), i.e., if
theta_1 - theta_3 > 0, when b/c exceeds
    (b/c)* = -theta_2 / (theta_1 - theta_3) + (r - 1) phi_{2,0} / (c (theta_1 - theta_3))
(Eq (5) of the main text). (b/c)*_single is the value at r = 1.

Conventions
-----------
* Nodes are taken in sorted order: row, column and vector entry k refer to
  the k-th node. G1 and G2 must have the same node set.
* P[i, j] = w_ij / s_i (edge attribute 'weight', 1 if absent) and
  pi_i = s_i / sum_l s_l, the reproductive value under the dB rule.
* The pair quantities beta_ij and gamma_ij are returned as N x N arrays and
  solved as dense linear systems with N^2 unknowns, the unknown (i, j) at
  position i*N + j. For gamma, i is a layer-1 node and j a layer-2 node.
  The matrices have N^4 entries and are filled by Python loops, so this
  module is meant for small networks (Fig G uses N = 10 and N = 15);
  larger_networks_solver.py is the sparse counterpart, used for example by
  point_b_mc_overlay.py and fig4_style_random.py.

Use in this folder
------------------
S1 Text item: Fig G (Text L). run_ws_validation_ring_baba.py calls
    analytical_summary, drho_C_ddelta_singlelayer, drho_C_ddelta_twolayer.
analytical_summary calls
    transition_matrix, solve_beta, compute_thetas, compute_phis,
    compute_bc_star, and solve_gamma_dBdB_v2 (rule 'dB-dB') or
    solve_gamma_dBBd_corrected (rule 'dB-Bd').
Not called by any script in this folder:
    transition_matrix_Bd, solve_gamma_dBdB (always raises
    NotImplementedError), solve_gamma_dBBd, solve_eta,
    _build_neighbor_arrays, _payoff_layer1_pf, simulate_multilayer,
    network_to_arrays. simulate_multilayer is the pure-NumPy reference for
    the numba simulator in weak_selection_sim_fast.py.

Running this file
-----------------
    python weak_selection_validation.py
(from the revision folder) prints analytical_summary for the ring with
N = 10 in both layers, dB-dB, r = 2: first with the cooperator on node 0 and
the mutant on node 1 ((b/c)*_single = 2.6667, (b/c)*_two = 2.5668), then
with both on node 0 ((b/c)*_two = 2.3912). Inputs: none. Output: stdout
only. Runtime: about 1 s. This check is not an S1 Text item.

Dependencies: numpy, networkx, scipy (imported; the solves use numpy.linalg).
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.linalg as sla


# ----------------------------------------------------------------------------
# 1. Random-walk transition matrices
# ----------------------------------------------------------------------------

def transition_matrix(G: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Random-walk matrix and dB reproductive values of one layer.

    Parameters
    ----------
    G : undirected networkx graph. Nodes are taken in sorted order; the edge
        attribute 'weight' is used as w_ij (1 if absent).

    Returns
    -------
    P  : dense N x N array, P[i, j] = w_ij / s_i with s_i = sum_j w_ij (the
         degree for an unweighted graph). This is the one-step random-walk
         matrix of the layer, and also the weight b P[i, j] with which
         neighbour j's cooperation enters i's payoff in the pf goods scheme.
    pi : length-N array, pi_i = s_i / sum_l s_l, the stationary distribution
         of P and the reproductive value of node i under the dB rule.
    """
    N = G.number_of_nodes()
    nodes = sorted(G.nodes())
    A = np.zeros((N, N))
    for i, u in enumerate(nodes):
        for v, data in G[u].items():
            j = nodes.index(v)
            A[i, j] = data.get("weight", 1.0)
    deg = A.sum(axis=1)
    P = A / deg[:, None]
    pi = deg / deg.sum()
    return P, pi


def transition_matrix_Bd(G: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Random-walk matrix and Bd reproductive values of one layer.

    Under the Bd rule the parent k is chosen with probability 1/N at
    delta = 0 and places its offspring on neighbour j with probability
    P[k, j] = w_kj / s_k.

    Returns
    -------
    P     : dense N x N array, P[i, j] = w_ij / s_i (as in transition_matrix).
    pi_Bd : length-N array, pi_Bd_i = (1/s_i) / sum_l (1/s_l), the
            reproductive value under the Bd rule (S1 Text, Eq (S65); Sood,
            Antal and Redner, Phys Rev E 77, 041121 (2008)).

    Not called by any script in this folder (analytical_summary computes the
    same pi_Bd inline).
    """
    N = G.number_of_nodes()
    nodes = sorted(G.nodes())
    A = np.zeros((N, N))
    for i, u in enumerate(nodes):
        for v, data in G[u].items():
            j = nodes.index(v)
            A[i, j] = data.get("weight", 1.0)
    deg = A.sum(axis=1)
    P = A / deg[:, None]
    inv_deg = 1.0 / deg
    pi_Bd = inv_deg / inv_deg.sum()
    return P, pi_Bd


# ----------------------------------------------------------------------------
# 2. Recurrences for beta_ij (layer 1, dB), gamma_ij (dB-dB and dB-Bd), eta_i
# ----------------------------------------------------------------------------

def solve_beta(P: np.ndarray, pi: np.ndarray, xi1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """beta_ij of layer 1 under the dB rule (S1 Text, Text B).

    Parameters
    ----------
    P   : N x N random-walk matrix of layer 1 (from transition_matrix).
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
       set to the values from step 1.

    Returns
    -------
    beta_i    : length-N array, beta_ii from step 1.
    beta_full : N x N array, beta_full[i, j] = beta_ij (diagonal = beta_i).
    """
    N = P.shape[0]
    xi_hat = pi @ xi1

    # ---- step 1: diagonal beta_ii ----
    # (I - P) beta = N (xi1 - xi_hat); the last equation is replaced by the
    # normalization sum_i pi_i beta_ii = 0 (Eq (S37)).
    I = np.eye(N)
    A = I - P
    b = N * (xi1 - xi_hat)
    # Replace last equation with the normalization.
    A[-1, :] = pi
    b[-1] = 0.0
    beta_i = np.linalg.solve(A, b)

    # ---- step 2: all beta_ij ----
    # Unknowns beta[i, j] for all (i, j), stored at position i*N + j.
    # Rows for i != j:
    #   beta_ij - (1/2) sum_k P[i,k] beta_kj - (1/2) sum_k P[j,k] beta_ik
    #     = (N/2) (xi1_i xi1_j - xi_hat).
    # Rows for i == j are overwritten below with beta_ii = beta_i.
    #
    # Build the N^2 x N^2 system in row-major (i, j) order.
    nn = N * N
    M = np.zeros((nn, nn))
    rhs = np.zeros(nn)
    for i in range(N):
        for j in range(N):
            r = i * N + j
            M[r, r] = 1.0
            for k in range(N):
                M[r, k * N + j] -= 0.5 * P[i, k]
                M[r, i * N + k] -= 0.5 * P[j, k]
            rhs[r] = (N / 2.0) * (xi1[i] * xi1[j] - xi_hat)
    # Pin diagonal entries beta_ii = beta_i (override these N rows).
    for i in range(N):
        r = i * N + i
        M[r, :] = 0.0
        M[r, r] = 1.0
        rhs[r] = beta_i[i]
    beta_full = np.linalg.solve(M, rhs).reshape(N, N)
    return beta_i, beta_full


def solve_gamma_dBdB(P1: np.ndarray, P2: np.ndarray, pi1: np.ndarray,
                      xi1: np.ndarray, xi2: np.ndarray) -> np.ndarray:
    """Placeholder for gamma_ij under dB-dB; always raises NotImplementedError.

    The recurrence it refers to is

        gamma_ij = (N^2/(2N-1)) [xi1_i xi2_j - xi1_hat xi2_hat]
                 + (1/(2N-1)) sum_{k1,k2} P1[i,k1] P2[j,k2] gamma_{k1,k2}
                 + ((N-1)/(2N-1)) sum_k P1[i,k] gamma_{k,j}
                 + ((N-1)/(2N-1)) sum_k P2[j,k] gamma_{i,k},

    with sum_i pi1_i gamma_ii = 0. xi2_hat needs the reproductive values pi2
    of layer 2, which are not among the arguments, so the function raises
    NotImplementedError. solve_gamma_dBdB_v2 takes pi2 and solves this
    recurrence. Not called by any script in this folder.
    """
    # The assignments below are never used: the function raises before
    # returning.
    N = P1.shape[0]
    xi1_hat = pi1 @ xi1
    pi2_norm = np.ones(N) / N
    xi2_hat = xi2.mean() if False else (np.ones(N) / N) @ xi2
    raise NotImplementedError("Use solve_gamma_dBdB_v2 with explicit pi2")


def solve_gamma_dBdB_v2(P1: np.ndarray, P2: np.ndarray,
                         pi1: np.ndarray, pi2: np.ndarray,
                         xi1: np.ndarray, xi2: np.ndarray) -> np.ndarray:
    """gamma_ij under the dB-dB rule (S1 Text, Text B, Eq (S56)).

    Parameters
    ----------
    P1, P2   : N x N random-walk matrices of layers 1 and 2.
    pi1, pi2 : dB reproductive values of layers 1 and 2.
    xi1, xi2 : length-N 0/1 initial states (xi1: cooperator in layer 1,
               xi2: mutant in layer 2).

    Solves, for all (i, j), with xi1_hat = sum_i pi1_i xi1_i and
    xi2_hat = sum_j pi2_j xi2_j,

        gamma_ij = (N^2/(2N-1)) [xi1_i xi2_j - xi1_hat xi2_hat]
                 + (1/(2N-1)) sum_{k1,k2} P1[i,k1] P2[j,k2] gamma_{k1,k2}
                 + ((N-1)/(2N-1)) sum_k P1[i,k] gamma_{k,j}
                 + ((N-1)/(2N-1)) sum_k P2[j,k] gamma_{i,k},

    as one system with N^2 unknowns (gamma_ij at position i*N + j). The
    equation of (i, j) = (N-1, N-1) is replaced by the normalization
    sum_i pi1_i gamma_ii = 0, Eq (S38).

    Returns
    -------
    gamma : N x N array, gamma[i, j] with i a layer-1 node and j a layer-2
            node.
    """
    N = P1.shape[0]
    xi1_hat = pi1 @ xi1
    xi2_hat = pi2 @ xi2

    a = (N * N) / (2 * N - 1.0)
    c1 = 1.0 / (2 * N - 1.0)
    c2 = (N - 1.0) / (2 * N - 1.0)

    nn = N * N
    M = np.zeros((nn, nn))
    rhs = np.zeros(nn)
    for i in range(N):
        for j in range(N):
            r = i * N + j                     # row of the unknown gamma_ij
            M[r, r] = 1.0
            # term c1 * sum_{k1,k2} P1[i,k1] P2[j,k2] gamma_{k1,k2}
            for k1 in range(N):
                for k2 in range(N):
                    M[r, k1 * N + k2] -= c1 * P1[i, k1] * P2[j, k2]
            # term c2 * sum_k P1[i,k] gamma_{k,j}
            for k in range(N):
                M[r, k * N + j] -= c2 * P1[i, k]
            # term c2 * sum_k P2[j,k] gamma_{i,k}
            for k in range(N):
                M[r, i * N + k] -= c2 * P2[j, k]
            rhs[r] = a * (xi1[i] * xi2[j] - xi1_hat * xi2_hat)

    # Replace the last row, (i, j) = (N-1, N-1), with the normalization
    # sum_i pi1_i gamma_ii = 0.
    constraint_row = np.zeros(nn)
    for i in range(N):
        constraint_row[i * N + i] = pi1[i]
    M[-1, :] = constraint_row
    rhs[-1] = 0.0
    gamma = np.linalg.solve(M, rhs).reshape(N, N)
    return gamma


def solve_gamma_dBBd(P1: np.ndarray, P2: np.ndarray,
                      pi1: np.ndarray, pi2_Bd: np.ndarray,
                      xi1: np.ndarray, xi2: np.ndarray) -> np.ndarray:
    """gamma_ij under the dB-Bd rule, with the coefficients listed below.

    Parameters
    ----------
    P1, P2   : N x N random-walk matrices of layers 1 and 2.
    pi1      : dB reproductive values of layer 1.
    pi2_Bd   : Bd reproductive values of layer 2, (1/s_j) / sum_l (1/s_l)
               with s_j the degree of node j in layer 2.
    xi1, xi2 : length-N 0/1 initial states of layers 1 and 2.

    With q_j = sum_k P2[k, j] (the column sums of P2; variable sj below),
    xi1_hat = sum_i pi1_i xi1_i and xi2_hat = sum_j pi2_Bd_j xi2_j, solves
    for all (i, j)

        gamma_ij = xi1_i xi2_j - xi1_hat xi2_hat
                 + (1/N^2)           sum_{k1,k2} P1[i,k1] P2[k2,j] gamma_{k1,k2}
                 + (1/N)(1 - 1/N)    sum_{k1} P1[i,k1] gamma_{k1,j}
                 + (1/N)(1 - q_j/N)  sum_{k2} P2[k2,j] gamma_{i,k2}
                 + (1 - 1/N)(1 - q_j/N) gamma_ij,

    as one system with N^2 unknowns (gamma_ij at position i*N + j), the
    equation of (i, j) = (N-1, N-1) replaced by sum_i pi1_i gamma_ii = 0.
    In layer 2 the sums run over the parent k2 of node j, with weight
    P2[k2, j] = w_{k2 j} / s_{k2} (column j of P2).

    Returns
    -------
    gamma : N x N array, gamma[i, j] with i a layer-1 node and j a layer-2
            node.

    Not called by any script in this folder; analytical_summary uses
    solve_gamma_dBBd_corrected for the rule 'dB-Bd'.
    """
    N = P1.shape[0]
    xi1_hat = pi1 @ xi1
    xi2_hat = pi2_Bd @ xi2

    nn = N * N
    M = np.zeros((nn, nn))
    rhs = np.zeros(nn)
    # s_col[j] = q_j = sum_{k2} P2[k2, j]
    s_col = P2.sum(axis=0)
    for i in range(N):
        for j in range(N):
            r = i * N + j                     # row of the unknown gamma_ij
            sj = s_col[j]
            # 1 minus the coefficient (1 - 1/N)(1 - q_j/N) of gamma_ij
            self_coeff = 1.0 - (1.0 - 1.0 / N) * (1.0 - sj / N)
            M[r, r] = self_coeff
            # term (1/N^2) sum_{k1,k2} P1[i,k1] P2[k2,j] gamma_{k1,k2}
            for k1 in range(N):
                for k2 in range(N):
                    M[r, k1 * N + k2] -= (1.0 / (N * N)) * P1[i, k1] * P2[k2, j]
            # term (1/N)(1 - 1/N) sum_{k1} P1[i,k1] gamma_{k1,j}
            coef1 = (1.0 / N) * (1.0 - 1.0 / N)
            for k1 in range(N):
                M[r, k1 * N + j] -= coef1 * P1[i, k1]
            # term (1/N)(1 - q_j/N) sum_{k2} P2[k2,j] gamma_{i,k2}
            coef2_factor = (1.0 / N) * (1.0 - sj / N)
            for k2 in range(N):
                M[r, i * N + k2] -= coef2_factor * P2[k2, j]
            rhs[r] = xi1[i] * xi2[j] - xi1_hat * xi2_hat
    # Replace the last row with the normalization sum_i pi1_i gamma_ii = 0.
    constraint_row = np.zeros(nn)
    for i in range(N):
        constraint_row[i * N + i] = pi1[i]
    M[-1, :] = constraint_row
    rhs[-1] = 0.0
    gamma = np.linalg.solve(M, rhs).reshape(N, N)
    return gamma


def solve_gamma_dBBd_corrected(P1: np.ndarray, P2: np.ndarray,
                                pi1: np.ndarray, pi2_Bd: np.ndarray,
                                xi1: np.ndarray, xi2: np.ndarray) -> np.ndarray:
    """gamma_ij under the dB-Bd rule, with the coefficients listed below.

    analytical_summary calls this function for the rule 'dB-Bd'.

    Parameters
    ----------
    P1, P2   : N x N random-walk matrices of layers 1 and 2.
    pi1      : dB reproductive values of layer 1.
    pi2_Bd   : Bd reproductive values of layer 2, (1/s_j) / sum_l (1/s_l)
               with s_j the degree of node j in layer 2.
    xi1, xi2 : length-N 0/1 initial states of layers 1 and 2.

    With q_j = sum_k P2[k, j] (the column sums of P2; variable sj below),
    xi1_hat = sum_i pi1_i xi1_i and xi2_hat = sum_j pi2_Bd_j xi2_j, solves
    for all (i, j)

        gamma_ij = xi1_i xi2_j - xi1_hat xi2_hat
                 + (1/N^2)           sum_{k1,k2} P1[i,k1] P2[k2,j] gamma_{k1,k2}
                 + (1/N)(1 - q_j/N)  sum_{k1} P1[i,k1] gamma_{k1,j}
                 + (1/N)(1 - 1/N)    sum_{k2} P2[k2,j] gamma_{i,k2}
                 + (1 - 1/N)(1 - q_j/N) gamma_ij,

    as one system with N^2 unknowns (gamma_ij at position i*N + j), the
    equation of (i, j) = (N-1, N-1) replaced by sum_i pi1_i gamma_ii = 0.
    In layer 2 the sums run over the parent k2 of node j, with weight
    P2[k2, j] = w_{k2 j} / s_{k2} (column j of P2).

    Returns
    -------
    gamma : N x N array, gamma[i, j] with i a layer-1 node and j a layer-2
            node.
    """
    N = P1.shape[0]
    xi1_hat = pi1 @ xi1
    xi2_hat = pi2_Bd @ xi2
    nn = N * N
    M = np.zeros((nn, nn))
    rhs = np.zeros(nn)
    s_col = P2.sum(axis=0)                      # s_col[j] = q_j = sum_{k} P2[k, j]
    for i in range(N):
        for j in range(N):
            r = i * N + j                       # row of the unknown gamma_ij
            sj = s_col[j]
            # 1 minus the coefficient (1 - 1/N)(1 - q_j/N) of gamma_ij
            M[r, r] = 1.0 - (1.0 - 1.0 / N) * (1.0 - sj / N)
            # term (1/N^2) sum_{k1,k2} P1[i,k1] P2[k2,j] gamma_{k1,k2}
            for k1 in range(N):
                for k2 in range(N):
                    M[r, k1 * N + k2] -= (1.0 / (N * N)) * P1[i, k1] * P2[k2, j]
            # term (1/N)(1 - q_j/N) sum_{k1} P1[i,k1] gamma_{k1,j}
            coef1 = (1.0 / N) * (1.0 - sj / N)
            for k1 in range(N):
                M[r, k1 * N + j] -= coef1 * P1[i, k1]
            # term (1/N)(1 - 1/N) sum_{k2} P2[k2,j] gamma_{i,k2}
            coef2_factor = (1.0 / N) * (1.0 - 1.0 / N)
            for k2 in range(N):
                M[r, i * N + k2] -= coef2_factor * P2[k2, j]
            rhs[r] = xi1[i] * xi2[j] - xi1_hat * xi2_hat
    # Replace the last row with the normalization sum_i pi1_i gamma_ii = 0.
    constraint_row = np.zeros(nn)
    for i in range(N):
        constraint_row[i * N + i] = pi1[i]
    M[-1, :] = constraint_row
    rhs[-1] = 0.0
    return np.linalg.solve(M, rhs).reshape(N, N)


def solve_eta(P: np.ndarray, xi1: np.ndarray, xi1_hat: float) -> np.ndarray:
    """Solve eta_i = xi1_i - xi1_hat + sum_k P[i,k] eta_k.

    Parameters
    ----------
    P       : N x N random-walk matrix.
    xi1     : length-N initial state.
    xi1_hat : scalar subtracted from every entry of xi1.

    The linear system (I - P) eta = xi1 - xi1_hat is solved with its last
    equation replaced by sum_i eta_i = 0. Returns eta as a length-N array.
    Not called by any script in this folder (eta does not enter theta_n or
    phi_{n,m}).
    """
    N = P.shape[0]
    A = np.eye(N) - P
    b = xi1 - xi1_hat
    A[-1, :] = 1.0
    b[-1] = 0.0
    return np.linalg.solve(A, b)


# ----------------------------------------------------------------------------
# 3. Compute theta_n, phi_{n,m} and (b/c)*
# ----------------------------------------------------------------------------

def compute_thetas(P1: np.ndarray, pi1: np.ndarray, beta_full: np.ndarray) -> tuple[float, float, float]:
    """theta_1, theta_2, theta_3 of layer 1 (S1 Text, Eq (S47)).

    theta_n = sum_{i,j} pi1_i (P1^n)[i,j] beta_ij, where P1^n is the n-step
    random-walk matrix of layer 1 and beta_full is the N x N array returned
    by solve_beta. Returns (theta_1, theta_2, theta_3) as floats.
    """
    M1 = P1
    M2 = M1 @ M1
    M3 = M2 @ M1
    theta1 = float(np.sum(pi1[:, None] * M1 * beta_full))
    theta2 = float(np.sum(pi1[:, None] * M2 * beta_full))
    theta3 = float(np.sum(pi1[:, None] * M3 * beta_full))
    return theta1, theta2, theta3


def compute_phis(P1: np.ndarray, P2: np.ndarray, pi1: np.ndarray,
                  gamma: np.ndarray) -> tuple[float, float, float]:
    """phi_{0,1}, phi_{2,0}, phi_{2,1} (S1 Text, Eq (S48)).

    phi_{n,m} = sum_{i,j} pi1_i (P1^n P2^m)[i,j] gamma[i,j]: a random walk of
    n steps in layer 1 followed by m steps in layer 2, from node i to node
    j, with gamma the N x N array from the gamma solver of the rule.

    Returns
    -------
    (phi_{0,1}, phi_{2,0}, phi_{2,1}) as floats. Only phi_{2,0} enters the
    condition for cooperation.
    """
    M1 = P1
    M2 = M1 @ M1
    P2m = P2

    phi01 = float(np.sum(pi1[:, None] * P2m * gamma))
    phi20 = float(np.sum(pi1[:, None] * M2 * gamma))
    phi21 = float(np.sum(pi1[:, None] * (M2 @ P2m) * gamma))
    return phi01, phi20, phi21


def compute_bc_star(theta1, theta2, theta3, phi20, r, c=1.0):
    """Threshold (b/c)* of Eq (5) of the main text.

        (b/c)* = -theta2 / (theta1 - theta3) + (r - 1) phi20 / (c (theta1 - theta3)).

    With r = 1 (or phi20 = 0) this is (b/c)*_single. Returns float('inf')
    when |theta1 - theta3| < 1e-14.
    """
    denom = theta1 - theta3
    if abs(denom) < 1e-14:
        return float("inf")
    return (-theta2 / denom) + ((r - 1.0) * phi20) / (c * denom)


def drho_C_ddelta_singlelayer(b, c, theta1, theta2, theta3, N):
    """d rho_C / d delta at delta = 0 for layer 1 on its own (dB rule).

        (1/N) [c theta2 + b (theta1 - theta3)]

    This is the two-layer expression of drho_C_ddelta_twolayer at r = 1
    (Allen et al., Nature 544, 227 (2017) for one-layer networks).
    run_ws_validation_ring_baba.py uses pi_init + delta * (this value) as the
    one-layer prediction of rho_C, where pi_init is the value at delta = 0.
    """
    return (1.0 / N) * (c * theta2 + b * (theta1 - theta3))


def drho_C_ddelta_twolayer(b, c, r, theta1, theta2, theta3, phi20, N):
    """d rho_C / d delta at delta = 0 for the two-layer network.

        (1/N) [c theta2 + b (theta1 - theta3) - (r - 1) phi20]

    obtained from S1 Text, Eq (S44) with Eqs (S47) and (S48). The same
    expression holds under dB-dB and dB-Bd, because layer 1 uses the dB rule
    in both; the rule enters through phi20. run_ws_validation_ring_baba.py
    uses pi_init + delta * (this value) as the two-layer prediction of rho_C.
    """
    return (1.0 / N) * (c * theta2 + b * (theta1 - theta3) - (r - 1.0) * phi20)


# ----------------------------------------------------------------------------
# 4. Pure-NumPy reference Monte Carlo simulator (not used by any script in
#    this folder; the SI simulations use weak_selection_sim_fast.py)
# ----------------------------------------------------------------------------

def _build_neighbor_arrays(G: nx.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack graph G into flat neighbour, weight and offset arrays.

    Nodes are relabelled 0..N-1 in sorted order. Not called by any script in
    this folder (weak_selection_sim_fast.pack_graph builds the same arrays).

    Returns
    -------
    ptr   : (N+1,) int64 array; node i's neighbours are at indices ptr[i]:ptr[i+1].
    neigh : (E,) int64 array of neighbour indices (each undirected edge twice).
    w     : (E,) float64 array of edge weights, aligned with neigh.
    deg   : (N,) float64 array of weighted degrees s_i.
    """
    N = G.number_of_nodes()
    nodes = sorted(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    ptr = [0]
    neigh_list = []
    w_list = []
    for u in nodes:
        for v, data in G[u].items():
            neigh_list.append(idx[v])
            w_list.append(data.get("weight", 1.0))
        ptr.append(len(neigh_list))
    return (np.asarray(ptr, dtype=np.int64),
            np.asarray(neigh_list, dtype=np.int64),
            np.asarray(w_list, dtype=np.float64),
            np.asarray([sum(data.get("weight", 1.0) for _, data in G[u].items())
                        for u in nodes], dtype=np.float64))


def _payoff_layer1_pf(x1: np.ndarray, A1: np.ndarray, deg1: np.ndarray, b: float, c: float) -> np.ndarray:
    """Layer-1 payoffs of all nodes under the pf goods scheme.

        u1_i = -c x1_i + b (A1 x1)_i / deg1_i

    with A1 the (weighted) adjacency matrix and deg1 its row sums. Returns a
    length-N array. Not called by any script in this folder
    (simulate_multilayer computes the same expression inline).
    """
    return -c * x1 + b * (A1 @ x1) / deg1


def simulate_multilayer(
    A1: np.ndarray,
    A2: np.ndarray,
    deg1: np.ndarray,
    deg2: np.ndarray,
    b: float,
    c: float,
    r: float,
    delta: float,
    layer2_rule: str,        # 'dB' or 'Bd'
    initial_C_node: int,
    initial_M_node: int,
    coupled: bool,           # True: both layers update; False: only L1 updates (single-layer game)
    n_runs: int,
    seed: int,
    max_steps: int = 10_000_000,
) -> dict:
    """Estimate the fixation probability of a single cooperator by simulation.

    Pure-NumPy reference implementation of the dynamics that
    weak_selection_sim_fast.py implements with numba. Not called by any
    script in this folder.

    Parameters
    ----------
    A1, A2          : N x N (weighted) adjacency matrices of layers 1 and 2.
    deg1, deg2      : their row sums (deg2 is not used).
    b, c, r, delta  : benefit, cost, mutant fitness, selection strength.
    layer2_rule     : 'dB', or any other value for Bd.
    initial_C_node  : node of the single initial cooperator in layer 1.
    initial_M_node  : node of the single initial mutant in layer 2.
    coupled         : True for the two-layer model; False for layer 1 on its
                      own (layer 2 is neither updated nor part of the
                      fecundity).
    n_runs, seed    : number of runs; seed of numpy.random.default_rng.
    max_steps       : a run that has not ended after this many steps is
                      counted as timed out.

    One time step: fecundities F_i = 1 + delta (u1_i + u2_i) are computed
    from the current state (F_i = 1 + delta u1_i if not coupled), with
    u1_i = -c x1_i + b (A1 x1)_i / deg1_i and u2_i = x2_i (r - 1) + 1. Then
    layer 1 makes one dB update (a uniformly chosen node d copies neighbour
    k with probability proportional to A1[d, k] F_k) and, if coupled, layer
    2 makes one update with the same F: dB as in layer 1 but on A2, or Bd
    (parent k chosen with probability F_k / sum_l F_l, its type copied to
    neighbour j with probability A2[k, j] / sum_l A2[k, l]). A run ends when
    layer 1 is all-C or all-D.

    Returns
    -------
    dict with
      n_runs, completed (runs that ended), timed_out,
      n_C        runs that ended with all-C in layer 1,
      rho_C      n_C / completed,
      se_rho_C   sqrt(rho_C (1 - rho_C) / completed),
      mean_steps mean number of steps of the completed runs.
    """
    rng = np.random.default_rng(seed)
    N = A1.shape[0]

    rho_count = 0
    total_steps = 0
    completed = 0
    timed_out = 0

    for _ in range(n_runs):
        x1 = np.zeros(N, dtype=np.int8)
        x1[initial_C_node] = 1
        if coupled:
            x2 = np.zeros(N, dtype=np.int8)
            x2[initial_M_node] = 1
        # else: x2 unused

        steps = 0
        while True:
            # Check L1 absorption.
            s1 = int(x1.sum())
            if s1 == 0 or s1 == N:
                if s1 == N:
                    rho_count += 1
                completed += 1
                total_steps += steps
                break
            if steps >= max_steps:
                timed_out += 1
                break

            # Payoffs and fecundities from the state at the start of the step.
            u1 = -c * x1 + b * (A1 @ x1) / deg1
            if coupled:
                u2 = x2 * (r - 1.0) + 1.0
                F = 1.0 + delta * (u1 + u2)
            else:
                F = 1.0 + delta * u1

            # ---- dB update on layer 1 ----
            d1 = rng.integers(N)
            row = A1[d1]
            mask = row > 0
            if mask.any():
                w = row[mask] * F[mask]
                idx_neighbors = np.flatnonzero(mask)
                p = w / w.sum()
                parent = rng.choice(idx_neighbors, p=p)
                x1[d1] = x1[parent]

            # ---- update on layer 2, if coupled ----
            if coupled:
                if layer2_rule == "dB":
                    d2 = rng.integers(N)
                    row2 = A2[d2]
                    mask2 = row2 > 0
                    if mask2.any():
                        w2 = row2[mask2] * F[mask2]
                        idx2 = np.flatnonzero(mask2)
                        p2 = w2 / w2.sum()
                        parent2 = rng.choice(idx2, p=p2)
                        x2[d2] = x2[parent2]
                else:  # Bd
                    # Parent k chosen with probability F_k / sum_l F_l over
                    # all nodes; its offspring replaces a neighbour chosen
                    # with probability proportional to A2[k, j].
                    p_parent = F / F.sum()
                    k = rng.choice(N, p=p_parent)
                    rowk = A2[k]
                    mask2 = rowk > 0
                    if mask2.any():
                        w_off = rowk[mask2]
                        idx2 = np.flatnonzero(mask2)
                        p_off = w_off / w_off.sum()
                        target = rng.choice(idx2, p=p_off)
                        x2[target] = x2[k]
            steps += 1

    rho_C = rho_count / max(1, completed)
    se = np.sqrt(rho_C * (1.0 - rho_C) / max(1, completed))
    mean_steps = total_steps / max(1, completed)
    return {
        "n_runs": n_runs,
        "completed": completed,
        "timed_out": timed_out,
        "n_C": rho_count,
        "rho_C": rho_C,
        "se_rho_C": se,
        "mean_steps": mean_steps,
    }


# ----------------------------------------------------------------------------
# 5. Full theory pipeline for one two-layer network and initial condition
# ----------------------------------------------------------------------------

def network_to_arrays(G: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Dense adjacency matrix A (nodes in sorted order, edge attribute
    'weight', 1 if absent) and weighted degrees deg = A.sum(axis=1), for
    use with simulate_multilayer. Not called by any script in this folder.
    """
    nodes = sorted(G.nodes())
    N = len(nodes)
    A = np.zeros((N, N))
    for i, u in enumerate(nodes):
        for v, data in G[u].items():
            j = nodes.index(v)
            A[i, j] = data.get("weight", 1.0)
    deg = A.sum(axis=1)
    return A, deg


def analytical_summary(G1: nx.Graph, G2: nx.Graph,
                        initial_C_node: int, initial_M_node: int,
                        rule: str, r: float, c: float = 1.0) -> dict:
    """theta_n, phi_{n,m} and both thresholds for one network and initial condition.

    Parameters
    ----------
    G1, G2         : layer-1 and layer-2 networks on the same node set
                     (nodes taken in sorted order; edge attribute 'weight',
                     1 if absent).
    initial_C_node : position, in sorted node order, of the single initial
                     cooperator in layer 1.
    initial_M_node : position of the single initial mutant in layer 2.
    rule           : 'dB-dB' or 'dB-Bd' (anything else raises ValueError).
    r              : mutant fitness in layer 2.
    c              : cost of cooperation (default 1).

    Steps: P1, pi1 (and P2) from transition_matrix; beta_ij from solve_beta;
    gamma_ij from solve_gamma_dBdB_v2 (dB-dB, with the dB reproductive
    values pi2 of layer 2) or solve_gamma_dBBd_corrected (dB-Bd, with the Bd
    reproductive values pi2_Bd of layer 2); theta_n from compute_thetas,
    phi_{n,m} from compute_phis, and the thresholds from compute_bc_star.

    Returns
    -------
    dict with
      N                       number of nodes,
      theta1, theta2, theta3  Eq (S47),
      phi01, phi20, phi21     Eq (S48),
      bc_star_single          -theta2 / (theta1 - theta3),
      bc_star_two             (b/c)* at the given r (Eq (5) of the main text),
      pi1_initialC            pi1 of the initial cooperator's node.
    """
    P1, pi1 = transition_matrix(G1)
    if rule == "dB-dB":
        P2, pi2 = transition_matrix(G2)
        gamma_solver = lambda xi1, xi2: solve_gamma_dBdB_v2(P1, P2, pi1, pi2, xi1, xi2)
    elif rule == "dB-Bd":
        # P2[k, j] = w_kj / s_k is also the Bd offspring-placement probability.
        P2, pi2 = transition_matrix(G2)
        # Bd reproductive values of layer 2: pi2_Bd_j = (1/s_j) / sum_l (1/s_l)
        # (S1 Text, Eq (S65)).
        deg2 = np.asarray([sum(d.get("weight", 1.0) for _, d in G2[u].items())
                           for u in sorted(G2.nodes())])
        pi2_Bd = (1.0 / deg2) / (1.0 / deg2).sum()
        # gamma_ij under dB-Bd; coefficients in the docstring of
        # solve_gamma_dBBd_corrected.
        gamma_solver = lambda xi1, xi2: solve_gamma_dBBd_corrected(P1, P2, pi1, pi2_Bd, xi1, xi2)
    else:
        raise ValueError(rule)

    N = P1.shape[0]
    # Initial states: one cooperator in layer 1, one mutant in layer 2.
    xi1 = np.zeros(N); xi1[initial_C_node] = 1.0
    xi2 = np.zeros(N); xi2[initial_M_node] = 1.0

    beta_i, beta_full = solve_beta(P1, pi1, xi1)
    gamma = gamma_solver(xi1, xi2)

    theta1, theta2, theta3 = compute_thetas(P1, pi1, beta_full)
    phi01, phi20, phi21 = compute_phis(P1, P2, pi1, gamma)

    # (b/c)*_single: r = 1 and phi20 = 0; (b/c)*_two: the given r and phi20.
    bc_star_single = compute_bc_star(theta1, theta2, theta3, 0.0, 1.0, c)
    bc_star_two = compute_bc_star(theta1, theta2, theta3, phi20, r, c)

    return {
        "N": N,
        "theta1": theta1, "theta2": theta2, "theta3": theta3,
        "phi01": phi01, "phi20": phi20, "phi21": phi21,
        "bc_star_single": bc_star_single,
        "bc_star_two": bc_star_two,
        "pi1_initialC": pi1[initial_C_node],
    }


if __name__ == "__main__":
    # ---- Check on the two-layer ring with N = 10 (dB-dB, r = 2) ----
    # First initial condition: cooperator on node 0 of layer 1, mutant on its
    # neighbour, node 1, of layer 2 (distance d = 1). The printed lines that
    # start with "Expected" are reference values for (b/c)*_single and
    # (b/c)*_two of this case.
    G = nx.cycle_graph(10)
    G2 = nx.cycle_graph(10)
    print("\n=== Adjacent IC (d=1), Qi Su Fig 3a convention ===")
    s = analytical_summary(G, G2, 0, 1, "dB-dB", r=2.0)
    for k, v in s.items():
        print(f"  {k} = {v}")
    print(f"\n  Expected (b/c)*_single = 8/3 = {8/3:.4f}")
    print(f"  Expected (b/c)*_two (r=2) = 2.57 (per main text v35 line 369)")

    # Second initial condition: cooperator and mutant both on node 0 (d = 0).
    print("\n=== Co-located IC (d=0) ===")
    s = analytical_summary(G, G2, 0, 0, "dB-dB", r=2.0)
    for k, v in s.items():
        print(f"  {k} = {v}")
