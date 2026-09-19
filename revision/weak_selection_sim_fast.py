"""
Numba-compiled Monte Carlo simulator of the two-layer (bilayer) model under the
dB-dB and dB-Bd updating rules. Library module: it produces no SI item by itself.

SI items it supports (S1 Text)
------------------------------
* Text L, Fig G (ring N=10 and BA N=15, theory vs simulation) and
  Text L, Fig I (ER/BA N=100, theory vs simulation):
  `run_ws_validation_ring_baba.py` and `point_b_mc_overlay.py` call `run_sim`
  twice per b/c value, with coupled=True ("two-layer" markers) and
  coupled=False ("one-layer" markers, i.e. layer 1 considered separately),
  using the default max_steps=2_000_000.
  Call chain: run_sim -> pack_graph, simulate_many -> simulate_one_run ->
  _payoffs_layer1_pf, _step_dB_layer, _step_Bd_layer.
* Text F, Fig C (strategy-type correlation, N=6):
  `r3_strategy_coupling_ic.py` imports `pack_graph`; `r3_strategy_coupling.py`
  imports `_payoffs_layer1_pf`, `_step_dB_layer`, `_step_Bd_layer` and builds
  its own trajectory loop around them (Fig C uses dB-dB, so `_step_Bd_layer`
  is not exercised there).

Functions used by the published pipelines vs. unused code
---------------------------------------------------------
used   : pack_graph, _payoffs_layer1_pf, _step_dB_layer, _step_Bd_layer,
         simulate_one_run, simulate_many, run_sim
unused : _sample_weighted (not called anywhere in this folder); the
         `rng_state` argument of _step_dB_layer/_step_Bd_layer (callers pass 0;
         random numbers come from Numba's internal np.random state); the
         __main__ block (ring N=10 pilot, not an SI item).

Model conventions (same as the S1 Text)
---------------------------------------
* Node index i = position of the node in sorted(G.nodes()); the same
  convention as larger_networks_solver._transition_matrix_sparse, so init_C /
  init_M refer to the same node in the simulator and in the weak-selection
  solver. Individual i occupies index i in both layers.
* State: x1[i] = 1 cooperator / 0 defector (layer 1, donation game);
         x2[i] = 1 mutant / 0 resident (layer 2, constant selection).
* Layer-1 payoff, pf goods scheme used in the main text (main-text Eq. (1);
  S1 Text Eq. (S135) in Text M; the ff scheme of Text M is not implemented
  here):
      u1_i = -c x1_i + b * sum_j (w_ij / s_i) x1_j ,  s_i = sum_j w_ij.
* Layer-2 payoff (main-text Eq. (2); S1 Text Eq. (S23) in Text A):
      u2_i = x2_i (r-1) + 1.
* Fecundity F_i = 1 + delta (u1_i + u2_i) when coupled;
  F_i = 1 + delta u1_i when coupled=False (layer 2 neither enters F nor updates).
* One time step = one dB update in layer 1 followed (if coupled) by one dB or
  Bd update in layer 2; both updates use the F vector computed from the state
  at the start of the step.
  dB (S1 Text Eq. (S41) in Text B): a uniformly random node d dies and copies
     neighbour k with probability w_dk F_k / sum_l w_dl F_l.
  Bd (S1 Text Eq. (S62) in Text C): parent k is chosen with probability
     F_k / sum_l F_l and its type replaces neighbour j chosen with probability
     w_kj / s_k.
* Initial condition: x1[init_C] = 1, x2[init_M] = 1, all other entries 0.
* A run ends when layer 1 is monomorphic (layer 2 may still be polymorphic);
  rho_C = P(layer 1 reaches all-C) is estimated as n_C / (n_C + n_D).

The dynamics are the same as the pure-NumPy reference
`weak_selection_validation.simulate_multilayer` (same transition probabilities;
different random-number streams, so results agree in distribution, not bit for
bit), with the inner loop compiled and neighbours stored in CSR-style arrays.

Seeding / reproducibility
-------------------------
`simulate_many` is compiled with parallel=True. np.random.seed(seed) inside it
seeds the Numba random state of the calling thread; prange iterations that run
on other Numba worker threads use those threads' own random states. With more
than one Numba thread, repeated calls with the same seed therefore return
different counts (observed with 8 threads: ring N=10, b=3, r=2, delta=0.02,
20,000 runs, seed=5 -> n_C = 2074, 2028, 2061 in three calls). With
NUMBA_NUM_THREADS=1 the same call returned n_C = 2050 every time, within and
across processes. The kernels imported by r3_strategy_coupling.py are called
from a serial loop there and are seed-deterministic.

Usage
-----
Library: `from weak_selection_sim_fast import run_sim`.
`python weak_selection_sim_fast.py` (from this folder) runs a pilot (dB-dB ring
N=10, b/c=3.5, r=2, delta=0.02, 10^6 runs; ~12 s wall including JIT on an
8-thread Mac). Inputs: none. Outputs: stdout only.
`cache=True` makes Numba write compiled-code cache files into __pycache__/ next
to this file.

Dependencies: numpy, networkx, numba (tested with Python 3.13, numpy 2.1.3,
networkx 3.4.2, numba 0.61.0).
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from numba import njit, prange


# ---------------------------------------------------------------
# CSR-style packing
# ---------------------------------------------------------------

def pack_graph(G: nx.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack an undirected (optionally weighted) graph into CSR-like arrays.

    Parameters
    ----------
    G : networkx.Graph
        One network layer. Nodes are relabelled 0..N-1 in sorted(G.nodes())
        order; edge attribute "weight" is used if present, otherwise 1.0.

    Returns
    -------
    (ptr, neigh, w, deg)
    ptr  : (N+1,) int64 -- neighbours of node i are neigh[ptr[i]:ptr[i+1]]
    neigh: (E,) int64 -- flat neighbour indices (each undirected edge appears
           twice, once per endpoint)
    w    : (E,) float64 -- flat edge weights w_ij aligned with `neigh`
    deg  : (N,) float64 -- weighted degree s_i = sum_j w_ij

    Within each node's block the neighbours follow networkx's adjacency
    (insertion) order G[u], not sorted order. The cumulative-sum samplers in
    _step_dB_layer / _step_Bd_layer walk this order, so it is part of what a
    given random seed maps to.
    """
    N = G.number_of_nodes()
    nodes = sorted(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    ptr = [0]
    neigh: list[int] = []
    weights: list[float] = []
    for u in nodes:
        for v, data in G[u].items():
            neigh.append(idx[v])
            weights.append(data.get("weight", 1.0))
        ptr.append(len(neigh))
    deg = np.zeros(N)
    for i in range(N):
        deg[i] = sum(weights[ptr[i]:ptr[i + 1]])
    return (np.asarray(ptr, dtype=np.int64),
            np.asarray(neigh, dtype=np.int64),
            np.asarray(weights, dtype=np.float64),
            deg.astype(np.float64))


# ---------------------------------------------------------------
# Numba-jitted simulation
# ---------------------------------------------------------------

@njit(cache=True)
def _sample_weighted(weights, total, r_uniform):
    """Return the first index i with r_uniform <= weights[0] + ... + weights[i].

    `r_uniform` is a draw from U(0, total); `total` itself is not used inside.
    Falls back to the last index if round-off leaves r_uniform above the final
    cumulative sum. Not called by any script in this folder.
    """
    cum = 0.0
    for i in range(weights.shape[0]):
        cum += weights[i]
        if r_uniform <= cum:
            return i
    return weights.shape[0] - 1


@njit(cache=True)
def _payoffs_layer1_pf(x1, ptr, neigh, w, deg, b, c):
    """Layer-1 donation-game payoffs under the pf goods scheme.

    u1_i = -c * x1_i + b * sum_j p_ij x1_j,  with p_ij = w_ij / s_i
    (main-text Eq. (1); S1 Text Eq. (S135)). If s_i = 0, u1_i = -c x1_i.

    Parameters: x1 (N,) int8 layer-1 state (1 = C); ptr, neigh, w, deg from
    pack_graph(G1); b benefit; c cost.
    Returns: (N,) float64 array of u1_i.
    """
    N = x1.shape[0]
    u = np.empty(N)
    for i in range(N):
        s = 0.0
        for k in range(ptr[i], ptr[i + 1]):
            j = neigh[k]
            s += w[k] * x1[j]
        if deg[i] > 0.0:
            u[i] = -c * x1[i] + b * s / deg[i]
        else:
            u[i] = -c * x1[i]
    return u


@njit(cache=True)
def _step_dB_layer(x, F, ptr, neigh, w, rng_state):
    """One death-Birth (dB) update on one layer; modifies x in place.

    A node d is chosen uniformly at random to die; it adopts the type of
    neighbour k with probability w_dk F_k / sum_l w_dl F_l (S1 Text Eq. (S41)).
    If sum_l w_dl F_l is zero, x is left unchanged.

    Parameters: x (N,) int8 state of this layer; F (N,) fecundities; ptr,
    neigh, w from pack_graph of this layer; rng_state is unused (random numbers
    come from Numba's np.random state).
    Returns: None. Consumes two uniform draws (one if sum_l w_dl F_l is zero).
    """
    N = x.shape[0]
    # death: pick uniformly random
    d = int(np.floor(np.random.random() * N))
    if d >= N:
        d = N - 1
    deg_d_F = 0.0
    for k in range(ptr[d], ptr[d + 1]):
        deg_d_F += w[k] * F[neigh[k]]
    if deg_d_F == 0.0:
        return
    # birth: neighbour chosen by cumulative sum over w_dk * F_k, in pack_graph order
    r = np.random.random() * deg_d_F
    cum = 0.0
    for k in range(ptr[d], ptr[d + 1]):
        cum += w[k] * F[neigh[k]]
        if r <= cum:
            x[d] = x[neigh[k]]
            return
    # fallback (numerical edge case)
    x[d] = x[neigh[ptr[d + 1] - 1]]


@njit(cache=True)
def _step_Bd_layer(x, F, ptr, neigh, w, rng_state):
    """One Birth-death (Bd) update on one layer; modifies x in place.

    A parent is chosen among all N nodes with probability F_parent / sum_l F_l;
    its type replaces a neighbour j chosen with probability w_parent,j / s_parent
    (fitness does not enter this second choice) (S1 Text Eq. (S62)).

    Parameters: as in _step_dB_layer (rng_state unused).
    Returns: None. Consumes two uniform draws (one if the parent has no
    neighbour entries in pack_graph, i.e. ptr[parent] == ptr[parent + 1]; x is
    then left unchanged).
    """
    N = x.shape[0]
    total_F = 0.0
    for i in range(N):
        total_F += F[i]
    r1 = np.random.random() * total_F
    cum = 0.0
    parent = N - 1
    for i in range(N):
        cum += F[i]
        if r1 <= cum:
            parent = i
            break
    # Pick neighbor of parent (random by weight)
    if ptr[parent] == ptr[parent + 1]:
        return  # parent has no neighbour entries: x is left unchanged
    # deg_F is the (unweighted-by-fitness) weighted degree s_parent
    deg_F = 0.0
    for k in range(ptr[parent], ptr[parent + 1]):
        deg_F += w[k]
    r2 = np.random.random() * deg_F
    cum = 0.0
    for k in range(ptr[parent], ptr[parent + 1]):
        cum += w[k]
        if r2 <= cum:
            x[neigh[k]] = x[parent]
            return
    x[neigh[ptr[parent + 1] - 1]] = x[parent]


@njit(cache=True)
def simulate_one_run(
    ptr1, neigh1, w1, deg1,
    ptr2, neigh2, w2, deg2,
    init_C, init_M,
    b, c, r, delta,
    coupled, layer2_is_Bd,
    max_steps,
):
    """Run one stochastic trajectory until layer 1 is absorbed.

    Parameters
    ----------
    ptr1, neigh1, w1, deg1 : pack_graph(G1), layer 1 (donation game, dB rule)
    ptr2, neigh2, w2, deg2 : pack_graph(G2), layer 2 (constant selection);
                             deg2 is not used
    init_C, init_M         : node index of the initial cooperator (layer 1) and
                             of the initial mutant (layer 2)
    b, c                   : donation-game benefit and cost
    r                      : mutant fitness in layer 2 (resident = 1)
    delta                  : selection strength
    coupled                : True -> two-layer model, F = 1 + delta(u1 + u2),
                             layer 2 updated every step;
                             False -> one-layer model, F = 1 + delta u1, x2 stays
                             all 0 and is never updated
    layer2_is_Bd           : True -> Bd in layer 2 (dB-Bd rule); False -> dB (dB-dB)
    max_steps              : cap on the number of time steps

    Returns
    -------
    (final_state_layer1, n_steps)
    final_state_layer1: 1 if all-C, 0 if all-D, -1 if timed out.
    n_steps: number of completed steps when layer 1 was found absorbed
             (max_steps on time-out).
    """
    N = ptr1.shape[0] - 1
    x1 = np.zeros(N, dtype=np.int8)
    x1[init_C] = 1
    x2 = np.zeros(N, dtype=np.int8)
    if coupled:
        x2[init_M] = 1

    steps = 0
    while steps < max_steps:
        # check absorption layer 1
        s1 = 0
        for i in range(N):
            s1 += x1[i]
        if s1 == 0:
            return 0, steps
        if s1 == N:
            return 1, steps

        # compute payoffs
        u1 = _payoffs_layer1_pf(x1, ptr1, neigh1, w1, deg1, b, c)
        F = np.empty(N)
        if coupled:
            # u2_i = x2_i (r-1) + 1  (mutant r, resident 1)
            for i in range(N):
                F[i] = 1.0 + delta * (u1[i] + (x2[i] * (r - 1.0) + 1.0))
        else:
            for i in range(N):
                F[i] = 1.0 + delta * u1[i]

        # ensure positivity (if a non-positive fitness shows up, clip)
        # (u1_i >= -c for b >= 0, so the clip is inactive whenever delta*c < 1)
        for i in range(N):
            if F[i] <= 0.0:
                F[i] = 1e-12

        # dB update on layer 1
        _step_dB_layer(x1, F, ptr1, neigh1, w1, 0)
        # update layer 2 if coupled (uses the same F, computed before the layer-1 update)
        if coupled:
            if layer2_is_Bd:
                _step_Bd_layer(x2, F, ptr2, neigh2, w2, 0)
            else:
                _step_dB_layer(x2, F, ptr2, neigh2, w2, 0)
        steps += 1
    return -1, steps


@njit(cache=True, parallel=True)
def simulate_many(
    ptr1, neigh1, w1, deg1,
    ptr2, neigh2, w2, deg2,
    init_C, init_M,
    b, c, r, delta,
    coupled, layer2_is_Bd,
    n_runs, max_steps, seed,
):
    """Run n_runs independent trajectories of simulate_one_run with numba prange.

    Arguments are those of simulate_one_run plus n_runs (number of runs) and
    seed (passed to np.random.seed).

    Returns (n_C, n_D, n_timeout, sum_steps): the number of runs ending in
    all-C, all-D, and not absorbed within max_steps, and the sum of n_steps over
    all runs (time-outs contribute max_steps).

    See the module docstring ("Seeding / reproducibility"): with more than one
    Numba thread the counts are not a deterministic function of `seed`.
    """
    n_C = 0
    n_D = 0
    n_timeout = 0
    sum_steps = 0

    # np.random.seed seeds the Numba random state of the calling thread only;
    # prange iterations that run on other worker threads use those threads'
    # own random states (see "Seeding / reproducibility" above).
    np.random.seed(seed)

    for _ in prange(n_runs):
        outcome, steps = simulate_one_run(
            ptr1, neigh1, w1, deg1,
            ptr2, neigh2, w2, deg2,
            init_C, init_M,
            b, c, r, delta,
            coupled, layer2_is_Bd,
            max_steps,
        )
        # n_C, n_D, n_timeout, sum_steps are prange reduction variables
        if outcome == 1:
            n_C += 1
        elif outcome == 0:
            n_D += 1
        else:
            n_timeout += 1
        sum_steps += steps
    return n_C, n_D, n_timeout, sum_steps


def run_sim(
    G1: nx.Graph, G2: nx.Graph,
    init_C: int, init_M: int,
    b: float, c: float, r: float, delta: float,
    rule: str,         # 'dB-dB' or 'dB-Bd'
    coupled: bool,
    n_runs: int,
    seed: int = 0,
    max_steps: int = 2_000_000,
) -> dict:
    """Estimate the fixation probability of a single cooperator by simulation.

    Packs both layers with pack_graph and calls simulate_many.

    Parameters
    ----------
    G1, G2         : layer-1 and layer-2 networks on the same N nodes
    init_C, init_M : index (in sorted node order) of the initial cooperator /
                     initial mutant
    b, c, r, delta : benefit, cost, mutant fitness, selection strength
    rule           : 'dB-dB' or 'dB-Bd' (any other string is treated as dB-dB)
    coupled        : True for the two-layer model, False for layer 1 alone
    n_runs, seed, max_steps : see simulate_many / simulate_one_run

    Returns
    -------
    dict with
      n_runs, completed = n_C + n_D, n_C, n_D, n_timeout,
      rho_C     = n_C / completed (time-outs excluded; NaN if completed == 0),
      se_rho_C  = sqrt(rho_C (1 - rho_C) / completed), binomial standard error,
      mean_steps = sum_steps / n_runs (mean layer-1 absorption time over all
                   runs, time-outs counted as max_steps).
    """
    ptr1, neigh1, w1, deg1 = pack_graph(G1)
    ptr2, neigh2, w2, deg2 = pack_graph(G2)
    layer2_is_Bd = (rule == "dB-Bd")

    n_C, n_D, n_timeout, sum_steps = simulate_many(
        ptr1, neigh1, w1, deg1,
        ptr2, neigh2, w2, deg2,
        init_C, init_M,
        b, c, r, delta,
        coupled, layer2_is_Bd,
        n_runs, max_steps, seed,
    )
    completed = n_C + n_D
    rho_C = n_C / completed if completed > 0 else float("nan")
    se = (rho_C * (1.0 - rho_C) / completed) ** 0.5 if completed > 0 else float("nan")
    return {
        "n_runs": n_runs,
        "completed": completed,
        "n_C": n_C, "n_D": n_D,
        "n_timeout": n_timeout,
        "rho_C": rho_C,
        "se_rho_C": se,
        "mean_steps": sum_steps / max(n_runs, 1),
    }


if __name__ == "__main__":
    # Pilot / timing check only (not an SI item).
    import time
    G1 = nx.cycle_graph(10)
    G2 = nx.cycle_graph(10)
    # Warm up JIT
    print("Warming up JIT...")
    _ = run_sim(G1, G2, 0, 1, 3.0, 1.0, 2.0, 0.02, "dB-dB", True, 100, seed=0)
    print("Pilot dB-dB ring N=10, b/c=3.5, r=2, delta=0.02:")
    t0 = time.time()
    res = run_sim(G1, G2, 0, 1, 3.5, 1.0, 2.0, 0.02, "dB-dB", True,
                   n_runs=10**6, seed=0)
    elapsed = time.time() - t0
    print(f"  rho_C = {res['rho_C']:.4f} +/- {res['se_rho_C']:.4f}")
    print(f"  rho_C / (1/N) = {res['rho_C']*10:.4f}")
    print(f"  completed = {res['completed']}, mean steps = {res['mean_steps']:.1f}")
    print(f"  elapsed = {elapsed:.1f}s")
