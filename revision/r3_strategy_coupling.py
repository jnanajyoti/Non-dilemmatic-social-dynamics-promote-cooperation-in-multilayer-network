"""
Simulation kernels for the strategy-type correlation between the two layers
(S1 Text, Text F, Fig C), plus an earlier stand-alone version of that analysis
which is NOT used for any SI item.

Role in the published pipeline
------------------------------
Fig C is produced by `r3_strategy_coupling_ic.py`, which imports from this
module only
    _sim_many_with_corr   (-> _sim_one_with_corr -> _record_R -> _pearson_phi)
    TIME_BINS, MAX_STEPS
and, through these kernels, weak_selection_sim_fast._payoffs_layer1_pf and
_step_dB_layer (_step_Bd_layer is imported but only reached when
layer2_is_Bd=True, which Fig C does not use). See r3_strategy_coupling_ic.py for
the Fig C command, networks, parameters and outputs.

Quantity computed: at time t (after the t-th update step) the Pearson
correlation coefficient across the N nodes between the layer-1 state x1
(defector 0, cooperator 1) and the layer-2 state x2 (resident 0, mutant 1).
For binary vectors this equals the phi coefficient of the 2x2 table of node
counts. It is undefined (NaN, excluded from averages) whenever either layer is
monomorphic, which is the "surviving runs" condition stated in Text F.

Legacy code in this file (not used for Fig C or any other SI item)
-----------------------------------------------------------------
pick_pairs, run_pair, b_for_pair, main, plot, the __main__ block, and the
constants RULE, R (=5.0), C, DELTA, INIT_C, INIT_M, N_RUNS, N_BINS,
EXHAUSTIVE_CSV, OUT_RESULTS, OUT_PAIRS, OUT_FIG. This earlier pipeline
  * reads r3_n6_exhaustive_results.csv (from r3_topology_correlation_n6_exhaustive.py;
    dB-dB, r=5, theta/phi averaged over the six co-located initial conditions),
  * picks 5 "enhancing" (most negative slope phi20/theta_diff) and 5
    "suppressing" (most positive slope) (G1, G2) pairs subject to
    |theta_diff| >= 0.1, |bc_single| <= 30, |bc_two| <= 15 and at most one pair
    per G1,
  * simulates only the co-located initial condition (INIT_C = INIT_M = 0) at r=5,
  * writes r3_strategy_coupling_pairs.csv, r3_strategy_coupling_results.csv and
    r3_strategy_coupling.png (none of these files is in the repository).
Fig C instead uses the five hard-coded networks NETS_A in
r3_strategy_coupling_ic.py (selected by the Text F rule described there), r=4,
and both the correlated and the uncorrelated initial condition.

Legacy usage (not needed to reproduce the SI; runtime not measured):
    cd <this folder>
    python r3_strategy_coupling.py              # pick pairs, simulate, plot
    python r3_strategy_coupling.py --plot-only  # re-plot from the two CSVs above
Inputs/outputs are relative to the current working directory.

Dependencies: numpy, networkx, pandas, scipy (via larger_networks_solver),
matplotlib, numba; local modules weak_selection_sim_fast and
larger_networks_solver (bc_star_two_layer is imported but not called here).
Numba `cache=True` writes compiled-code cache files into __pycache__/ next to
this file.
"""
from __future__ import annotations
import os, time, json
import warnings, scipy.sparse as sp
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

from weak_selection_sim_fast import pack_graph, _payoffs_layer1_pf, \
    _step_dB_layer, _step_Bd_layer
from larger_networks_solver import bc_star_two_layer


# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
# Only TIME_BINS and MAX_STEPS are imported by r3_strategy_coupling_ic.py (Fig C).
# All other constants in this block configure the legacy pipeline (main/plot).
RULE        = "dB-dB"
R           = 5.0
C           = 1.0
DELTA       = 0.02
INIT_C      = 0       # sorted-first node in each graph
INIT_M      = 0       # co-located: starting Pearson R(0) = +1
N_RUNS      = 500_000  # per (G1, G2)
MAX_STEPS   = 5_000    # Fig C: no run of either layer reached this cap (0 time-outs in all 15 jobs)
EXHAUSTIVE_CSV = "r3_n6_exhaustive_results.csv"
OUT_RESULTS    = "r3_strategy_coupling_results.csv"
OUT_PAIRS      = "r3_strategy_coupling_pairs.csv"
OUT_FIG        = "r3_strategy_coupling.png"

# Time bins (in update steps). Mean fixation time on N=6 is O(N)~10-20 steps,
# so we densify the early region and cap at 200.
# (In the Fig C runs the mean absorption time was 8.5-10.5 steps in layer 1 and
# 9.4-22.3 steps in layer 2.) Must be strictly increasing and >= 1, because R is
# recorded only right after a step whose index equals the next bin.
TIME_BINS = np.array(
    [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200],
    dtype=np.int64,
)
N_BINS = len(TIME_BINS)


# --------------------------------------------------------------
# Pick 5 enhancing + 5 suppressing pairs
# --------------------------------------------------------------
def pick_pairs(csv_path: str = EXHAUSTIVE_CSV, n_each: int = 5) -> pd.DataFrame:
    """LEGACY (not used for Fig C): choose "enhancing" and "suppressing" pairs.

    Parameters
    ----------
    csv_path : path of r3_n6_exhaustive_results.csv (one row per ordered pair of
               connected 6-node graphs; columns g1_idx, g2_idx, theta_diff,
               phi20, bc_single, bc_two, ... averaged over the six co-located
               initial conditions, dB-dB, r=5).
    n_each   : number of pairs per group.

    Returns
    -------
    DataFrame with 2*n_each rows (the CSV columns plus "slope" and "group").
    slope = phi20 / theta_diff (= d(b/c)*/dr for c=1), NaN if |theta_diff| <= 1e-12.
    Candidates: finite bc_two, |theta_diff| >= 0.10, |bc_single| <= 30,
    |bc_two| <= 15. "enhancing" = n_each most negative slopes, "suppressing" =
    n_each most positive slopes, each with at most one row per g1_idx.
    """
    df = pd.read_csv(csv_path)
    df = df[np.isfinite(df["bc_two"])].copy()
    TOL = 1e-12
    td  = df["theta_diff"].values
    ph  = df["phi20"].values
    df["slope"] = np.where(np.abs(td) > TOL, ph / td, np.nan)

    # Sanity: |theta_diff| big enough to avoid 1/0 blow-up; bc moderate.
    sane = df[(np.abs(df["theta_diff"]) >= 0.10)
                & (df["bc_single"].abs() <= 30)
                & (df["bc_two"].abs()    <= 15)].copy()

    def diverse(sub: pd.DataFrame, by: str, asc: bool, n: int):
        """Take the first n rows of `sub` sorted by `by`, skipping repeated g1_idx."""
        sorted_sub = sub.sort_values(by, ascending=asc)
        seen, out = set(), []
        for _, row in sorted_sub.iterrows():
            if row["g1_idx"] in seen:
                continue
            seen.add(row["g1_idx"]); out.append(row)
            if len(out) == n:
                break
        return pd.DataFrame(out)

    enh = diverse(sane, "slope", asc=True,  n=n_each).assign(group="enhancing")
    sup = diverse(sane, "slope", asc=False, n=n_each).assign(group="suppressing")
    return pd.concat([enh, sup], ignore_index=True)


# --------------------------------------------------------------
# Simulator: track Pearson R(x1, x2) across nodes at each time bin
# --------------------------------------------------------------
@njit(cache=True)
def _pearson_phi(n_CM, n_CW, n_DM, n_DW):
    """Phi-coefficient = Pearson R for a 2x2 binary table.

    Arguments are node counts: n_CM cooperator & mutant, n_CW cooperator &
    resident ("W" = wild type), n_DM defector & mutant, n_DW defector & resident.

    Returns (n_CM n_DW - n_CW n_DM) / sqrt(n_C n_D n_M n_W), which equals the
    Pearson correlation coefficient between the binary vectors x1 and x2 over
    nodes; NaN if any marginal (n_C, n_D, n_M, n_W) is zero, i.e. if either
    layer is monomorphic.
    """
    n_C = n_CM + n_CW
    n_D = n_DM + n_DW
    n_M = n_CM + n_DM
    n_W = n_CW + n_DW
    if n_C == 0 or n_D == 0 or n_M == 0 or n_W == 0:
        return np.nan
    num = float(n_CM * n_DW - n_CW * n_DM)
    den = (float(n_C) * float(n_D) * float(n_M) * float(n_W)) ** 0.5
    return num / den


@njit(cache=True)
def _record_R(x1, x2, N):
    """Correlation between the two layers' states at the current time.

    Parameters: x1 (N,) int8 layer-1 state (1 = cooperator, 0 = defector);
    x2 (N,) int8 layer-2 state (1 = mutant, 0 = resident); N number of nodes.
    Node i is the same individual in both layers.

    Returns: _pearson_phi of the 2x2 count table (float; NaN if either layer
    is monomorphic).
    """
    n_CM = 0; n_CW = 0; n_DM = 0; n_DW = 0
    for i in range(N):
        if x1[i] == 1 and x2[i] == 1:
            n_CM += 1
        elif x1[i] == 1 and x2[i] == 0:
            n_CW += 1
        elif x1[i] == 0 and x2[i] == 1:
            n_DM += 1
        else:
            n_DW += 1
    return _pearson_phi(n_CM, n_CW, n_DM, n_DW)


@njit(cache=True)
def _sim_one_with_corr(
    ptr1, neigh1, w1, deg1,
    ptr2, neigh2, w2, deg2,
    init_C, init_M,
    b, c, r, delta,
    layer2_is_Bd,
    time_bins, max_steps,
):
    """Run one trajectory; record Pearson R(x1, x2) at each time bin.

    Parameters
    ----------
    ptr1, neigh1, w1, deg1 : weak_selection_sim_fast.pack_graph(G1) (layer 1)
    ptr2, neigh2, w2, deg2 : pack_graph(G2) (layer 2; deg2 unused)
    init_C, init_M : node index (sorted node order) of the single initial
                     cooperator in layer 1 and the single initial mutant in layer 2
    b, c           : donation-game benefit and cost (pf goods scheme)
    r              : mutant fitness in layer 2 (resident 1)
    delta          : selection strength
    layer2_is_Bd   : False -> dB-dB rule (Fig C); True -> dB-Bd rule
    time_bins      : strictly increasing int64 array of step counts (>= 1)
    max_steps      : cap on the number of steps

    Dynamics per step (same model as weak_selection_sim_fast.simulate_one_run
    with coupled=True): F_i = 1 + delta (u1_i + u2_i) with
    u1_i = -c x1_i + b sum_j p_ij x1_j and u2_i = x2_i (r-1) + 1; one dB update in
    layer 1, then one dB (or Bd) update in layer 2 using the same F.
    Unlike simulate_one_run, the run continues until BOTH layers are absorbed
    (or max_steps is reached).

    Returns
    -------
    (R_arr, final_l1, final_l2, n_steps, t_l1, t_l2)
    R_arr    : (n_bins,) float64; R_arr[k] = correlation right after step
               time_bins[k]; NaN if either layer was monomorphic then or the run
               had already stopped. R at t=0 is not recorded.
    final_l1 : 1 all-C, 0 all-D, -1 layer 1 not absorbed within max_steps
    final_l2 : 1 all-mutant, 0 all-resident, -1 not absorbed
    n_steps  : number of steps performed
    t_l1, t_l2 : step count at which each layer was first found absorbed (-1 if never)
    """
    N = ptr1.shape[0] - 1
    n_bins = time_bins.shape[0]
    R_arr = np.full(n_bins, np.nan)
    bin_idx = 0

    x1 = np.zeros(N, dtype=np.int8)
    x1[init_C] = 1
    x2 = np.zeros(N, dtype=np.int8)
    x2[init_M] = 1

    # R at t=0 is never recorded: bins are compared with the step counter only
    # after it has been incremented, so time_bins[0] must be >= 1.
    # We treat "R at bin b" as R immediately after step b.
    final_l1 = -1
    final_l2 = -1
    t_l1 = -1
    t_l2 = -1

    steps = 0
    while steps < max_steps:
        # Check absorption
        s1 = 0; s2 = 0
        for i in range(N):
            s1 += x1[i]; s2 += x2[i]
        if final_l1 < 0 and (s1 == 0 or s1 == N):
            final_l1 = 1 if s1 == N else 0
            t_l1 = steps
        if final_l2 < 0 and (s2 == 0 or s2 == N):
            final_l2 = 1 if s2 == N else 0
            t_l2 = steps
        # Stop only when BOTH layers are absorbed (the run continues while one
        # layer is fixed and the other is polymorphic; R is NaN in that phase)
        if final_l1 >= 0 and final_l2 >= 0:
            break

        # Compute payoffs and fitness
        u1 = _payoffs_layer1_pf(x1, ptr1, neigh1, w1, deg1, b, c)
        F = np.empty(N)
        for i in range(N):
            F[i] = 1.0 + delta * (u1[i] + (x2[i] * (r - 1.0) + 1.0))
        for i in range(N):
            if F[i] <= 0.0:
                F[i] = 1e-12

        # Update layer 1 (always dB)
        _step_dB_layer(x1, F, ptr1, neigh1, w1, 0)
        # Update layer 2 (F is not recomputed after the layer-1 update)
        if layer2_is_Bd:
            _step_Bd_layer(x2, F, ptr2, neigh2, w2, 0)
        else:
            _step_dB_layer(x2, F, ptr2, neigh2, w2, 0)
        steps += 1

        # If we've crossed a time bin boundary, record R
        while bin_idx < n_bins and steps == time_bins[bin_idx]:
            R_arr[bin_idx] = _record_R(x1, x2, N)
            bin_idx += 1

    # If layers absorbed (or timed out) before some bins, leave them NaN.
    return R_arr, final_l1, final_l2, steps, t_l1, t_l2


@njit(cache=True)
def _sim_many_with_corr(
    ptr1, neigh1, w1, deg1,
    ptr2, neigh2, w2, deg2,
    init_C, init_M,
    b, c, r, delta,
    layer2_is_Bd,
    time_bins, n_runs, max_steps, seed,
):
    """Aggregate R(t) across runs.

    Arguments are those of _sim_one_with_corr plus n_runs (number of independent
    runs) and seed. The function is serial (no prange) and calls
    np.random.seed(seed) once before the first run, so its output is a
    deterministic function of the arguments (Fig C reproduces byte-for-byte).

    Per time bin k, the valid (non-NaN) R values are summed over
      all runs, runs whose layer 1 ended all-C, runs whose layer 1 ended all-D;
    runs in which layer 1 hit max_steps contribute to "all" only.
    The mean correlation plotted in Fig C is sum_R_*[k] / cnt_R_*[k].

    Returns (16-tuple, in this order):
      sum_R_all (n_bins,), cnt_R_all (n_bins,)        : over all runs
      sum_R_coop (n_bins,), cnt_R_coop (n_bins,)      : runs that fixate to all-C in L1
      sum_R_def  (n_bins,), cnt_R_def  (n_bins,)      : runs that fixate to all-D in L1
      n_C, n_D, n_to                                  : outcome counts in L1 (all-C, all-D, time-out)
      n_M, n_W, n_to_M                                : outcome counts in L2 (all-mutant, all-resident, time-out)
      sum_t_l1, n_t_l1                                : sum and count of L1 absorption times
      sum_t_l2, n_t_l2                                : sum and count of L2 absorption times
    cnt_* arrays are float64 counts.
    """
    np.random.seed(seed)
    n_bins = time_bins.shape[0]
    sum_R_all  = np.zeros(n_bins); cnt_R_all  = np.zeros(n_bins)
    sum_R_coop = np.zeros(n_bins); cnt_R_coop = np.zeros(n_bins)
    sum_R_def  = np.zeros(n_bins); cnt_R_def  = np.zeros(n_bins)

    n_C = 0; n_D = 0; n_to = 0
    n_M = 0; n_W = 0; n_to_M = 0
    sum_t_l1 = 0.0; sum_t_l2 = 0.0
    n_t_l1 = 0; n_t_l2 = 0

    for _ in range(n_runs):
        R_arr, fl1, fl2, steps, tl1, tl2 = _sim_one_with_corr(
            ptr1, neigh1, w1, deg1,
            ptr2, neigh2, w2, deg2,
            init_C, init_M,
            b, c, r, delta,
            layer2_is_Bd,
            time_bins, max_steps,
        )
        # Accumulate per-bin valid R
        # (runs are classified by their FINAL layer-1 outcome, known only now)
        for k in range(n_bins):
            v = R_arr[k]
            if not np.isnan(v):
                sum_R_all[k] += v
                cnt_R_all[k] += 1.0
                if fl1 == 1:
                    sum_R_coop[k] += v; cnt_R_coop[k] += 1.0
                elif fl1 == 0:
                    sum_R_def[k]  += v; cnt_R_def[k]  += 1.0

        if fl1 == 1: n_C += 1
        elif fl1 == 0: n_D += 1
        else: n_to += 1
        if fl2 == 1: n_M += 1
        elif fl2 == 0: n_W += 1
        else: n_to_M += 1
        if tl1 >= 0:
            sum_t_l1 += float(tl1); n_t_l1 += 1
        if tl2 >= 0:
            sum_t_l2 += float(tl2); n_t_l2 += 1

    return (sum_R_all, cnt_R_all, sum_R_coop, cnt_R_coop, sum_R_def, cnt_R_def,
            n_C, n_D, n_to, n_M, n_W, n_to_M,
            sum_t_l1, n_t_l1, sum_t_l2, n_t_l2)


def run_pair(G1, G2, b: float, group_label: str, pair_meta: dict,
             rule: str = RULE, n_runs: int = N_RUNS,
             time_bins: np.ndarray = TIME_BINS) -> pd.DataFrame:
    """LEGACY (not used for Fig C): simulate one (G1, G2) pair at the co-located IC.

    Parameters
    ----------
    G1, G2      : networkx graphs for layer 1 and layer 2
    b           : benefit (c = C = 1)
    group_label : "enhancing" / "suppressing" / "warmup", copied to the output
    pair_meta   : dict with g1_idx, g2_idx, bc_single, bc_two, slope (copied to
                  the output; g1_idx and g2_idx also set the seed
                  1000 + 13*g1_idx + 7*g2_idx)
    rule        : "dB-dB" or "dB-Bd"
    n_runs, time_bins : passed to _sim_many_with_corr (with INIT_C, INIT_M,
                  C, R=5, DELTA, MAX_STEPS from the module constants)

    Returns
    -------
    DataFrame with one row per time bin: mean/count of R over all, all-C and
    all-D runs, outcome counts n_C, n_D, n_to, n_M, n_W, n_to_M,
    rho_C = n_C/(n_C+n_D), rho_M = n_M/(n_M+n_W), mean absorption times.
    """
    ptr1, neigh1, w1, deg1 = pack_graph(G1)
    ptr2, neigh2, w2, deg2 = pack_graph(G2)
    layer2_is_Bd = (rule == "dB-Bd")
    seed = int(1000 + 13 * pair_meta["g1_idx"] + 7 * pair_meta["g2_idx"])
    out = _sim_many_with_corr(
        ptr1, neigh1, w1, deg1,
        ptr2, neigh2, w2, deg2,
        INIT_C, INIT_M,
        b, C, R, DELTA,
        layer2_is_Bd,
        time_bins, n_runs, MAX_STEPS, seed,
    )
    (sum_R_all, cnt_R_all, sum_R_coop, cnt_R_coop, sum_R_def, cnt_R_def,
     n_C, n_D, n_to, n_M, n_W, n_to_M,
     sum_t_l1, n_t_l1, sum_t_l2, n_t_l2) = out

    rows = []
    for k, t in enumerate(time_bins):
        rows.append({
            "g1_idx": pair_meta["g1_idx"],
            "g2_idx": pair_meta["g2_idx"],
            "group":  group_label,
            "bc_single": pair_meta["bc_single"],
            "bc_two":    pair_meta["bc_two"],
            "slope":     pair_meta["slope"],
            "b_over_c":  b,
            "t":         int(t),
            "mean_R_all":  (sum_R_all[k]  / cnt_R_all[k])  if cnt_R_all[k]  > 0 else np.nan,
            "n_R_all":     int(cnt_R_all[k]),
            "mean_R_coop": (sum_R_coop[k] / cnt_R_coop[k]) if cnt_R_coop[k] > 0 else np.nan,
            "n_R_coop":    int(cnt_R_coop[k]),
            "mean_R_def":  (sum_R_def[k]  / cnt_R_def[k])  if cnt_R_def[k]  > 0 else np.nan,
            "n_R_def":     int(cnt_R_def[k]),
            "n_C": n_C, "n_D": n_D, "n_to": n_to,
            "n_M": n_M, "n_W": n_W, "n_to_M": n_to_M,
            "rho_C":  n_C / max(n_C + n_D, 1),
            "rho_M":  n_M / max(n_M + n_W, 1),
            "mean_t_l1": (sum_t_l1 / n_t_l1) if n_t_l1 > 0 else np.nan,
            "mean_t_l2": (sum_t_l2 / n_t_l2) if n_t_l2 > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------
# Driver
# --------------------------------------------------------------
def b_for_pair(pair_meta: dict) -> float:
    """Pick b/c so cooperation is in the favored regime under bilayer.

    LEGACY (not used for Fig C; r3_strategy_coupling_ic.b_for applies the same
    rule). Use max(1.5 * bc_two, 2.0) if bc_two > 0; else use 5.0 (any positive
    b/c works when bc_two < 0). Returns b (with c = 1).
    """
    bct = pair_meta["bc_two"]
    if bct > 0:
        return float(max(1.5 * bct, 2.0))
    return 5.0


def main():
    """LEGACY pipeline (not used for any SI item).

    Picks pairs with pick_pairs() and writes OUT_PAIRS, warms up the JIT on a
    6-node cycle, then simulates every picked pair with run_pair (N_RUNS runs,
    b from b_for_pair) and appends the rows to OUT_RESULTS (deleted first).
    Graph indices refer to the connected 6-node graphs of nx.graph_atlas_g()
    in atlas order. Returns None.
    """
    print("# Picking pairs from exhaustive N=6 scan...", flush=True)
    pairs = pick_pairs()
    pairs.to_csv(OUT_PAIRS, index=False)
    print(f"# Saved picks to {OUT_PAIRS}")
    print(pairs[["group", "g1_idx", "g2_idx", "bc_single", "bc_two", "slope"]].to_string())

    graphs = [G for G in nx.graph_atlas_g()
              if G.number_of_nodes() == 6 and nx.is_connected(G)]

    print("\n# Warming up Numba JIT...", flush=True)
    Gw = nx.cycle_graph(6)
    _ = run_pair(Gw, Gw, 5.0, "warmup",
                  {"g1_idx": 0, "g2_idx": 0, "bc_single": 0.0,
                   "bc_two": 0.0, "slope": 0.0},
                  n_runs=200, time_bins=TIME_BINS[:3])
    print("# JIT warmed.", flush=True)

    if os.path.exists(OUT_RESULTS):
        os.remove(OUT_RESULTS)

    for _, row in pairs.iterrows():
        g1 = int(row["g1_idx"]); g2 = int(row["g2_idx"])
        G1, G2 = graphs[g1], graphs[g2]
        b = b_for_pair(row.to_dict())
        t0 = time.time()
        print(f"\n# === {row['group']} pair (g1={g1}, g2={g2}) "
              f"bc_single={row['bc_single']:.2f} bc_two={row['bc_two']:.2f} "
              f"slope={row['slope']:.2f}  b/c={b:.2f}  n_runs={N_RUNS} ===",
              flush=True)
        df = run_pair(G1, G2, b, row["group"], row.to_dict())
        dt = time.time() - t0
        rho = df["rho_C"].iloc[0]
        rhoM = df["rho_M"].iloc[0]
        print(f"  done in {dt:.1f}s  rho_C={rho:.4f}  rho_M={rhoM:.4f}  "
              f"mean_t_l1={df['mean_t_l1'].iloc[0]:.0f}  "
              f"mean_t_l2={df['mean_t_l2'].iloc[0]:.0f}", flush=True)
        df.to_csv(OUT_RESULTS, mode="a", index=False,
                   header=not os.path.exists(OUT_RESULTS) or
                   os.path.getsize(OUT_RESULTS) == 0)

    print(f"\n# All results saved to {OUT_RESULTS}")


# --------------------------------------------------------------
# Plot
# --------------------------------------------------------------
def plot(out_path: str = OUT_FIG):
    """LEGACY figure (not in the SI): 2 x 5 grid of R(t) curves.

    Reads OUT_RESULTS and OUT_PAIRS; top row = "enhancing" pairs, bottom row =
    "suppressing" pairs; each panel shows the mean correlation over all runs,
    runs ending all-C and runs ending all-D versus step t (log axis).
    Saves the figure to `out_path` (default r3_strategy_coupling.png).
    """
    df = pd.read_csv(OUT_RESULTS)
    pairs = pd.read_csv(OUT_PAIRS)

    enh = pairs[pairs["group"] == "enhancing"].reset_index(drop=True)
    sup = pairs[pairs["group"] == "suppressing"].reset_index(drop=True)

    fig, axes = plt.subplots(2, 5, figsize=(22, 10), squeeze=False, sharey=True)
    F_LABEL = 14
    F_TICK  = 12
    F_PANEL = 13
    F_LEG   = 13

    panel_labels = [chr(ord("a") + k) for k in range(10)]

    def _plot_panel(ax, sub, panel_label, row):
        """Draw the three R(t) curves of one pair and its title text on `ax`."""
        ax.plot(sub["t"], sub["mean_R_all"],
                marker="o", color="black", ms=5, lw=1.8,
                label="all runs")
        ax.plot(sub["t"], sub["mean_R_coop"],
                marker="s", color="#1f77b4", ms=5, lw=1.4,
                label=r"runs ending all-C")
        ax.plot(sub["t"], sub["mean_R_def"],
                marker="^", color="#d62728", ms=5, lw=1.4,
                label=r"runs ending all-D")
        ax.axhline(0.0, color="gray", lw=0.9, ls=":", alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlim(0.9, sub["t"].max() * 1.1)
        ax.set_ylim(-0.5, 1.05)
        ax.set_xlabel(r"step $t$", fontsize=F_LABEL)
        ax.tick_params(labelsize=F_TICK)
        ax.grid(alpha=0.25)
        rho = sub["rho_C"].iloc[0]
        title_main = (rf"({panel_label}) $(G_1,G_2) = "
                       rf"({int(row['g1_idx'])},{int(row['g2_idx'])})$")
        title_sub  = (rf"$(b/c)^{{*}}_{{\rm two}}={row['bc_two']:.2f}$, "
                       rf"slope$={row['slope']:+.2f}$, "
                       rf"$\rho_C={rho:.3f}$")
        ax.text(0.0, 1.085, title_main, transform=ax.transAxes,
                 ha="left", va="bottom", fontsize=F_PANEL)
        ax.text(0.0, 1.015, title_sub, transform=ax.transAxes,
                 ha="left", va="bottom", fontsize=F_PANEL - 1)

    for k, row in enh.iterrows():
        ax = axes[0, k]
        sub = df[(df["g1_idx"] == row["g1_idx"])
                 & (df["g2_idx"] == row["g2_idx"])].sort_values("t")
        _plot_panel(ax, sub, panel_labels[k], row)

    for k, row in sup.iterrows():
        ax = axes[1, k]
        sub = df[(df["g1_idx"] == row["g1_idx"])
                 & (df["g2_idx"] == row["g2_idx"])].sort_values("t")
        _plot_panel(ax, sub, panel_labels[5 + k], row)

    for ax in axes[:, 0]:
        ax.set_ylabel(r"$R(t)=\mathrm{Pearson}(x_1, x_2)$", fontsize=F_LABEL)

    fig.text(0.008, 0.745, "ENHANCING\n(slope $<$ 0)",
              rotation=90, ha="left", va="center", fontsize=14)
    fig.text(0.008, 0.30, "SUPPRESSING\n(slope $>$ 0)",
              rotation=90, ha="left", va="center", fontsize=14)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
                fontsize=F_LEG,
                bbox_to_anchor=(0.10, -0.005, 0.80, 0.04),
                mode="expand", frameon=False)
    fig.suptitle(r"Strategy-coupling correlation $R(t)=$ "
                  r"Pearson$(x_1, x_2)$ across nodes "
                  rf"(dB-dB, $r=5$, $\delta=0.02$, $N_{{\rm runs}}={N_RUNS:,}$, "
                  rf"init $x_1[0]=x_2[0]=1$)",
                  fontsize=15, y=1.005)
    fig.tight_layout(rect=[0.025, 0.04, 1, 0.985], h_pad=4.5, w_pad=1.2)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"# Figure saved: {out_path}")


if __name__ == "__main__":
    # Legacy pipeline only; Fig C is produced by r3_strategy_coupling_ic.py.
    import sys
    if "--plot-only" in sys.argv:
        plot()
    else:
        main()
        plot()
