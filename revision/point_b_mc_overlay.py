"""
Entry script for Fig I of S1 Text (Text L): fixation probability of a single
cooperator, weak-selection theory versus direct simulation, on two-layer ER
and BA networks with N = 100 and r = 5. Panels: (a) ER dB-dB, (b) ER dB-Bd,
(c) BA dB-dB, (d) BA dB-Bd.

What the script computes
------------------------
run_one(model, N, param, b_lo, b_hi, n_runs, delta, rule) builds one
two-layer network with get_network, puts the initial cooperator (layer 1)
and the initial mutant (layer 2) on node 0, and obtains theta_1, theta_2,
theta_3, phi_{2,0}, pi_init, (b/c)*_single and (b/c)*_two from
larger_networks_solver.bc_star_two_layer with r = R = 5. For
N_B_POINTS = 8 equidistant b/c values in [b_lo, b_hi] (c = 1) it evaluates

    rho_C_two_pred    = pi_init + delta * (1/N) [theta_2 + b (theta_1 - theta_3) - (r - 1) phi_{2,0}]
    rho_C_single_pred = pi_init + delta * (1/N) [theta_2 + b (theta_1 - theta_3)]

and estimates the same two quantities with weak_selection_sim_fast.run_sim,
with coupled=True (two-layer model) and coupled=False (layer 1 on its own).
pi_init = s_0 / sum_l s_l (s = degree in layer 1) is the fixation
probability of cooperation at delta = 0. run_all() calls run_one for every
entry of CONFIG and every delta in DELTAS under the rule RULE; plot() draws
the rows of the CSV.

Networks (get_network)
----------------------
* ER, param = p: each layer is nx.erdos_renyi_graph(N, p, seed=s), with the
  seeds s drawn from np.random.default_rng(10 N) for layer 1 and from
  np.random.default_rng(10 N + 7) for layer 2 (see the loops in
  get_network).
* BA, param = m: layer 1 is nx.barabasi_albert_graph(N, m, seed=10 N + 1)
  and layer 2 is nx.barabasi_albert_graph(N, m, seed=10 N + 8), both with
  networkx's default initial graph and without relabelling.
Node i of layer 1 and node i of layer 2 are the same individual. The same
network is used for both rules and both deltas. Fig I uses ER with p = 0.1
and BA with m = 2 at N = 100.

How to run (from the revision folder)
-------------------------------------
Plot from the stored CSV (the published figure is drawn from the stored CSV
by plot()):

    python point_b_mc_overlay.py --plot-only

This calls plot(ns=[100]); with matplotlib 3.10.0 and its default backend
on macOS ("macosx") it writes a file identical to
published_figures/point_b_mc_overlay.png.

Full run (simulations, theory, CSV and JSON, then plot(ns=[100])):

    NUMBA_NUM_THREADS=4 python point_b_mc_overlay.py

The full run replaces point_b_mc_results.csv with the rows described under
"Stored CSV and the constants below", so its plot has two panels (ER and BA
at N = 100, dB-dB, delta = 0.02).

No environment variables are required; NUMBA_NUM_THREADS only sets the
number of numba threads.

Stored CSV and the constants below
----------------------------------
point_b_mc_results.csv has 128 rows in 16 blocks of 8 b/c values: at
delta = 0.02 and at delta = 0.2, the dB-dB rows of BA N = 15 (b/c in [4, 8]),
ER N = 30 with p = 0.2 ([7, 11]), BA N = 30 ([3.5, 6.5]) and BA N = 50
([3, 6]), and the dB-dB and dB-Bd rows of ER N = 100 (dB-dB [10, 15],
dB-Bd [11, 16]) and BA N = 100 ([3.5, 5] for both rules). plot(ns=[100])
draws the 64 rows with N = 100.
run_all() with CONFIG, DELTAS and RULE as set below deletes the CSV and
writes only the dB-dB rows at delta = 0.02 (6 networks, 48 rows; BA N = 100
with the window [3, 5]). The other blocks of the stored CSV (delta = 0.2,
rule 'dB-Bd', the BA N = 100 window [3.5, 5] and other run counts) are not
written by run_all() with these constants; they were added to the CSV by
separate runs. A block with other settings is computed with run_one and
appended to the CSV with, for example,

    df, meta = run_one("BA", 100, 2, 3.5, 5.0, 50_000, delta=0.2, rule="dB-Bd")
    df.to_csv(OUT_CSV, mode="a", index=False, header=False)

Inputs
------
Full run: none; the networks are generated in the script.
--plot-only: point_b_mc_results.csv, read from the working directory.
Both need legend_utils.py (draw_uniform_legend, imported inside plot()) on
the Python path, i.e. in this folder when the script is run as above.

Outputs (written to the working directory)
------------------------------------------
point_b_mc_results.csv
    One row per (network, rule, delta, b/c). Columns:
    model, N, param, b_over_c, r, delta, rule   parameters of the row
    pi_init                                     s_0 / sum_l s_l in layer 1
    bc_star_single, bc_star_two                 thresholds
    rho_C_two_sim, se_two                       two-layer simulation: estimate and binomial standard error
    rho_C_single_sim, se_single                 one-layer simulation: estimate and binomial standard error
    rho_C_two_pred, rho_C_single_pred           first-order predictions (see above)
point_b_mc_networks.json
    Edge lists, initial nodes, r and thresholds of each network (run_all).
point_b_mc_overlay.png
    Fig I: 2 x 2 panels, 180 dpi.
numba also writes compiled-code cache files into __pycache__/ next to
weak_selection_sim_fast.py.

Runtime
-------
--plot-only: about 2 to 4 s. A bc_star_two_layer call at N = 100 takes about
1 to 2.5 min and 0.5 to 0.9 GB of memory; calls with N <= 50 take under 1 s.
Full run with the constants below: about 15 to 17 min with
NUMBA_NUM_THREADS=4 (estimated from short pilot runs: about 6 min of
simulation for N < 100, about 7 min of simulation for N = 100 and about
4 min for the two N = 100 solver calls).

Reproducibility
---------------
The theory columns are deterministic. The simulation columns are
reproducible in distribution only, because weak_selection_sim_fast runs the
replicates in a numba prange loop whose worker threads have random streams
that the seed does not fix (see the docstring of weak_selection_sim_fast.py).

Dependencies
------------
numpy, networkx, pandas, scipy, matplotlib, numba (through
weak_selection_sim_fast.py), and the local modules
    larger_networks_solver.py   bc_star_two_layer
    weak_selection_sim_fast.py  run_sim
    legend_utils.py             draw_uniform_legend (plot() only)
"""
from __future__ import annotations
import os, time, json
import warnings, scipy.sparse as sp
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from larger_networks_solver import bc_star_two_layer
from weak_selection_sim_fast import run_sim


# Seed values: ER layers 10*N + 0 and 10*N + 7 (seeds of numpy Generators
# that supply the networkx seeds), BA layers 10*N + 1 and 10*N + 8 (networkx
# seeds).
def get_network(model: str, N: int, param):
    """Return the two layers (G1, G2) of one two-layer network.

    Parameters
    ----------
    model : 'ER' or 'BA' (any other value raises ValueError).
    N     : number of nodes.
    param : edge probability p for 'ER'; number m of edges added per new
            node for 'BA'.

    'ER': each layer is nx.erdos_renyi_graph(N, p, seed=s). For layer 1 the
    seeds s are drawn from np.random.default_rng(10 N), for layer 2 from
    np.random.default_rng(10 N + 7); each loop below draws up to 500 seeds
    and keeps the first graph that passes the test in the loop (otherwise
    RuntimeError).
    'BA': G1 = nx.barabasi_albert_graph(N, m, seed=10 N + 1) and
    G2 = nx.barabasi_albert_graph(N, m, seed=10 N + 8), networkx's default
    initial graph, no relabelling.

    The result depends only on (model, N, param), so repeated calls return
    the same network. Nodes are 0..N-1 in both layers; node i of G1 and node
    i of G2 are the same individual.
    """
    if model == "ER":
        # Layer 1: networkx seeds drawn from default_rng(10 N).
        rng = np.random.default_rng(10 * N)
        for _ in range(500):
            G1 = nx.erdos_renyi_graph(N, param,
                                       seed=int(rng.integers(0, 2**31 - 1)))
            if nx.is_connected(G1):
                break
        else:
            raise RuntimeError(f"ER N={N} p={param} too sparse to connect")
        # Layer 2: networkx seeds drawn from default_rng(10 N + 7).
        rng = np.random.default_rng(10 * N + 7)
        for _ in range(500):
            G2 = nx.erdos_renyi_graph(N, param,
                                       seed=int(rng.integers(0, 2**31 - 1)))
            if nx.is_connected(G2):
                break
        else:
            raise RuntimeError(f"ER N={N} p={param} too sparse to connect")
        return G1, G2
    elif model == "BA":
        seed1, seed2 = 10 * N + 1, 10 * N + 8
        return (nx.barabasi_albert_graph(N, int(param), seed=seed1),
                nx.barabasi_albert_graph(N, int(param), seed=seed2))
    raise ValueError(model)


# Networks and b/c windows looped over by run_all(). Each window contains
# (b/c)*_two and (b/c)*_single of the network at r = 5 under dB-dB.
CONFIG = [
    # (model, N, param, (b_lo, b_hi), n_runs per b/c value).
    # param = m for BA, p for ER. Comments: (b/c)*_two, (b/c)*_single
    # under dB-dB.
    ("BA", 15, 2,   (4.0, 8.0),  300_000),       # (b/c)*: 5.27, 7.45
    ("ER", 30, 0.2, (7.0, 11.0), 200_000),       # (b/c)*: 8.38, 9.84
    ("BA", 30, 2,   (3.5, 6.5),  200_000),       # (b/c)*: 4.50, 5.52
    ("BA", 50, 2,   (3.0, 6.0),  100_000),       # (b/c)*: 4.32, 4.96
    ("BA", 100, 2,   (3.0,  5.0), 50_000),       # the stored BA N = 100 rows use [3.5, 5.0]
    ("ER", 100, 0.1, (10.0, 15.0), 50_000),      # dB-dB window; the stored dB-Bd rows use [11, 16]
]
N_B_POINTS_DEFAULT = 8         # not used
RULE = "dB-dB"                 # updating rule used by run_all()
R = 5.0                        # mutant fitness r in layer 2
DELTAS = [0.02]                # selection strengths looped over by run_all()
N_B_POINTS = 8                 # number of equidistant b/c values per window

# Output files, relative to the working directory. plot() also reads OUT_CSV.
OUT_CSV = "point_b_mc_results.csv"
OUT_FIG = "point_b_mc_overlay.png"
OUT_NETWORKS = "point_b_mc_networks.json"


def run_one(model, N, param, b_lo, b_hi, n_runs, delta=0.02, rule=RULE):
    """Theory and simulation for one network, rule and delta.

    Parameters
    ----------
    model, N, param : network, passed to get_network.
    b_lo, b_hi      : ends of the b/c window; N_B_POINTS equidistant values
                      (np.linspace) are used, with c = 1.
    n_runs          : simulation runs per b/c value, for each of the
                      two-layer and the one-layer simulation.
    delta           : selection strength.
    rule            : 'dB-dB' or 'dB-Bd'.

    The initial cooperator and the initial mutant are on node 0, and r = R.
    The simulation seed of a b/c value is
    int(11 + 100 b + 10 N + 1000 delta), plus 1 under dB-Bd; the one-layer
    simulation uses that seed plus 1.

    Returns
    -------
    (df, meta)
    df   : pandas.DataFrame with one row per b/c value and the CSV columns
           listed in the module docstring.
    meta : dict with model, N, param, rule, the edge lists of both layers
           (edges_L1, edges_L2), init_C, init_M, r, bc_star_single and
           bc_star_two (the record written to OUT_NETWORKS by run_all).
    """
    G1, G2 = get_network(model, N, param)
    init_C = init_M = 0
    # theta_n, phi_{n,m}, thresholds and pi_init = pi1[init_C] for this
    # network, rule and initial condition (independent of b and delta).
    s = bc_star_two_layer(G1, G2, init_C, init_M, rule, r=R)
    th1, th2, th3, phi20 = s["theta1"], s["theta2"], s["theta3"], s["phi20"]
    bc_single, bc_two = s["bc_star_single"], s["bc_star_two"]
    pi_init = s["pi_init"]
    actual_N = G1.number_of_nodes()

    b_grid = np.linspace(b_lo, b_hi, N_B_POINTS)
    rows = []
    for b in b_grid:
        # Seed of this b/c value; int() truncates the float sum.
        seed = int(11 + b * 100 + N * 10 + delta * 1000
                    + (1 if rule == "dB-Bd" else 0))
        sim_two    = run_sim(G1, G2, init_C, init_M, b, 1.0, R, delta,
                              rule, coupled=True,  n_runs=n_runs, seed=seed)
        sim_single = run_sim(G1, G2, init_C, init_M, b, 1.0, R, delta,
                              rule, coupled=False, n_runs=n_runs, seed=seed + 1)
        # First-order weak-selection predictions with c = 1:
        # d rho_C / d delta = (1/N) [theta_2 + b (theta_1 - theta_3) - (r - 1) phi_{2,0}]
        # (two-layer) and the same without the phi term (one-layer).
        drho_two = (1.0 / actual_N) * (
            th2 + b * (th1 - th3) - (R - 1.0) * phi20
        )
        drho_single = (1.0 / actual_N) * (th2 + b * (th1 - th3))
        rho_pred_two = pi_init + delta * drho_two
        rho_pred_single = pi_init + delta * drho_single
        rows.append({
            "model": model, "N": actual_N, "param": param,
            "b_over_c": b, "r": R, "delta": delta, "rule": rule,
            "pi_init": pi_init,
            "bc_star_single": bc_single, "bc_star_two": bc_two,
            "rho_C_two_sim": sim_two["rho_C"], "se_two": sim_two["se_rho_C"],
            "rho_C_single_sim": sim_single["rho_C"], "se_single": sim_single["se_rho_C"],
            "rho_C_two_pred": rho_pred_two, "rho_C_single_pred": rho_pred_single,
        })
    return pd.DataFrame(rows), {
        "model": model, "N": actual_N, "param": param, "rule": rule,
        "edges_L1": list(G1.edges()), "edges_L2": list(G2.edges()),
        "init_C": init_C, "init_M": init_M, "r": R,
        "bc_star_single": bc_single, "bc_star_two": bc_two,
    }


def run_all():
    """Run every CONFIG entry at every delta in DELTAS under RULE.

    Compiles the numba kernels with one short ring run (result discarded),
    deletes OUT_CSV if it exists, calls run_one for each (CONFIG entry,
    delta) and appends each returned block to OUT_CSV, then writes the
    network records (one per CONFIG entry) to OUT_NETWORKS. With the
    constants as set in this file the CSV gets 6 x 8 = 48 rows (dB-dB,
    delta = 0.02). Returns None.
    """
    # Compile the numba kernels once; output discarded.
    print("# Warming up Numba JIT...", flush=True)
    Gw = nx.cycle_graph(10)
    _ = run_sim(Gw, Gw, 0, 1, 3.0, 1.0, 2.0, 0.02, RULE, True, 200, seed=0)
    print("# JIT warmed.", flush=True)

    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
    networks = []
    for (model, N, param, (b_lo, b_hi), n_runs) in CONFIG:
        for delta in DELTAS:
            t0 = time.time()
            print(f"# === {model} N={N} param={param}  b/c∈[{b_lo}, {b_hi}]  "
                  f"r={R}  delta={delta}  n_runs={n_runs} ===", flush=True)
            df, net_meta = run_one(model, N, param, b_lo, b_hi, n_runs, delta)
            dt = time.time() - t0
            print(f"  done in {dt:.1f}s   (b/c)*_single={net_meta['bc_star_single']:.2f},"
                  f" (b/c)*_two={net_meta['bc_star_two']:.2f}", flush=True)
            # Append this block; the header is written only into a new or
            # empty file.
            df.to_csv(OUT_CSV, mode="a", index=False,
                      header=not os.path.exists(OUT_CSV) or
                      os.path.getsize(OUT_CSV) == 0)
            # One network record per CONFIG entry (the network does not
            # depend on delta).
            if delta == DELTAS[0]:
                networks.append(net_meta)
    with open(OUT_NETWORKS, "w") as f:
        json.dump(networks, f, indent=2)
    print(f"\n# Networks saved to {OUT_NETWORKS}", flush=True)


# Marker and colour of each delta in plot().
DELTA_STYLES = {
    0.02: dict(marker="s", color="#1f77b4"),
    0.20: dict(marker="o", color="#d62728"),
}


def plot(out_path: str = OUT_FIG, deltas=None, ns=None, rules=None):
    """Draw theory versus simulation panels from OUT_CSV and save them.

    Parameters
    ----------
    out_path : output PNG path (default OUT_FIG); saved at 180 dpi.
    deltas   : list of delta values to keep; None keeps every delta in the CSV.
    ns       : list of network sizes N to keep; None keeps all. The
               published Fig I is plot(ns=[100]).
    rules    : list of rules to keep; None keeps all.

    Reads OUT_CSV from the working directory. There is one panel per
    (model, N, rule) when the kept rows contain more than one rule, and one
    per (model, N) otherwise; panels are ordered by N, then ER before BA,
    then dB-dB before dB-Bd, and laid out in two columns. For N = 100 this
    gives (a) ER dB-dB, (b) ER dB-Bd, (c) BA dB-dB, (d) BA dB-Bd. At most
    six panels can be drawn (six panel letters), so on the full stored CSV
    a size filter such as ns=[100] is needed.

    Each panel shows, divided by pi_init (so that 1, the dotted gray line,
    is the value at delta = 0), for each delta (marker and colour from
    DELTA_STYLES): the one-layer theory (dashed), the two-layer theory
    (solid), the one-layer simulation (open markers) and the two-layer
    simulation (filled markers), with error bars of one standard error.
    Vertical lines mark (b/c)*_single (green) and (b/c)*_two (purple). The
    legend is drawn by legend_utils.draw_uniform_legend; it has four columns
    when delta = 0.2 is present and three otherwise.
    """
    df = pd.read_csv(OUT_CSV)
    if deltas is not None:
        df = df[df["delta"].isin(deltas)]
    if ns is not None:
        df = df[df["N"].isin(ns)]
    if rules is not None:
        df = df[df["rule"].isin(rules)]
    rule_order = {"dB-dB": 0, "dB-Bd": 1}
    rules_present = sorted(df["rule"].unique(),
                           key=lambda r: rule_order.get(r, 99))
    # Panel order within each N: ER first (panels a, b), then BA (c, d).
    model_order = {"ER": 0, "BA": 1}
    if len(rules_present) > 1:
        keys = sorted(set(zip(df["model"], df["N"], df["rule"])),
                      key=lambda mnr: (mnr[1],
                                        model_order.get(mnr[0], 99),
                                        rule_order.get(mnr[2], 99)))
    else:
        keys = sorted(set(zip(df["model"], df["N"])),
                      key=lambda mn: (mn[1], model_order.get(mn[0], 99)))
    n = len(keys)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5.5 * rows),
                              squeeze=False)

    deltas_present = sorted(df["delta"].unique())
    bc_color_single = "#2ca02c"   # green
    bc_color_two    = "#9467bd"   # purple
    panel_labels = ["a", "b", "c", "d", "e", "f"]

    F_LABEL = 16
    F_TICK  = 14
    F_PANEL = 16
    F_LEG   = 11   # smaller than other text so that four equally spaced
                    # legend columns fit without overlap

    # y-axis range and ticks by model (with the N = 100 panels: top row ER,
    # bottom row BA).
    yrange_by_model = {
        "ER": (0.0, 3.0,  [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        "BA": (0.0, 1.8,  [0.0, 0.5, 1.0, 1.5]),
    }
    # Common x range of the BA N = 100 panels: the b/c window [3.5, 5.0] of
    # the stored rows plus a small margin so that the end markers are not
    # clipped.
    ba_n100_xrange = (3.46, 5.04)

    for idx, key in enumerate(keys):
        r_, c_ = idx // cols, idx % cols
        ax = axes[r_, c_]
        if len(rules_present) > 1:
            model, N, rule_ = key
            sub = df[(df["model"] == model) & (df["N"] == N)
                     & (df["rule"] == rule_)]
        else:
            model, N = key
            rule_ = rules_present[0] if rules_present else RULE
            sub = df[(df["model"] == model) & (df["N"] == N)
                     & (df["rule"] == rule_)]
        if sub.empty:
            ax.set_visible(False); continue
        # pi_init and the thresholds are the same in every row of a panel.
        pi0 = sub["pi_init"].iloc[0]
        bc_s = sub["bc_star_single"].iloc[0]
        bc_t = sub["bc_star_two"].iloc[0]

        for delta in sorted(sub["delta"].unique()):
            grp = sub[sub["delta"] == delta].sort_values("b_over_c")
            style = DELTA_STYLES.get(delta,
                                      dict(marker="^", color="#2ca02c"))
            color = style["color"]
            marker = style["marker"]
            ax.plot(grp["b_over_c"], grp["rho_C_single_pred"] / pi0,
                    ls="--", color=color, alpha=0.55, lw=2.0)
            ax.plot(grp["b_over_c"], grp["rho_C_two_pred"] / pi0,
                    ls="-",  color=color, alpha=0.9,  lw=2.2)
            ax.errorbar(grp["b_over_c"],
                        grp["rho_C_single_sim"] / pi0,
                        yerr=grp["se_single"] / pi0,
                        fmt=marker, mfc="white", mec=color, color=color,
                        ms=8, capsize=2.5, lw=1.0, zorder=4)
            ax.errorbar(grp["b_over_c"],
                        grp["rho_C_two_sim"] / pi0,
                        yerr=grp["se_two"] / pi0,
                        fmt=marker, mfc=color, mec=color, color=color,
                        ms=8, capsize=2.5, lw=1.0, zorder=4)
        ax.axhline(1.0, color="gray", lw=0.9, ls=":", alpha=0.7)
        if np.isfinite(bc_s):
            ax.axvline(bc_s, color=bc_color_single, lw=1.8, ls="-", alpha=0.85)
        if np.isfinite(bc_t):
            ax.axvline(bc_t, color=bc_color_two, lw=1.8, ls="-", alpha=0.85)
        ax.set_xlabel(r"$b/c$", fontsize=F_LABEL)
        ax.set_ylabel(r"$\rho_{\rm C} / \rho_{\rm C}^{\circ}$", fontsize=F_LABEL)

        # y-axis range and ticks of this model
        if model in yrange_by_model:
            ymin, ymax, yticks = yrange_by_model[model]
            ax.set_ylim(ymin, ymax)
            ax.set_yticks(yticks)
        # Shared x-range for BA at N=100
        if model == "BA" and N == 100:
            ax.set_xlim(*ba_n100_xrange)

        ax.tick_params(labelsize=F_TICK)
        ax.grid(alpha=0.25)

        # Panel label "(a) ER (N=100, r=5), dB-dB" in regular weight,
        # left-aligned above the axes.
        r_str = f"r={int(R)}" if float(R).is_integer() else f"r={R}"
        title = rf"({panel_labels[idx]}) {model} ($N={N}$, ${r_str}$), {rule_}"
        ax.set_title("")
        ax.text(0.0, 1.04, title, transform=ax.transAxes,
                 ha="left", va="bottom", fontsize=F_PANEL)

    # The bottom legend is drawn on its own axes by
    # legend_utils.draw_uniform_legend, which places the legend markers at
    # equally spaced x positions (fig.legend does not give equal column
    # widths). legend_utils.py must be on the Python path (this folder).
    from legend_utils import draw_uniform_legend
    # Legend rows: row 1 one-layer entries, row 2 two-layer entries. The
    # delta = 0.2 marker entries are included only if delta = 0.2 is among
    # the plotted rows.
    sim_delta_02 = DELTA_STYLES.get(0.02, dict(marker="s", color="#1f77b4"))
    sim_delta_2  = DELTA_STYLES.get(0.20, dict(marker="o", color="#d62728"))
    has_delta_2  = (0.20 in deltas_present)
    legend_rows = [
        [
            dict(kind="marker", marker=sim_delta_02["marker"],
                 mfc="white", mec=sim_delta_02["color"],
                 label=r"simulation, one-layer, $\delta=0.02$"),
        ] + ([
            dict(kind="marker", marker=sim_delta_2["marker"],
                 mfc="white", mec=sim_delta_2["color"],
                 label=r"simulation, one-layer, $\delta=0.2$"),
        ] if has_delta_2 else []) + [
            dict(kind="line", color="gray",  linestyle="--", linewidth=2.2,
                 label=r"weak-selection theory, one-layer"),
            dict(kind="line", color=bc_color_single, linestyle="-", linewidth=2.4,
                 label=r"$(b/c)^{*}$, one-layer"),
        ],
        [
            dict(kind="marker", marker=sim_delta_02["marker"],
                 mfc=sim_delta_02["color"], mec=sim_delta_02["color"],
                 label=r"simulation, two-layer, $\delta=0.02$"),
        ] + ([
            dict(kind="marker", marker=sim_delta_2["marker"],
                 mfc=sim_delta_2["color"], mec=sim_delta_2["color"],
                 label=r"simulation, two-layer, $\delta=0.2$"),
        ] if has_delta_2 else []) + [
            dict(kind="line", color="black", linestyle="-",  linewidth=2.4,
                 label=r"weak-selection theory, two-layer"),
            dict(kind="line", color=bc_color_two, linestyle="-", linewidth=2.4,
                 label=r"$(b/c)^{*}$, two-layer"),
        ],
    ]
    ncols_legend = 4 if has_delta_2 else 3
    # Reserve the bottom 8% of the figure for the legend.
    fig.tight_layout(rect=[0, 0.08, 1, 0.99], h_pad=3.0)
    draw_uniform_legend(fig, legend_rows, ncols=ncols_legend,
                         bbox=(0.02, 0.0, 0.96, 0.07),
                         fontsize=F_LEG, marker_size=10)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"# Figure saved: {out_path}  (deltas: {deltas_present})")


if __name__ == "__main__":
    import sys
    # --plot-only: draw Fig I (N = 100 panels) from the stored CSV.
    # Otherwise: run_all() (rewrites the CSV), then the same plot.
    if "--plot-only" in sys.argv:
        plot(ns=[100])
    else:
        run_all()
        plot(ns=[100])
