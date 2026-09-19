"""
Entry script for Fig G of S1 Text (Text L): fixation probability of a single
cooperator, weak-selection theory versus direct simulation, on a two-layer
ring with N = 10 and a two-layer BA network with N = 15.

What the script computes
------------------------
For each network, each updating rule ('dB-dB', 'dB-Bd'), each selection
strength delta in DELTAS = [0.02, 0.2] and each b/c value of the network's
grid (c = 1):

1. Theory. theta_1, theta_2, theta_3 and phi_{2,0} of the network and initial
   condition come from weak_selection_validation.analytical_summary. The
   first-order prediction of the fixation probability of cooperation is

       rho_C_two_pred    = pi_init + delta * (1/N) [c theta_2 + b (theta_1 - theta_3) - (r - 1) phi_{2,0}]
       rho_C_single_pred = pi_init + delta * (1/N) [c theta_2 + b (theta_1 - theta_3)]

   for the two-layer network and for layer 1 on its own.
   pi_init = s_v / sum_l s_l, with s the degree in layer 1 and v the node of
   the initial cooperator, is the fixation probability of cooperation at
   delta = 0 (the normalizing quantity rho_C^circ on the vertical axis of
   Fig G). The thresholds (b/c)*_single and (b/c)*_two are those of Eq (5)
   of the main text.
2. Simulation. weak_selection_sim_fast.run_sim is called with coupled=True
   (two-layer model) and with coupled=False (layer 1 on its own; layer 2
   neither updates nor enters the fecundity), with n_runs runs each.

Networks and parameters (the list NETWORKS below)
-------------------------------------------------
* "Ring (N=10)": both layers nx.cycle_graph(10). Initial cooperator on node 0
  of layer 1, initial mutant on node 1 of layer 2. r = 2.5,
  b/c = np.linspace(2.30, 2.80, 10), 10^6 runs per b/c value at delta = 0.02
  and 2 x 10^5 at delta = 0.2.
* "BA-BA (N=15, seed=4)": layer 1 nx.barabasi_albert_graph(15, 2, seed=104),
  layer 2 nx.barabasi_albert_graph(15, 2, seed=204), both with networkx's
  default initial graph and without relabelling. Initial cooperator and
  initial mutant on node 0. r = 5, b/c = np.linspace(5, 12, 10), 5 x 10^5 runs
  per b/c value at delta = 0.02 and 10^5 at delta = 0.2. The string is the
  network label written to the CSV; "seed=4" stands for the layer seeds 104
  and 204.
In both networks node i of layer 1 and node i of layer 2 are the same
individual. The simulation seed of a row is
17 + round(100 b) + int(1000 delta), plus 1 under dB-Bd; the one-layer
simulation of the same row uses that seed plus 1.

How to run (from the revision folder)
-------------------------------------
Redraw Fig G from the stored CSV (the published figure is drawn from the
stored CSV by plot()):

    python -c "import pandas as pd, run_ws_validation_ring_baba as m; m.plot(pd.read_csv(m.OUT_CSV))"

With matplotlib 3.10.0 and its default backend on macOS ("macosx") this
writes a file identical to published_figures/ws_validation_ring_baba.png.

Full run (simulations, theory, CSV, JSON and PNG):

    NUMBA_NUM_THREADS=4 python run_ws_validation_ring_baba.py

The script takes no command-line arguments and needs no environment
variables; NUMBA_NUM_THREADS only sets the number of numba threads.

Inputs
------
Full run: none; the networks are generated in the script.
Redraw: ws_validation_ring_baba_results.csv in the working directory.
Both need legend_utils.py (draw_uniform_legend, imported inside plot()) on
the Python path, i.e. in this folder when the commands above are run from it.

Outputs (written to the working directory)
------------------------------------------
ws_validation_ring_baba_results.csv
    80 rows = 2 networks x 2 rules x 2 deltas x 10 b/c values. run_all()
    deletes an existing file first and then appends one block per
    (network, rule, delta). Columns:
    network, rule, delta, b_over_c, r, N        parameters of the row
    pi_init                                     s_v / sum_l s_l (see above)
    theta1, theta2, theta3, phi20               theory inputs
    bc_star_single, bc_star_two                 thresholds
    rho_C_two_sim, se_two                       two-layer simulation: estimate and binomial standard error
    rho_C_single_sim, se_single                 one-layer simulation: estimate and binomial standard error
    rho_C_two_pred, rho_C_single_pred           first-order predictions (see above)
    completed_two, completed_single             number of runs in which layer 1 reached all-C or all-D
ws_validation_ring_baba.png
    Fig G: 2 x 2 panels, rows = networks, columns = rules, 180 dpi.
ws_validation_ring_baba_networks.json
    Edge lists of both layers, initial nodes, r and b/c grid of each network.
numba also writes compiled-code cache files into __pycache__/ next to
weak_selection_sim_fast.py.

Runtime
-------
Redraw: about 2 to 3 s. Theory: well under 1 s per analytical_summary call.
Full run: about 4 min with NUMBA_NUM_THREADS=4 (estimated from a short pilot
run; most of the time is spent in the 10^6-run ring simulations and the
5 x 10^5-run BA simulations).

Reproducibility
---------------
The theory columns are deterministic. The simulation columns are
reproducible in distribution only: weak_selection_sim_fast.simulate_many
runs the replicates in a numba prange loop, and with more than one numba
thread the random streams of the worker threads are not fixed by the seed
(see the docstring of weak_selection_sim_fast.py).

Dependencies
------------
numpy, networkx, pandas, scipy, matplotlib, numba (through
weak_selection_sim_fast.py), and the local modules
    weak_selection_validation.py  analytical_summary, drho_C_ddelta_singlelayer,
                                  drho_C_ddelta_twolayer
    weak_selection_sim_fast.py    run_sim
    legend_utils.py               draw_uniform_legend (plot() only)
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

from weak_selection_validation import (
    analytical_summary,
    drho_C_ddelta_singlelayer, drho_C_ddelta_twolayer,
)
from weak_selection_sim_fast import run_sim


# --------------------------------------------------------------
# Networks (generated from fixed seeds)
# --------------------------------------------------------------
def make_ring_n10():
    """Return (G1, G2), the two layers of the ring network.

    Both layers are nx.cycle_graph(10): nodes 0..9, node i adjacent to
    nodes i - 1 and i + 1 (mod 10).
    """
    return nx.cycle_graph(10), nx.cycle_graph(10)


def make_ba_n15():
    """Return (G1, G2), the two layers of the BA network with N = 15.

    G1 = nx.barabasi_albert_graph(15, 2, seed=104) and
    G2 = nx.barabasi_albert_graph(15, 2, seed=204): preferential attachment
    with m = 2 edges per new node, starting from networkx's default initial
    graph. Layer 2 is not relabelled, so node i of G1 and node i of G2 are the
    same individual.
    """
    G1 = nx.barabasi_albert_graph(15, 2, seed=104)
    G2 = nx.barabasi_albert_graph(15, 2, seed=204)
    return G1, G2


# --------------------------------------------------------------
# Experiment specification
# --------------------------------------------------------------
# One dict per network:
#   name               network label written to the CSV and used by plot()
#   loader             function returning (G1, G2)
#   init_C, init_M     node of the initial cooperator (layer 1) and of the
#                      initial mutant (layer 2), in sorted node order
#   r                  mutant fitness in layer 2
#   b_grid             b/c values (c = 1)
#   n_runs_low_delta   runs per b/c value at delta = 0.02
#   n_runs_high_delta  runs per b/c value at delta = 0.2
NETWORKS = [
    {
        "name": "Ring (N=10)",
        "loader": make_ring_n10,
        "init_C": 0, "init_M": 1,    # cooperator on node 0, mutant on its neighbour node 1
        "r": 2.5,
        "b_grid": np.linspace(2.30, 2.80, 10),    # 10 equidistant values
        "n_runs_low_delta":  1_000_000,
        "n_runs_high_delta":   200_000,
    },
    {
        "name": "BA-BA (N=15, seed=4)",
        "loader": make_ba_n15,
        "init_C": 0, "init_M": 0,    # cooperator and mutant on the same node 0
        "r": 5.0,
        "b_grid": np.linspace(5.0, 12.0, 10),    # contains (b/c)*_single and (b/c)*_two
        "n_runs_low_delta":  500_000,
        "n_runs_high_delta": 100_000,
    },
]
DELTAS = [0.02, 0.20]            # selection strengths
RULES = ["dB-dB", "dB-Bd"]       # updating rules (layer 1 always dB)

# Output files, relative to the working directory.
OUT_CSV = "ws_validation_ring_baba_results.csv"
OUT_FIG = "ws_validation_ring_baba.png"
OUT_NETWORKS = "ws_validation_ring_baba_networks.json"


# --------------------------------------------------------------
# Run one (network, rule, delta) panel
# --------------------------------------------------------------
def run_panel(spec, rule, delta, b_grid, n_runs):
    """Theory and simulation for one (network, rule, delta) block of Fig G.

    Parameters
    ----------
    spec   : one entry of NETWORKS (network, initial nodes, r).
    rule   : 'dB-dB' or 'dB-Bd'.
    delta  : selection strength.
    b_grid : b/c values; c = 1, so these are the values of b.
    n_runs : number of simulation runs per b/c value, for each of the two
             simulations (two-layer and one-layer).

    Returns
    -------
    pandas.DataFrame with one row per b/c value and the columns listed in
    the module docstring. theta_n, phi_{2,0}, pi_init and the thresholds
    depend only on the network, the rule and the initial condition, so they
    are the same in every row of the block.
    """
    G1, G2 = spec["loader"]()
    init_C, init_M, r = spec["init_C"], spec["init_M"], spec["r"]
    N = G1.number_of_nodes()

    # pi_init = s_v / sum_l s_l for the initial cooperator's node v in layer 1:
    # the dB reproductive value of v, i.e. the fixation probability of
    # cooperation at delta = 0.
    A1 = nx.to_numpy_array(G1, nodelist=sorted(G1.nodes()))
    deg1 = A1.sum(axis=1)
    pi_init = float(deg1[init_C] / deg1.sum())

    s = analytical_summary(G1, G2, init_C, init_M, rule, r=r)
    th1, th2, th3, phi20 = s["theta1"], s["theta2"], s["theta3"], s["phi20"]
    bc_single, bc_two = s["bc_star_single"], s["bc_star_two"]

    rows = []
    for b in b_grid:
        # First-order weak-selection prediction, c = 1:
        # rho_C = pi_init + delta * d rho_C / d delta at delta = 0.
        drho_two    = drho_C_ddelta_twolayer (b, 1.0, r, th1, th2, th3, phi20, N)
        drho_single = drho_C_ddelta_singlelayer(b, 1.0, th1, th2, th3, N)
        rho_pred_two    = pi_init + delta * drho_two
        rho_pred_single = pi_init + delta * drho_single

        # Simulation seed of this row; the one-layer run below uses seed + 1.
        seed = 17 + int(round(b * 100)) + int(delta * 1000) + (1 if rule == "dB-Bd" else 0)
        sim_two    = run_sim(G1, G2, init_C, init_M, b, 1.0, r, delta,
                              rule, coupled=True,  n_runs=n_runs, seed=seed)
        sim_single = run_sim(G1, G2, init_C, init_M, b, 1.0, r, delta,
                              rule, coupled=False, n_runs=n_runs, seed=seed + 1)
        rows.append({
            "network": spec["name"], "rule": rule, "delta": delta,
            "b_over_c": b, "r": r, "N": N, "pi_init": pi_init,
            "theta1": th1, "theta2": th2, "theta3": th3, "phi20": phi20,
            "bc_star_single": bc_single, "bc_star_two": bc_two,
            "rho_C_two_sim": sim_two["rho_C"], "se_two": sim_two["se_rho_C"],
            "rho_C_single_sim": sim_single["rho_C"], "se_single": sim_single["se_rho_C"],
            "rho_C_two_pred": rho_pred_two, "rho_C_single_pred": rho_pred_single,
            "completed_two": sim_two["completed"], "completed_single": sim_single["completed"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------
# Driver
# --------------------------------------------------------------
def run_all(out_csv: str = OUT_CSV) -> pd.DataFrame:
    """Run every (network, rule, delta) block and write the CSV and JSON.

    Steps: compile the numba kernels with two short runs on a ring (results
    discarded); delete `out_csv` if it exists; write the edge lists and
    settings of both networks to OUT_NETWORKS; then call run_panel for
    each network in NETWORKS, each rule in RULES and each delta in DELTAS,
    appending each block to `out_csv` as soon as it is finished.

    Parameters
    ----------
    out_csv : path of the results CSV (default OUT_CSV, relative to the
              working directory).

    Returns
    -------
    pandas.DataFrame with all 80 rows (the content of `out_csv`).
    """
    # Compile the numba kernels once (dB-dB and dB-Bd); output discarded.
    print("# Warming up Numba JIT...", flush=True)
    Gw = nx.cycle_graph(10)
    _ = run_sim(Gw, Gw, 0, 1, 3.0, 1.0, 2.0, 0.02, "dB-dB", True, 200, seed=0)
    _ = run_sim(Gw, Gw, 0, 1, 3.0, 1.0, 2.0, 0.02, "dB-Bd", True, 200, seed=0)
    print("# JIT warmed.", flush=True)

    if os.path.exists(out_csv):
        os.remove(out_csv)

    networks_meta = []
    for spec in NETWORKS:
        G1, G2 = spec["loader"]()
        # Record the network instance and its settings for the JSON file.
        networks_meta.append({
            "name": spec["name"],
            "N": G1.number_of_nodes(),
            "edges_L1": list(G1.edges()),
            "edges_L2": list(G2.edges()),
            "init_C": spec["init_C"], "init_M": spec["init_M"],
            "r": spec["r"],
            "b_grid": list(spec["b_grid"]),
        })

    with open(OUT_NETWORKS, "w") as f:
        json.dump(networks_meta, f, indent=2)
    print(f"# Saved network instances to {OUT_NETWORKS}", flush=True)

    all_rows = []
    for spec in NETWORKS:
        print(f"\n# === {spec['name']}, r={spec['r']} ===", flush=True)
        for rule in RULES:
            for delta in DELTAS:
                # delta = 0.02 uses n_runs_low_delta, delta = 0.2 n_runs_high_delta.
                n_runs = (spec["n_runs_low_delta"] if delta < 0.05
                          else spec["n_runs_high_delta"])
                t0 = time.time()
                df = run_panel(spec, rule, delta, spec["b_grid"], n_runs)
                dt = time.time() - t0
                print(f"  {rule}, delta={delta}, n_runs={n_runs}: "
                      f"{len(df)} b-values in {dt:.1f}s", flush=True)
                # Append this block; the header is written only into a new
                # or empty file.
                df.to_csv(out_csv, mode="a", index=False,
                          header=not os.path.exists(out_csv) or
                          os.path.getsize(out_csv) == 0)
                all_rows.append(df)
    full = pd.concat(all_rows, ignore_index=True)
    print(f"\n# Saved {len(full)} rows to {out_csv}", flush=True)
    return full


# --------------------------------------------------------------
# Plot: 2 x 2 panels (rows = networks, columns = rules)
# --------------------------------------------------------------
def plot(df: pd.DataFrame, fname: str = OUT_FIG):
    """Draw Fig G from a results table and save it to `fname`.

    Parameters
    ----------
    df    : results in the format of OUT_CSV (the output of run_all, or
            pd.read_csv(OUT_CSV)).
    fname : output PNG path (default OUT_FIG, relative to the working
            directory); saved at 180 dpi.

    Panels: (a) ring dB-dB, (b) ring dB-Bd, (c) BA dB-dB, (d) BA dB-Bd.
    In each panel, all y values are divided by pi_init, so 1 (dotted gray
    line) is the value at delta = 0. For each delta (blue squares 0.02,
    red circles 0.2):
      dashed line    rho_C_single_pred / pi_init (one-layer theory)
      solid line     rho_C_two_pred / pi_init (two-layer theory)
      open markers   rho_C_single_sim / pi_init with error bars se_single / pi_init
      filled markers rho_C_two_sim / pi_init with error bars se_two / pi_init
    Vertical lines: (b/c)*_single (green) and (b/c)*_two (purple). A panel
    whose rows are missing from `df` is left empty. The legend is drawn by
    legend_utils.draw_uniform_legend.
    """
    networks = ["Ring (N=10)", "BA-BA (N=15, seed=4)"]
    rules = ["dB-dB", "dB-Bd"]
    # Figure size leaves room for the bottom legend.
    fig, axes = plt.subplots(len(networks), len(rules),
                              figsize=(15, 11), squeeze=False)
    delta_styles = {
        0.02: dict(marker="s", color="#1f77b4"),
        0.20: dict(marker="o", color="#d62728"),
    }
    bc_color_single = "#2ca02c"   # green
    bc_color_two    = "#9467bd"   # purple

    # Font sizes
    F_LABEL = 16
    F_TICK  = 14
    F_PANEL = 16    # panel-label text "(a) Ring (...), dB-dB"
    F_LEG   = 11    # legend text; smaller than the rest so that four equally
                    # spaced legend columns fit without overlap

    panel_labels = ["a", "b", "c", "d"]
    panel_titles = {
        ("Ring (N=10)", "dB-dB"):           r"Ring ($N=10$, $r=2.5$), dB-dB",
        ("Ring (N=10)", "dB-Bd"):           r"Ring ($N=10$, $r=2.5$), dB-Bd",
        ("BA-BA (N=15, seed=4)", "dB-dB"):  r"BA ($N=15$, $r=5$), dB-dB",
        ("BA-BA (N=15, seed=4)", "dB-Bd"):  r"BA ($N=15$, $r=5$), dB-Bd",
    }

    # Fixed y-axis range and ticks for each row of panels.
    yrange_by_row = {
        0: (0.875, 1.075, [0.9, 0.95, 1.0, 1.05]),       # Ring panels (a, b)
        1: (0.65,  1.25,  [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),  # BA panels (c, d)
    }

    panel_idx = 0
    for i, net in enumerate(networks):
        for j, rule in enumerate(rules):
            ax = axes[i, j]
            sub = df[(df["network"] == net) & (df["rule"] == rule)]
            if sub.empty:
                continue
            pi0 = sub["pi_init"].iloc[0]
            bc_single = sub["bc_star_single"].iloc[0]
            bc_two = sub["bc_star_two"].iloc[0]

            for delta in sorted(sub["delta"].unique()):
                grp = sub[sub["delta"] == delta].sort_values("b_over_c")
                color = delta_styles[delta]["color"]
                marker = delta_styles[delta]["marker"]
                ax.plot(grp["b_over_c"], grp["rho_C_single_pred"] / pi0,
                        ls="--", color=color, alpha=0.55, lw=2.0)
                ax.plot(grp["b_over_c"], grp["rho_C_two_pred"] / pi0,
                        ls="-",  color=color, alpha=0.9, lw=2.2)
                ax.errorbar(grp["b_over_c"], grp["rho_C_single_sim"] / pi0,
                            yerr=grp["se_single"] / pi0,
                            fmt=marker, mfc="white", mec=color, color=color,
                            ms=8, capsize=2.5, lw=1.0, zorder=4)
                ax.errorbar(grp["b_over_c"], grp["rho_C_two_sim"] / pi0,
                            yerr=grp["se_two"] / pi0,
                            fmt=marker, mfc=color, mec=color, color=color,
                            ms=8, capsize=2.5, lw=1.0, zorder=4)

            ax.axhline(1.0, color="gray", lw=0.9, ls=":", alpha=0.7)
            if np.isfinite(bc_single):
                ax.axvline(bc_single, color=bc_color_single, lw=1.8, ls="-",
                            alpha=0.85)
            if np.isfinite(bc_two):
                ax.axvline(bc_two, color=bc_color_two, lw=1.8, ls="-",
                            alpha=0.85)

            ax.set_xlabel(r"$b/c$", fontsize=F_LABEL)
            ax.set_ylabel(r"$\rho_{\rm C} / \rho_{\rm C}^{\circ}$", fontsize=F_LABEL)

            ymin, ymax, yticks = yrange_by_row[i]
            ax.set_ylim(ymin, ymax)
            ax.set_yticks(yticks)
            ax.tick_params(labelsize=F_TICK)
            ax.grid(alpha=0.25)

            # Panel label "(a) Ring ..." in regular weight, left-aligned
            # above the axes.
            ax.set_title("")
            ax.text(0.0, 1.04,
                     f"({panel_labels[panel_idx]}) {panel_titles[(net, rule)]}",
                     transform=ax.transAxes, ha="left", va="bottom",
                     fontsize=F_PANEL)
            panel_idx += 1

    # The bottom legend is drawn on its own axes by
    # legend_utils.draw_uniform_legend, which places the legend markers at
    # equally spaced x positions (fig.legend does not give equal column
    # widths). legend_utils.py must be on the Python path (this folder).
    from legend_utils import draw_uniform_legend
    legend_rows = [
        # Row 1: open markers, gray dashed theory line, green (b/c)*
        [
            dict(kind="marker", marker="s", mfc="white", mec="#1f77b4",
                 label=r"simulation, one-layer, $\delta=0.02$"),
            dict(kind="marker", marker="o", mfc="white", mec="#d62728",
                 label=r"simulation, one-layer, $\delta=0.2$"),
            dict(kind="line",   color="gray",  linestyle="--", linewidth=2.2,
                 label=r"weak-selection theory, one-layer"),
            dict(kind="line",   color=bc_color_single, linestyle="-", linewidth=2.4,
                 label=r"$(b/c)^{*}$, one-layer"),
        ],
        # Row 2: filled markers, black solid theory line, purple (b/c)*
        [
            dict(kind="marker", marker="s", mfc="#1f77b4", mec="#1f77b4",
                 label=r"simulation, two-layer, $\delta=0.02$"),
            dict(kind="marker", marker="o", mfc="#d62728", mec="#d62728",
                 label=r"simulation, two-layer, $\delta=0.2$"),
            dict(kind="line",   color="black", linestyle="-",  linewidth=2.4,
                 label=r"weak-selection theory, two-layer"),
            dict(kind="line",   color=bc_color_two, linestyle="-", linewidth=2.4,
                 label=r"$(b/c)^{*}$, two-layer"),
        ],
    ]
    # Reserve the bottom 8% of the figure for the legend; fit panels into the
    # remaining vertical space first, then draw the legend axes.
    fig.tight_layout(rect=[0, 0.08, 1, 0.99], h_pad=3.0)
    draw_uniform_legend(fig, legend_rows, ncols=4,
                         bbox=(0.02, 0.0, 0.96, 0.07),
                         fontsize=F_LEG, marker_size=10)
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    print(f"# Figure saved: {fname}", flush=True)


if __name__ == "__main__":
    # Full run: simulations and theory, then the figure.
    df = run_all()
    plot(df)
