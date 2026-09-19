"""
Strategy-type correlation between the two layers during evolutionary dynamics:
produces S1 Text, Fig C (Text F).

What is computed
----------------
Five two-layer networks with N = 6 individuals (layer 1: donation game, layer 2:
constant selection), dB-dB rule, delta = 0.02, c = 1, r = 4. For each network
and panel, 5 x 10^5 independent runs start from one cooperator (layer 1) and one
mutant (layer 2). After update steps t in TIME_BINS the Pearson correlation
coefficient across nodes between x1 (defector 0, cooperator 1) and x2 (resident
0, mutant 1) is recorded when both layers are polymorphic, and averaged over
(i) all runs, (ii) runs whose layer 1 ends all-C, (iii) runs whose layer 1 ends
all-D. rho_C = n_C / (n_C + n_D) is the simulated fixation probability of
cooperation, where n_C and n_D are the numbers of runs whose layer 1 ends
all-C and all-D, respectively.

Panels (value of the CSV column "panel")
  top    : correlated IC, cooperator and mutant on node 0 (iC=0, iM=0);
           b = b_for((b/c)*_two at IC (0,0))            -> Fig C (a)-(e)
  bot_ii : uncorrelated IC, cooperator on node 0, mutant on node 1 (iC=0, iM=1);
           b = b_for((b/c)*_two at IC (0,1))            -> Fig C (f)-(j)
  bot_i  : uncorrelated IC (0,1) simulated with the "top" b value; written to
           the CSV and drawn in the *_conv_i* PNGs, not used in Fig C.
b_for(x) = max(1.5 x, 2.0) if x > 0 else 5.0; in all ten Fig C panels this is
exactly 1.5 (b/c)*_two. (b/c)* values are init-specific: theta_n and phi_{2,0}
are computed by larger_networks_solver.bc_star_two_layer for the simulated IC
only, with (b/c)*_single = -theta2/(theta1-theta3) and
(b/c)*_two = (b/c)*_single + (r-1) phi20 / (c (theta1-theta3)) (main-text
Eq. (5)).

Fig C panel map (columns follow NETS_A order; values as printed by bc_table()
and quoted in the Fig C caption; bc_single depends only on G1 and iC):
  panel (G1,G2)   IC     b      bc_single bc_two rho_C
  (a)   (25,5)   (0,0)  10.12  28.32     6.75   0.143
  (b)   (25,17)  (0,0)  12.17  28.32     8.11   0.144
  (c)   (28,37)  (0,0)   3.98  10.72     2.65   0.073
  (d)   (28,14)  (0,0)   2.70  10.72     1.80   0.071
  (e)   (25,8)   (0,0)  11.29  28.32     7.53   0.143
  (f)   (25,5)   (0,1)  39.71  28.32    26.47   0.142
  (g)   (25,17)  (0,1)  46.96  28.32    31.31   0.142
  (h)   (28,37)  (0,1)  15.34  10.72    10.23   0.072
  (i)   (28,14)  (0,1)  17.15  10.72    11.43   0.073
  (j)   (25,8)   (0,1)  48.68  28.32    32.45   0.142

Networks
--------
(G1, G2) are indices into GRAPHS = the 112 connected 6-node graphs of
nx.graph_atlas_g(), kept in atlas order (networkx 3.4.2). Edge lists of the
graphs used, so the indices can be checked against other networkx versions:
  25: (0,1) (0,4) (1,2) (1,4) (1,5) (2,3) (3,4)
  28: (0,2) (1,2) (1,4) (2,3) (2,5) (3,4) (4,5)
   5: (0,1) (0,5) (2,3) (3,4) (4,5)
  17: (0,1) (0,4) (1,2) (1,5) (2,3) (3,4)
   8: (0,4) (1,5) (2,3) (3,4) (3,5) (4,5)
  37: (0,1) (0,2) (0,3) (1,2) (3,4) (3,5) (4,5)
  14: (0,1) (0,2) (0,3) (3,4) (3,5) (4,5)
Node labels are 0..5, so node index = label.

NETS_A is hard-coded; no data file is read to choose it. It is the result of
the Text F selection applied offline to r3_n6_exhaustive_results.csv (written by
r3_topology_correlation_n6_exhaustive.py: dB-dB, r=5, theta_diff = theta1-theta3,
theta2 and phi20 averaged over the six correlated ICs v = 0..5): among rows with
finite bc_two and theta_diff > 0.1 (3136 pairs), rank by |phi20| / theta_diff
(independent of r) and take the top five:
(25,5) 3.5109, (25,17) 3.4180, (28,37) 3.4141, (28,14) 3.4005, (25,8) 3.3673
(sixth: (28,2) 3.3072). All five have phi20/theta_diff < 0, i.e.
d(b/c)*/dr < 0.

Command (from this folder; all paths are relative to the working directory)
-------
    SC_R=4 python r3_strategy_coupling_ic.py              # simulate + plot
    SC_R=4 python r3_strategy_coupling_ic.py --plot-only  # plot from the CSV
SC_R=4 is required for Fig C. Without it r = 5.0 and all file names lose the
"_r4" suffix (r3_strategy_coupling_ic_results.csv, r3_sc_*.png); that r = 5
output is not in the SI or the repository, so --plot-only without SC_R=4
works only after a full r = 5 run has written that CSV.
Runtime on an Apple M1 laptop: full run about 45 s (serial; Numba compilation,
then 15 jobs of 5 x 10^5 runs at 2 to 3 s each, then the plots); --plot-only
about 6 s.

Inputs
------
Full run: none (graphs from networkx). --plot-only:
r3_strategy_coupling_ic_results_r4.csv.
Hard-coded parameters: R=4 (via SC_R), C=1.0, DELTA=0.02, dB-dB
(layer2_is_Bd=False), N_RUNS=500_000, MAX_STEPS=5000 and TIME_BINS (imported
from r3_strategy_coupling), seed per job
1000 + 13*g1 + 7*g2 + 100003*iM + 17*round(b).

Outputs
-------
r3_strategy_coupling_ic_results_r4.csv
    210 rows = 5 networks x 3 panels x 14 time bins, rows ordered by
    sorted(NETS_A) then top, bot_i, bot_ii. Columns: g1, g2, panel, iC, iM,
    bc_single, bc_two (for the simulated IC), b_over_c (= b, c = 1), t,
    mean_R_all, n_R_all, mean_R_coop, n_R_coop, mean_R_def, n_R_def, rho_C.
r3_sc_versionA_conv_ii_r4.png
    Fig C: top row "top", bottom row "bot_ii". Identical bytes to
    published_figures/FigC_strategy_coupling_r4.png and to the SI source file
    R3C3_B_swap-7-26_perpanelBC_r4.png (md5 364bf9e17bb6209217038e93178f3b18).
r3_sc_versionB_conv_ii_r4.png
    Byte-identical to versionA_conv_ii (NETS_B is NETS_A).
r3_sc_versionA_conv_i_r4.png, r3_sc_versionB_conv_i_r4.png
    Bottom row "bot_i" instead of "bot_ii"; not in the SI; identical to each
    other.
stdout
    Per-job progress lines and the "(b/c)* per panel (init-specific)" table
    (bc_table), the source of the numbers in the Fig C caption.
The "versionA/versionB" and "conv_i/conv_ii" names are kept from the drafting
stage, when two network sets and two bottom-row conventions were compared.

Reproducibility
---------------
The simulation kernel is serial and seeded with np.random.seed inside Numba, so
rerunning with SC_R=4 regenerates the CSV and all four PNGs byte-for-byte
(checked with Python 3.13, numpy 2.1.3, numba 0.61.0, networkx 3.4.2,
pandas 2.2.3, scipy 1.15.3, matplotlib 3.10.0).

Dependencies
------------
numpy, networkx, pandas, scipy, matplotlib, numba; local modules
  r3_strategy_coupling     : _sim_many_with_corr, TIME_BINS, MAX_STEPS
  weak_selection_sim_fast  : pack_graph (plus the payoff/update kernels used
                             inside _sim_many_with_corr)
  larger_networks_solver   : bc_star_two_layer (dB-dB branch)
"""
from __future__ import annotations
import os, sys, time
import warnings, scipy.sparse as sp
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)
import numpy as np, networkx as nx, pandas as pd
import matplotlib.pyplot as plt

from weak_selection_sim_fast import pack_graph
from larger_networks_solver import bc_star_two_layer
from r3_strategy_coupling import _sim_many_with_corr, TIME_BINS, MAX_STEPS

# R is read from the environment variable SC_R (default 5.0). Fig C uses SC_R=4
# (at r=4 all five (b/c)*_two for the correlated IC are positive). For R != 5
# the output names get the suffix "_r<round(R)>" (e.g. "_r4"); R = 5 uses
# unsuffixed names.
R, C, DELTA = float(os.environ.get("SC_R", "5.0")), 1.0, 0.02
SUF = "" if abs(R - 5.0) < 1e-9 else f"_r{int(round(R))}"
N_RUNS = 500_000
OUT_CSV = f"r3_strategy_coupling_ic_results{SUF}.csv"

# The 112 connected 6-node graphs in networkx graph-atlas order; indices in
# NETS_A refer to this list (edge lists in the module docstring).
GRAPHS = [G for G in nx.graph_atlas_g()
          if G.number_of_nodes() == 6 and nx.is_connected(G)]
# (G1, G2) index pairs in Fig C column order. Selected offline from
# r3_n6_exhaustive_results.csv: top 5 by |phi20| / theta_diff among pairs with
# theta_diff > 0.1 (pairs may share G1 or G2). See the module docstring for
# the ranking values.
NETS_A = [(25, 5), (25, 17), (28, 37), (28, 14), (25, 8)]
NETS_B = NETS_A          # same list: versionB PNGs are copies of versionA PNGs
ALL_NETS = sorted(set(NETS_A))   # simulation / CSV order


def bc(g1, g2, iC, iM):
    """Weak-selection thresholds for one two-layer network and initial condition.

    Parameters
    ----------
    g1, g2 : indices into GRAPHS for layer 1 (donation game) and layer 2
             (constant selection)
    iC, iM : node index of the initial cooperator (layer 1) and initial mutant
             (layer 2)

    Returns
    -------
    (bc_single, bc_two) from larger_networks_solver.bc_star_two_layer with
    rule "dB-dB", r = R, c = C:
      bc_single = -theta2 / (theta1 - theta3)              (layer 1 alone)
      bc_two    = bc_single + (r-1) phi20 / (c (theta1 - theta3))
    i.e. the b/c at which c theta2 + b (theta1-theta3) - (r-1) phi20 = 0
    (main-text Eqs. (3) and (5); S1 Text Eq. (S50)), with theta_n and
    phi_{2,0} for this single initial condition (not averaged over initial
    conditions).
    """
    s = bc_star_two_layer(GRAPHS[g1], GRAPHS[g2], iC, iM, "dB-dB", r=R, c=C)
    return s["bc_star_single"], s["bc_star_two"]


def b_for(bc_two):
    """Benefit b used in a panel (c = C = 1, so b equals b/c).

    Returns max(1.5 * bc_two, 2.0) if bc_two > 0, else 5.0. For all Fig C
    panels bc_two >= 1.80, so b = 1.5 * bc_two.
    """
    return float(max(1.5 * bc_two, 2.0)) if bc_two > 0 else 5.0


def run_one(g1, g2, iC, iM, b, panel, n_runs):
    """Simulate one (network, initial condition, b) job and tabulate R(t).

    Parameters
    ----------
    g1, g2 : indices into GRAPHS (layer 1, layer 2)
    iC, iM : initial cooperator node (layer 1) and initial mutant node (layer 2)
    b      : benefit (c = C)
    panel  : label stored in the "panel" column ("top", "bot_i", "bot_ii";
             "warm" for the JIT warm-up call)
    n_runs : number of independent runs

    Calls r3_strategy_coupling._sim_many_with_corr with dB-dB
    (layer2_is_Bd=False), r = R, delta = DELTA, TIME_BINS, MAX_STEPS and
    seed = 1000 + 13*g1 + 7*g2 + 100003*iM + 17*round(b).

    Returns
    -------
    DataFrame with one row per entry of TIME_BINS and columns
    g1, g2, panel, iC, iM, bc_single, bc_two (from bc() for this (iC, iM)),
    b_over_c, t, mean_R_all/n_R_all (all runs), mean_R_coop/n_R_coop (runs whose
    layer 1 ends all-C), mean_R_def/n_R_def (layer 1 ends all-D), and
    rho_C = n_C / (n_C + n_D) (same value in every row). mean_* is NaN when no
    run contributed a defined correlation at that t.
    """
    G1, G2 = GRAPHS[g1], GRAPHS[g2]
    bc1, bc2 = bc(g1, g2, iC, iM)
    ptr1, neigh1, w1, deg1 = pack_graph(G1)
    ptr2, neigh2, w2, deg2 = pack_graph(G2)
    seed = int(1000 + 13 * g1 + 7 * g2 + 100003 * iM + 17 * int(round(b)))
    out = _sim_many_with_corr(ptr1, neigh1, w1, deg1, ptr2, neigh2, w2, deg2,
                              iC, iM, b, C, R, DELTA, False,
                              TIME_BINS, n_runs, MAX_STEPS, seed)
    # sums/counts of R per bin (all, all-C, all-D), layer-1 outcome counts
    # (nC, nD, time-outs), layer-2 outcome counts; absorption-time sums unused
    (sR, cR, sRc, cRc, sRd, cRd, nC, nD, nto, nM, nW, ntoM, *_ ) = out
    rows = []
    for k, t in enumerate(TIME_BINS):
        rows.append(dict(g1=g1, g2=g2, panel=panel, iC=iC, iM=iM,
            bc_single=bc1, bc_two=bc2, b_over_c=b, t=int(t),
            mean_R_all=(sR[k]/cR[k]) if cR[k] > 0 else np.nan, n_R_all=int(cR[k]),
            mean_R_coop=(sRc[k]/cRc[k]) if cRc[k] > 0 else np.nan, n_R_coop=int(cRc[k]),
            mean_R_def=(sRd[k]/cRd[k]) if cRd[k] > 0 else np.nan, n_R_def=int(cRd[k]),
            rho_C=nC/max(nC+nD, 1)))
    return pd.DataFrame(rows)


def main(n_runs=N_RUNS):
    """Run all 15 simulation jobs and write OUT_CSV (overwritten).

    For each (g1, g2) in ALL_NETS: computes (b/c)*_two for the correlated IC
    (0,0) and the uncorrelated IC (0,1), sets b_top = b_for(bc_two at (0,0)) and
    b_self = b_for(bc_two at (0,1)), and runs the jobs "top" (IC (0,0), b_top),
    "bot_i" (IC (0,1), b_top) and "bot_ii" (IC (0,1), b_self), appending each
    run_one table to OUT_CSV and printing a progress line. Returns None.
    """
    print("# warming up...", flush=True)
    # 200-run call only to trigger Numba compilation; its result is discarded
    _ = run_one(25, 5, 0, 0, 5.0, "warm", 200)
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
    for (g1, g2) in ALL_NETS:
        _, bc2_co  = bc(g1, g2, 0, 0)
        _, bc2_non = bc(g1, g2, 0, 1)
        b_top, b_self = b_for(bc2_co), b_for(bc2_non)
        jobs = [(0, 0, b_top, "top"),         # co-located, top-row b/c
                (0, 1, b_top, "bot_i"),       # non-co-located at top-row b/c
                (0, 1, b_self, "bot_ii")]     # non-co-located at its own b/c
        for iC, iM, b, panel in jobs:
            t0 = time.time()
            df = run_one(g1, g2, iC, iM, b, panel, n_runs)
            df.to_csv(OUT_CSV, mode="a", index=False,
                      header=not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0)
            print(f"# ({g1},{g2}) {panel:7}: b/c={b:5.2f} bc_two={df.bc_two.iloc[0]:+6.2f} "
                  f"rho_C={df.rho_C.iloc[0]:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"# saved {OUT_CSV}")


def _panel(ax, sub, letter, FL, FP):
    """Draw one Fig C panel.

    Parameters
    ----------
    ax     : matplotlib Axes
    sub    : rows of OUT_CSV for one (g1, g2, panel), sorted by t
    letter : panel letter, drawn as "(letter)" above the top-left corner
    FL     : axis-label font size (not used inside this function)
    FP     : panel-letter font size

    Plots mean_R_all (black circles), mean_R_coop (blue squares) and mean_R_def
    (red triangles) against t on a log x-axis, with a dotted line at 0.
    Returns None.
    """
    ax.plot(sub.t, sub.mean_R_all, marker="o", color="black", ms=5, lw=1.8, label="all runs")
    ax.plot(sub.t, sub.mean_R_coop, marker="s", color="#1f77b4", ms=5, lw=1.4, label="runs ending in all C")
    ax.plot(sub.t, sub.mean_R_def, marker="^", color="#d62728", ms=5, lw=1.4, label="runs ending in all D")
    ax.axhline(0.0, color="gray", lw=0.9, ls=":", alpha=0.7)
    ax.set_xscale("log"); ax.set_xlim(0.9, sub.t.max()*1.1); ax.set_ylim(-0.6, 1.05)
    ax.tick_params(labelsize=15); ax.grid(alpha=0.25)
    ax.text(0.0, 1.02, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=FP)


def plot_version(top_nets, bottom_panel, out_png):
    """Draw the 2 x 5 correlation figure from OUT_CSV and save it.

    Parameters
    ----------
    top_nets     : list of five (g1, g2); column k shows top_nets[k]
    bottom_panel : CSV panel label for the bottom row ("bot_ii" for Fig C,
                   "bot_i" for the unpublished conv_i variant)
    out_png      : output file name (saved at dpi=170, bbox_inches="tight")

    Top row (letters a-e) uses the "top" rows (correlated IC); bottom row
    (letters f-j) uses `bottom_panel` rows (uncorrelated IC). Returns None.
    """
    df = pd.read_csv(OUT_CSV)
    # Font sizes: axis & legend labels ~1.3x, panel letters ~1.5x the earlier draft.
    FL, FP, FLEG = 26, 30, 24
    fig, axes = plt.subplots(2, 5, figsize=(22, 9.4), squeeze=False, sharey=True)
    letters = [chr(ord("a")+k) for k in range(10)]
    for col, (g1, g2) in enumerate(top_nets):
        for r_i, panel in enumerate(["top", bottom_panel]):
            sub = df[(df.g1 == g1) & (df.g2 == g2) & (df.panel == panel)].sort_values("t")
            # letter index: row 0 -> a..e, row 1 -> f..j
            _panel(axes[r_i, col], sub, letters[r_i*5+col], FL, FP)
    for ax in axes[:, 0]:
        ax.set_ylabel("correlation", fontsize=FL)
    for ax in axes[1, :]:
        ax.set_xlabel("time", fontsize=FL)
    # Row labels use "individual(s)" to match the Text F wording.
    fig.text(0.004, 0.74, "cooperator & mutant\non the same individual",
             rotation=90, ha="left", va="center", fontsize=20)
    fig.text(0.004, 0.30, "cooperator & mutant\non different individuals",
             rotation=90, ha="left", va="center", fontsize=20)
    h, l = axes[0, 0].get_legend_handles_labels()
    # Legend: one row, no mode="expand", reduced column spacing so the three
    # entries (black/blue/red) sit close together.
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=FLEG,
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=[0.035, 0.07, 1, 0.99], h_pad=3.0, w_pad=1.2)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    print(f"# saved {out_png}")


def bc_table():
    """Print the per-panel table used for the Fig C caption.

    Reads OUT_CSV and prints, for each (g1, g2) in ALL_NETS and each panel
    (top, bot_i, bot_ii), the IC, b used, bc_single, bc_two (for that IC) and
    rho_C. Returns None. The Fig C caption quotes the "top" rows for (a)-(e) and
    the "bot_ii" rows for (f)-(j), in NETS_A column order.
    """
    df = pd.read_csv(OUT_CSV)
    print("\n# (b/c)* per panel (init-specific):")
    print(f"{'network':9} {'panel':7} {'IC':16} {'b/c used':>9} {'bc_single':>10} {'bc_two':>8} {'rho_C':>7}")
    seen = set()
    for (g1, g2) in ALL_NETS:
        for panel in ["top", "bot_i", "bot_ii"]:
            s = df[(df.g1 == g1) & (df.g2 == g2) & (df.panel == panel)]
            if len(s) == 0: continue
            s = s.iloc[0]
            ic = "node0 & node0" if s.iM == s.iC else "node0 & node1"
            print(f"({g1},{g2})".ljust(9), f"{panel:7}", f"{ic:16}",
                  f"{s.b_over_c:9.2f}", f"{s.bc_single:10.2f}", f"{s.bc_two:8.2f}", f"{s.rho_C:7.3f}")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        main()
    # Four PNGs; Fig C is the versionA_conv_ii file (versionB_* are duplicates,
    # conv_i variants are not in the SI).
    plot_version(NETS_A, "bot_i",  f"r3_sc_versionA_conv_i{SUF}.png")
    plot_version(NETS_A, "bot_ii", f"r3_sc_versionA_conv_ii{SUF}.png")
    plot_version(NETS_B, "bot_i",  f"r3_sc_versionB_conv_i{SUF}.png")
    plot_version(NETS_B, "bot_ii", f"r3_sc_versionB_conv_ii{SUF}.png")
    bc_table()
