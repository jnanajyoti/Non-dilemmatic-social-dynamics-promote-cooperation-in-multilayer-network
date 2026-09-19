"""
Region maps in the (b/c, r) plane for two-layer ER and BA networks of size N.

Produces Fig H of S1 Text (Text L, "Numerical simulations of the stochastic
evolutionary dynamics"; LaTeX label fig:N=100), file fig4_random_N100.png.
The stored copy of the published figure is
published_figures/fig4_random_N100.png.

What the script does
--------------------
For a network size N it builds one two-layer ER network and one two-layer BA
network, and for each of them and each updating rule (dB-dB and dB-Bd) it
computes theta_1, theta_2, theta_3, phi_{0,1}, phi_{2,0} and phi_{2,1} with
larger_networks_solver.bc_star_two_layer. On a grid of (b/c, r) values it
then evaluates

    x = c theta_2 + b (theta_1 - theta_3) - (r - 1) phi_{2,0}
        (cooperation favored where x > 0; Eq. (3) of the main text),
    y = -(r - 1) theta_2 + c phi_{2,0} + b (phi_{0,1} - phi_{2,1})
        (the map treats the mutant as favored where y > 0),

with c = 1, and colours each grid point by one of four regions:

    dark green  (#28A745)  cooperators and mutants favored
    light green (#A6DBA0)  cooperators but not mutants favored
    light coral (#F4A582)  mutants but not cooperators favored
    deep red    (#D6604D)  neither favored

These are the colours of main-text Fig 4 (Figures and Data/Figures.ipynb in
this repository). A dotted vertical line marks
(b/c)*_single = -theta_2 / (theta_1 - theta_3).

Panels: (a) ER, dB-dB; (b) ER, dB-Bd; (c) BA, dB-dB; (d) BA, dB-Bd.

Networks and initial condition (for size N; Fig H uses N = 100)
---------------------------------------------------------------
* ER: both layers are generated with networkx.erdos_renyi_graph(N, 0.1). The
  networkx seed of layer 1 is drawn from numpy.random.default_rng(10 N) and
  that of layer 2 from numpy.random.default_rng(10 N + 7), i.e. 1000 and 1007
  for N = 100.
* BA: both layers are ba_julia.ba_julia(N, 2, seed), i.e. preferential
  attachment with m = 2 edges per new node, grown from a star graph on
  m + 1 nodes centred at node m (the construction of Julia's
  Graphs.barabasi_albert(N, m); see ba_julia.py), with seeds 10 N + 1
  (layer 1) and 10 N + 8 (layer 2), i.e. 1001 and 1008 for N = 100.
  The node labels of BA layer 2 are then permuted by
  ba_julia.shuffle_layer2 with seed 20260525 + 10 N + 1 (20261526 for
  N = 100). ER layers are not permuted.
* The initial cooperator (layer 1) and the initial mutant (layer 2) are both
  on node 0.
* bc_star_two_layer is called with r = 2. The value of r enters only its
  bc_star_two output; theta and phi do not depend on r.

How to run (from the revision folder)
-------------------------------------
Fig H, full run:

    python fig4_style_random.py 100

Several sizes can be given (python fig4_style_random.py 50 100). With no
argument the script runs N = 50, which is not an SI item.

Fig H, redraw from the stored CSV without solving the linear systems (the
four CSV rows are passed to make_figure in file order in place of the solver
calls; the networks are still generated; only the PNG is written):

    python -c "import pandas as pd, fig4_style_random as m; it = iter(pd.read_csv('fig4_random_N100_results.csv', float_precision='round_trip').to_dict('records')); m.bc_star_two_layer = lambda *a, **k: next(it); m.make_figure(100, 'fig4_random_N100.png')"

Both commands reproduce published_figures/fig4_random_N100.png byte for byte
(MD5 e6b7fd1d4abc26049dd2b3ebd11436b0), and the full run also reproduces
fig4_random_N100_results.csv byte for byte, with the package versions listed
in README.md.

Inputs
------
Full run: none (all networks are generated in the script).
Redraw: fig4_random_N100_results.csv.

Outputs (written to the working directory)
------------------------------------------
fig4_random_N{N}.png          the 2 x 2 figure (Fig H for N = 100), dpi 180.
fig4_random_N{N}_results.csv  one row per panel, in the order ER dB-dB,
                              ER dB-Bd, BA dB-dB, BA dB-Bd, with columns
                              model, param (p for ER, m for BA), N, rule,
                              init_C, theta1, theta2, theta3, phi01, phi20,
                              phi21, bc_star_single, bc_star_two (at r = 2).

Runtime
-------
N = 100: about 6.5 min (measured 385 s on an Apple M1 laptop with
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4). Redraw from the
stored CSV: about 5 s (measured).

Dependencies
------------
numpy, networkx, pandas, scipy, matplotlib (numba is not used), and the local
modules larger_networks_solver.py (bc_star_two_layer) and ba_julia.py
(ba_julia and shuffle_layer2, imported inside get_network and make_figure).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import scipy.sparse as sp
# Silence the scipy SparseEfficiencyWarning raised by the row assignment on a
# CSR matrix in larger_networks_solver._solve_beta_sparse.
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

from larger_networks_solver import bc_star_two_layer


# Colours of the four regions, as in main-text Fig 4
# (Figures and Data/Figures.ipynb). PALETTE[k - 1] is the colour of region
# code k returned by classify().
COL_BOTH_FAV    = "#28A745"   # Cooperators AND mutants both favored
COL_COOP_ONLY   = "#A6DBA0"   # Cooperators favored, mutants NOT
COL_MUTANT_ONLY = "#F4A582"   # Mutants favored, cooperators NOT
COL_NEITHER     = "#D6604D"   # Neither favored
PALETTE = [COL_BOTH_FAV, COL_COOP_ONLY, COL_MUTANT_ONLY, COL_NEITHER]
# Legend texts, in the same order as PALETTE.
LABELS = [
    "Cooperators and mutants favored",
    "Cooperators but not mutants favored",
    "Mutants but not cooperators favored",
    "Neither favored",
]


def compute_xy(b_vals, r_vals, theta1, theta2, theta3, phi20, phi01, phi21, c=1.0):
    """Evaluate the cooperator and mutant indicators on a (b, r) grid.

    Arguments:
        b_vals, r_vals: 1-D arrays of benefit b and mutant fitness r values.
        theta1, theta2, theta3: theta_1, theta_2, theta_3 of layer 1 (floats,
            as returned by larger_networks_solver.bc_star_two_layer).
        phi20, phi01, phi21: phi_{2,0}, phi_{0,1}, phi_{2,1} (floats, same
            source).
        c: cost of cooperation (default 1, so that b is b/c).

    Returns (x, y), two arrays of shape (len(r_vals), len(b_vals)); row k
    corresponds to r_vals[k] and column l to b_vals[l]:
        x = c theta_2 + b (theta_1 - theta_3) - (r - 1) phi_{2,0}
            (cooperation favored where x > 0; Eq. (3) of the main text),
        y = -(r - 1) theta_2 + c phi_{2,0} + b (phi_{0,1} - phi_{2,1})
            (mutant treated as favored where y > 0).
    """
    B, R = np.meshgrid(b_vals, r_vals)
    # Equal to c*theta2 + b*(theta1 - theta3) - (r - 1)*phi20.
    x = theta1 * B - (theta3 * B - theta2 * c) - (R - 1.0) * phi20
    y = -(R - 1.0) * theta2 + phi20 * c + B * (phi01 - phi21)
    return x, y


def classify(x, y):
    """Assign a region code to every grid point.

    Arguments: x, y, arrays of equal shape from compute_xy.

    Returns an integer array of the same shape with
        1 if x > 1e-10 and y > 0   (cooperators and mutants favored),
        2 if x > 0 and y <= 0      (cooperators but not mutants favored),
        3 if x <= 1e-10 and y > 0  (mutants but not cooperators favored),
        4 otherwise                (neither favored).
    Codes 1 to 4 map to PALETTE and LABELS in order.
    """
    Z = np.full_like(x, 4, dtype=int)
    Z[(x > 1e-10) & (y > 0)] = 1
    Z[(x > 0) & (y <= 0)] = 2
    Z[(x <= 1e-10) & (y > 0)] = 3
    return Z


def plot_phase(ax, theta1, theta2, theta3, phi20, phi01, phi21,
               b_range, r_range, title="", points_per_unit=100):
    """Draw one region-map panel on the matplotlib Axes `ax`.

    Arguments:
        ax: matplotlib Axes to draw on.
        theta1, theta2, theta3, phi20, phi01, phi21: theta and phi values of
            one two-layer network, rule and initial condition (see
            compute_xy).
        b_range: (b_min, b_max), the b/c window of the panel (c = 1).
        r_range: (r_min, r_max), the r window of the panel.
        title: panel title, drawn left-aligned at fontsize 22.
        points_per_unit: grid points per unit of b/c and of r; each axis has
            max(800, int(window width * points_per_unit)) points.

    The panel shows the region codes of classify(compute_xy(...)) coloured
    with PALETTE, and a dotted vertical line at
    (b/c)*_single = -theta2 / (theta1 - theta3) if that value lies strictly
    inside b_range. Returns None.
    """
    n_b = max(800, int((b_range[1] - b_range[0]) * points_per_unit))
    n_r = max(800, int((r_range[1] - r_range[0]) * points_per_unit))
    b_vals = np.linspace(*b_range, n_b)
    r_vals = np.linspace(*r_range, n_r)
    x, y = compute_xy(b_vals, r_vals, theta1, theta2, theta3, phi20, phi01, phi21)
    Z = classify(x, y)
    # Discrete colour map: code k in {1, 2, 3, 4} gets PALETTE[k - 1].
    cmap = ListedColormap(PALETTE)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    ax.pcolormesh(b_vals, r_vals, Z, cmap=cmap, norm=norm,
                   shading="auto", rasterized=True)
    # Dotted vertical line at (b/c)*_single of layer 1.
    denom = theta1 - theta3
    if abs(denom) > 1e-14:
        bc_single = -theta2 / denom
        if b_range[0] < bc_single < b_range[1]:
            ax.axvline(bc_single, linewidth=2.8, color="#2C3E50",
                       linestyle=':', alpha=0.85, zorder=10)
    ax.set_xlabel(r"$b/c$", fontsize=27, fontweight="bold")
    ax.set_ylabel(r"$r$", fontsize=27, fontweight="bold")
    # Panel title left-aligned at the start of the plotting box.
    ax.set_title(title, fontsize=22, loc="left")
    # Fixed box aspect (height / width = 0.9) for every panel, independent of
    # the b/c window of the panel.
    ax.set_box_aspect(0.9)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("#34495E")
    ax.tick_params(width=1.5, length=6, labelsize=17)


def get_network(model: str, N: int, param, seed: int) -> nx.Graph:
    """Return one network layer.

    Arguments:
        model: "ER" or "BA" (anything else raises ValueError).
        N: number of nodes given to the graph generator.
        param: edge probability p for "ER"; number m of edges added per new
            node for "BA" (converted to int).
        seed: integer seed.

    model "ER": generated with networkx.erdos_renyi_graph(N, param), with its
    networkx seed drawn as rng.integers(0, 2**31 - 1) from
    rng = numpy.random.default_rng(seed).
    model "BA": ba_julia.ba_julia(N, param, seed), i.e. preferential attachment
    with param edges per new node, grown from a star graph on param + 1 nodes
    centred at node param (the construction of Julia's
    Graphs.barabasi_albert; see ba_julia.py).
    Nodes are labelled by integers starting at 0.
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


def make_figure(N: int, fname: str, b_range=None, r_range=(0, 8)):
    """Build the 2 x 2 region-map figure for network size N and save it.

    Arguments:
        N: number of nodes passed to get_network.
        fname: output PNG path (saved at dpi 180).
        b_range: optional (b_min, b_max) b/c window used for all panels. If
            None, each panel uses (b/c)*_single +/- max(8, 1.5 |(b/c)*_single|),
            or (-20, 20) if (b/c)*_single is not finite.
        r_range: r window of all panels (default 0 to 8).

    Rows: ER with p = 0.1 (row 0) and BA with m = 2 (row 1). Columns: dB-dB
    (column 0) and dB-Bd (column 1). For row index row, layer 1 is generated
    with seed 10 N + row and layer 2 with seed 10 N + row + 7; the node labels
    of BA layer 2 are permuted with seed 20260525 + 10 N + row. The initial
    cooperator (layer 1) and initial mutant (layer 2) are both on node 0.
    theta and phi come from bc_star_two_layer(G1, G2, 0, 0, rule, r=2.0); r
    enters only its bc_star_two output.

    Returns a pandas DataFrame with one row per panel (ER dB-dB, ER dB-Bd,
    BA dB-dB, BA dB-Bd) and columns model, param, N, rule, init_C, theta1,
    theta2, theta3, phi01, phi20, phi21, bc_star_single, bc_star_two.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    # Figure rows: (model, parameter, label used in the panel title).
    # ER: edge probability p = 0.1 in both layers. BA: m = 2 in both layers.
    instances = [
        ("ER", 0.1, "ER"),
        ("BA", 2,   "BA"),
    ]
    rules = ["dB-dB", "dB-Bd"]
    # Panel letters in row-major order: (a) (b) in row 0, (c) (d) in row 1.
    panel_letters = [["a", "b"], ["c", "d"]]

    rows = []
    for row_idx, (model, param, lbl) in enumerate(instances):
        # Same model and parameter in both layers, different seed per layer.
        G1 = get_network(model, N, param, seed=10 * N + row_idx)
        G2 = get_network(model, N, param, seed=10 * N + row_idx + 7)
        # Put both layers on the same node labels 0, ..., n-1; node i is the
        # same individual in layer 1 and layer 2.
        common = set(G1.nodes()) & set(G2.nodes())
        G1 = nx.convert_node_labels_to_integers(G1.subgraph(common).copy())
        common_sorted = sorted(common)
        mapping = {old: i for i, old in enumerate(common_sorted)}
        G2 = nx.relabel_nodes(G2, mapping)
        # BA only: relabel the nodes of layer 2 by a seeded uniform random
        # permutation (ba_julia.shuffle_layer2), as for the BA networks of
        # the main text. In a BA graph the node label is correlated with
        # degree (early nodes tend to become hubs), so without relabelling
        # the hubs of the two layers would tend to be the same individuals.
        # ER layers are not relabelled.
        if model == "BA":
            from ba_julia import shuffle_layer2
            G2, _perm = shuffle_layer2(G2, seed=20260525 + 10 * N + row_idx)
        # Initial cooperator (layer 1) and initial mutant (layer 2) both on
        # node 0.
        init_C = init_M = 0
        for col_idx, rule in enumerate(rules):
            # If bc_star_two_layer raises an exception, print a warning,
            # leave this panel empty and add no CSV row for it.
            try:
                res = bc_star_two_layer(G1, G2, init_C, init_M, rule, r=2.0)
            except Exception as e:
                print(f"[warn] {model} at N={N}, {rule}: {e}")
                continue
            ax = axes[row_idx, col_idx]
            title = f"({panel_letters[row_idx][col_idx]}) {lbl}, {rule}"
            # b/c window centred on (b/c)*_single unless b_range is given.
            bc_s = res["bc_star_single"]
            if b_range is None:
                if np.isfinite(bc_s):
                    half = max(8.0, abs(bc_s) * 1.5)
                    panel_b_range = (bc_s - half, bc_s + half)
                else:
                    panel_b_range = (-20, 20)
            else:
                panel_b_range = b_range
            plot_phase(ax, res["theta1"], res["theta2"], res["theta3"],
                        res["phi20"], res["phi01"], res["phi21"],
                        panel_b_range, r_range, title=title)
            rows.append({
                "model": model, "param": param, "N": G1.number_of_nodes(),
                "rule": rule, "init_C": init_C,
                "theta1": res["theta1"], "theta2": res["theta2"],
                "theta3": res["theta3"],
                "phi01": res["phi01"], "phi20": res["phi20"], "phi21": res["phi21"],
                "bc_star_single": res["bc_star_single"],
                "bc_star_two": res["bc_star_two"],
            })

    # Shared legend of the four regions below the panels.
    handles = [Patch(facecolor=PALETTE[i], edgecolor="black", lw=0.4, label=LABELS[i])
                for i in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=12,
                bbox_to_anchor=(0.5, -0.01), frameon=False)
    # No figure-level title. rect leaves room for the legend at the bottom;
    # h_pad=5.0 adds vertical space between the two rows of panels.
    fig.tight_layout(rect=[0, 0.04, 1, 0.99], h_pad=5.0)
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    print(f"# Figure saved: {fname}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Usage: python fig4_style_random.py [N ...]
    # Default N = 50; Fig H uses N = 100. Outputs are written to the working
    # directory.
    import sys, time
    sizes = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [50]
    for N in sizes:
        t0 = time.time()
        df = make_figure(N, f"fig4_random_N{N}.png")
        df.to_csv(f"fig4_random_N{N}_results.csv", index=False)
        print(f"# N={N} done in {time.time()-t0:.1f}s")
