# Code for the analyses added during peer review (S1 Text)

This folder contains the Python code, the stored results and the published
figure files for the analyses that were added during peer review to

Jnanajyoti Bhaumik and Naoki Masuda,
"Coupling with opinion dynamics promotes prosocial behavior in multilayer networks",
PLOS Computational Biology (2026); preprint arXiv:2601.00460.

These analyses are reported in the Supporting Information file S1 Text as
Fig A, Fig B, Fig C, Fig G, Fig H, Fig I, Table A and Table B. The table
below gives the script and command for each item; for Fig G and Fig I the
plotting step reads the stored CSV files that are included in this folder.
The other figures of the paper are produced by the notebooks in the parent
directory of this repository.

## Model and quantities

Layer 1 carries the donation game (cooperator or defector; benefit b, cost c).
Layer 2 carries constant selection (mutant with fitness r, resident with
fitness 1). The same individual occupies node i in both layers. Selection is
weak (strength delta), and the updating rule is dB-dB (death-Birth in both
layers) or dB-Bd (death-Birth in layer 1, Birth-death in layer 2). The initial
state has one cooperator in layer 1 and one mutant in layer 2.

For a given two-layer network, updating rule and initial condition, the
solvers `larger_networks_solver.py` (sparse matrices) and
`weak_selection_validation.py` (dense matrices) compute theta_1, theta_2,
theta_3 and phi_{n,m} as defined in S1 Text. Cooperation is favored when

    c theta_2 + b (theta_1 - theta_3) - (r - 1) phi_{2,0} > 0.

The threshold value of b/c for the one-layer network (layer 1 alone) and for
the two-layer network, and the rate of change of the latter with r, are

    (b/c)*_single = -theta_2 / (theta_1 - theta_3)
    (b/c)*_two    = (b/c)*_single + (r - 1) phi_{2,0} / (c (theta_1 - theta_3))
    d(b/c)*/dr    = phi_{2,0} / (c (theta_1 - theta_3)).

When theta_1 - theta_3 > 0, cooperation is favored for b/c above the
threshold.

## Map from SI item to code

Run every command from this folder (`cd revision`). `python` is the
interpreter of the environment described below. Runtimes are wall-clock times
on an Apple M1 laptop; "estimated" means extrapolated from short pilot runs.

| SI item | Entry script | Command | Inputs | Outputs | Runtime |
|---|---|---|---|---|---|
| Fig A (Text E.1) | `r3_naoki_6x6_heatmap.py` | `python r3_naoki_revisions.py`<br>`python r3_naoki_6x6_heatmap.py` | `r3_naoki_per_graph.csv` (written by the first command; included) | `fig1_R3_pairwise_5x5.png` (Fig A), `r3_pairwise_5x5_R.csv`, `r3_pairwise_5x5_p.csv` | 5 s + 2 s |
| Fig B (Text E.2) | `r3_naoki_revisions.py` | `python r3_naoki_revisions.py` | none (graphs from `networkx.graph_atlas_g()`) | `fig5_R3_per_node_centrality.png` (Fig B), `r3_per_node_centrality_results.csv`, `r3_per_node_centrality_summary.csv`; also `r3_naoki_per_graph.csv`, `r3_partial_correlations.csv` and `.txt`, `fig1_R3_scatter_matrix_5x5.png` | 5 s |
| Table A, six-node columns (Text E.1) | `r3_naoki_slope_bilayer_table.py` | `python r3_topology_correlation_n6_exhaustive.py`<br>`python r3_naoki_revisions.py`<br>`python r3_naoki_slope_bilayer_table.py` | `r3_n6_exhaustive_results.csv` and `r3_naoki_per_graph.csv` (written by the first two commands; both included, so the last command can be run on its own) | `r3_part_4_bilayer_slope_table.csv` (Table A), `r3_part_4_bilayer_slope_table.txt`, `fig4_R3_bilayer_slope_table.png`; the scan also writes `r3_n6_exhaustive_correlations.png` and `r3_n6_exhaustive_full_corr_matrix.png` | 2.5 min + 5 s + 4 s |
| Table A, larger column (Text E.1) | `r3_larger_networks_partB_extended.py` | `python r3_larger_networks_partB_extended.py` | `data/empirical_networks/edges.csv`, `data/empirical_networks/law_edges.csv`; ER and BA networks are generated in the scripts | `r3_part_B_extended_per_network.csv`, `r3_part_B_extended_correlations.csv` (Table A: the rows with layer "layer 1") | 4 min |
| Table B (Text E.2) | `r3_larger_networks_partA.py` | `python r3_larger_networks_partA.py` | the two edge lists above; ER and BA networks are generated in the script | `r3_part_A_table.csv` (Table B), `r3_part_A_per_network.csv`, `r3_part_A_per_node.csv` | 9.5 min |
| Fig C (Text F) | `r3_strategy_coupling_ic.py` | full run: `SC_R=4 python r3_strategy_coupling_ic.py`<br>plot only: `SC_R=4 python r3_strategy_coupling_ic.py --plot-only` | full run: none; plot only: `r3_strategy_coupling_ic_results_r4.csv` (included) | `r3_sc_versionA_conv_ii_r4.png` (Fig C), `r3_strategy_coupling_ic_results_r4.csv` (full run only), and three further PNGs (see Fig C below) | full run 45 s; plot only 6 s |
| Fig G (Text L) | `run_ws_validation_ring_baba.py` | plot from the stored CSV: `python -c "import pandas as pd, run_ws_validation_ring_baba as m; m.plot(pd.read_csv(m.OUT_CSV))"`<br>full run: `NUMBA_NUM_THREADS=4 python run_ws_validation_ring_baba.py` | plot: `ws_validation_ring_baba_results.csv` (included) and `legend_utils.py`; full run: no data files | `ws_validation_ring_baba.png` (Fig G); the full run also writes `ws_validation_ring_baba_results.csv` and `ws_validation_ring_baba_networks.json` | plot 3 s; full run about 4 min (estimated) |
| Fig H (Text L) | `fig4_style_random.py` | full run: `python fig4_style_random.py 100`<br>redraw from the stored CSV: see Fig H below | full run: none; redraw: `fig4_random_N100_results.csv` (included) | `fig4_random_N100.png` (Fig H), `fig4_random_N100_results.csv` (full run only) | full run 6.5 min; redraw 5 s |
| Fig I (Text L) | `point_b_mc_overlay.py` | plot from the stored CSV: `python point_b_mc_overlay.py --plot-only`<br>full run: `NUMBA_NUM_THREADS=4 python point_b_mc_overlay.py` | plot: `point_b_mc_results.csv` (included) and `legend_utils.py`; full run: no data files | `point_b_mc_overlay.png` (Fig I); the full run also writes `point_b_mc_results.csv` and `point_b_mc_networks.json` | plot 3 s; full run 15 to 17 min (estimated) |

Every script writes its outputs to the working directory and overwrites
files of the same name, including the stored CSV files of this folder. Copy
the folder first if the stored files should be kept.

## Details for each item

### Fig A, Fig B and the partial correlations of Text E.1

`r3_naoki_revisions.py` takes the 112 connected six-node graphs of
`networkx.graph_atlas_g()`. For each graph and each node v it calls
`larger_networks_solver.bc_star_two_layer` with the single initial cooperator
on v and records theta_1 - theta_3 and `(b/c)*_single`. It writes, per graph,
the mean of theta_1 - theta_3 over the six nodes (`theta_diff_mean`), the
number of nodes v with `(b/c)*_single` > 0 (`k_coop`, denoted n_init,C in
S1 Text), the mean degree, mean closeness, mean betweenness and clustering
coefficient (`r3_naoki_per_graph.csv`), and the partial Pearson correlations
with the mean degree partialled out (`r3_partial_correlations.csv`; the
clustering and mean-closeness rows are the values reported in Text E.1).

For Fig B it uses the graphs with `k_coop` = 6 and computes, for each graph,
the Pearson correlation coefficient across the six nodes between
`(b/c)*_single` and the degree, closeness, betweenness and local clustering
coefficient of the node of the initial cooperator. A graph is skipped for a
centrality that takes the same value at all six nodes; the number of graphs
used for each centrality is in `r3_per_node_centrality_summary.csv`.

`r3_naoki_6x6_heatmap.py` reads `r3_naoki_per_graph.csv` and draws Fig A,
the 5 x 5 matrix of Pearson correlation coefficients among `theta_diff_mean`,
`k_coop`, mean degree, mean closeness and clustering coefficient. The "6x6"
in the script name refers to an earlier six-quantity version of the figure.

In all six-node scripts a graph index (`graph_idx`, `g1_idx`, `g2_idx`, and
the indices of Fig C) is the position, from 0 to 111, in the list of connected
six-node graphs of `networkx.graph_atlas_g()` taken in atlas order. It is not
the graph-atlas index itself.

### Table A, six-node columns

`r3_topology_correlation_n6_exhaustive.py` runs over all 112 x 112 = 12,544
ordered pairs (G_1, G_2) of the six-node graphs under the dB-dB rule. For each
pair it places the initial cooperator and the initial mutant on the same node
v, averages theta_1 - theta_3, theta_2 and phi_{2,0} over v = 0, ..., 5, and
stores these averages with nine structural properties of each layer in
`r3_n6_exhaustive_results.csv`. Its two PNG files are exploratory and are not
part of S1 Text.

`r3_naoki_slope_bilayer_table.py` keeps the pairs whose G_1 has `k_coop` = 6
(G_2 is not restricted), forms `|d(b/c)*/dr| = |phi_{2,0} / (theta_1 - theta_3)|`
with c = 1 from the stored averages, and computes its Pearson correlation
with each property of G_1 ("layer 1") and of G_2 ("layer 2"). Table A uses
the rows mean_degree, max_degree, var_degree, mean_closeness, clustering,
diameter and alg_conn of `r3_part_4_bilayer_slope_table.csv`; the file also
contains a mean_betweenness row.

### Table A, larger column, and Table B

Both items use ER networks (p = 0.1, generated with
`networkx.erdos_renyi_graph`) and BA networks (m = 2, generated with
`ba_julia.ba_julia`) with N = 15, 30, 50, 75 and 100, one instance each with
seeds fixed in the code, and the two empirical networks in
`data/empirical_networks/`. The seeds and the construction are documented in
`r3_larger_networks_partA.get_network` and
`r3_larger_networks_partB.collect_bilayers`. The three `r3_larger_networks_*`
scripts locate `data/empirical_networks/` relative to their own file.

* Table B (`r3_larger_networks_partA.py`): for every node v of layer 1 of each
  network, `(b/c)*_single` with the initial cooperator on v, and the Pearson
  correlation across nodes between `(b/c)*_single` and the degree, closeness,
  betweenness and local clustering coefficient of v. `r3_part_A_table.csv`
  lists the networks with `(b/c)*_single` > 0 for every v; its `R_*` columns
  are Table B. An empty `R_clustering` cell means that the local clustering
  coefficient is 0 at every node of that network.
* Table A, larger column (`r3_larger_networks_partB_extended.py`): for each
  two-layer network (ER, BA, VC7 and LLF; dB-dB rule), `|d(b/c)*/dr|` with
  the initial cooperator and the initial mutant on the same node v, averaged
  over v; then the Pearson correlation of this average with seven metrics of
  layer 1 and of layer 2, across the two-layer networks whose layer 1 gives
  `(b/c)*_single` > 0 for every v (11 networks; VC7 is not among them). The
  "layer 1" rows of `r3_part_B_extended_correlations.csv` are the Table A
  column.
  `r3_larger_networks_partB.py` provides the network construction and the
  solver for this script. Run on its own, it writes a four-metric version of
  the same correlations (`r3_part_B_per_network.csv`,
  `r3_part_B_correlations.csv`), which is not part of S1 Text.

### Fig C

`r3_strategy_coupling_ic.py` simulates the dB-dB dynamics (delta = 0.02,
c = 1, r = 4) on five two-layer six-node networks with 5 x 10^5 runs per
panel. After each update step listed in `TIME_BINS` it records the Pearson
correlation coefficient across nodes between the layer-1 state (defector 0,
cooperator 1) and the layer-2 state (resident 0, mutant 1), averaged over all
runs, over the runs in which cooperation fixates and over the runs in which
defection fixates. The five network pairs are hard-coded in `NETS_A`; the
module docstring lists their edge lists and describes how they were selected
from `r3_n6_exhaustive_results.csv`. The top row of Fig C uses the correlated
initial condition (cooperator and mutant on node 0) and the bottom row the
uncorrelated one (cooperator on node 0, mutant on node 1); in each panel b/c
is 1.5 times `(b/c)*_two` of that panel's network and initial condition. The
script prints the table `(b/c)* per panel` with the values quoted in the
Fig C caption.

The environment variable `SC_R=4` sets r = 4 and the `_r4` suffix of the
file names. Without it the script uses r = 5, which is not an SI item. Of the
four PNG files written, `r3_sc_versionA_conv_ii_r4.png` is Fig C,
`r3_sc_versionB_conv_ii_r4.png` is an identical copy, and the two
`*_conv_i_r4.png` files draw the uncorrelated initial condition with the
b values of the top row; they are not part of S1 Text. The simulation kernel
is serial and seeded, so the full run regenerates the CSV and the PNG files
byte for byte.

### Fig G

`run_ws_validation_ring_baba.py` compares the weak-selection prediction of
the fixation probability of cooperation with direct simulations
(`weak_selection_sim_fast.run_sim`) on a two-layer ring with N = 10 and a
two-layer BA network with N = 15, under both updating rules and at
delta = 0.02 and 0.2. The theory uses
`weak_selection_validation.analytical_summary`. The networks, initial
conditions, b/c grids and numbers of runs are listed in the module docstring.

The plotting step, `plot()`, reads the stored CSV
`ws_validation_ring_baba_results.csv` included in this folder, and the
published Fig G is drawn from this file. The plot command in the table writes
`ws_validation_ring_baba.png`; with matplotlib 3.10.0 and its default backend
on macOS the file is identical to `published_figures/ws_validation_ring_baba.png`.
The full run recomputes the theory, repeats the simulations and replaces the
CSV before plotting.

### Fig H

`fig4_style_random.py 100` builds one two-layer ER network (p = 0.1) and one
two-layer BA network (m = 2, layer 2 relabelled by
`ba_julia.shuffle_layer2`) with N = 100, computes theta_n and phi_{n,m} with
`larger_networks_solver.bc_star_two_layer` under dB-dB and dB-Bd with the
initial cooperator and mutant on node 0, and colours the (b/c, r) plane by
whether the cooperator, the mutant, both or neither are favored, with the
colours of main-text Fig 4. The seeds are listed in the module docstring.
The theory inputs of the four panels are written to
`fig4_random_N100_results.csv` (its `bc_star_two` column is evaluated at
r = 2).

To redraw Fig H from the stored CSV without solving the linear systems (the
four CSV rows are passed to `make_figure` in place of the solver calls):

    python -c "import pandas as pd, fig4_style_random as m; it = iter(pd.read_csv('fig4_random_N100_results.csv', float_precision='round_trip').to_dict('records')); m.bc_star_two_layer = lambda *a, **k: next(it); m.make_figure(100, 'fig4_random_N100.png')"

Both the full run and the redraw reproduce
`published_figures/fig4_random_N100.png` byte for byte, and the full run also
reproduces `fig4_random_N100_results.csv` byte for byte.

### Fig I

`point_b_mc_overlay.py` compares the weak-selection prediction of the
fixation probability of cooperation (theory from
`larger_networks_solver.bc_star_two_layer`) with direct simulations
(`weak_selection_sim_fast.run_sim`) on two-layer ER and BA networks with
r = 5. The networks are built by `get_network`, and the published panels are
ER and BA with N = 100 under dB-dB and dB-Bd at delta = 0.02 and 0.2.

The plotting step, `python point_b_mc_overlay.py --plot-only`, reads the
stored CSV `point_b_mc_results.csv` included in this folder, and the
published Fig I is drawn from this file (`plot(ns=[100])`, i.e. the rows with
N = 100). With matplotlib 3.10.0 and its default backend on macOS the output
`point_b_mc_overlay.png` is identical to `published_figures/point_b_mc_overlay.png`.
The stored CSV also contains rows for smaller networks, which the figure
does not use.

A full run computes the rows defined by `CONFIG`, `DELTAS` and `RULE` in the
script (dB-dB, delta = 0.02), replaces the CSV with them and plots them. The
other blocks of the stored CSV were added by separate runs with other
settings; the module docstring shows how to compute such a block with
`run_one` and append it to the CSV.

### Simulations in Fig G and Fig I

`weak_selection_sim_fast.simulate_many` runs the replicates in a numba
`prange` loop. With more than one numba thread the random streams of the
worker threads are not fixed by the seed, so the simulated fixation
probabilities are reproducible in distribution but not digit for digit.
`NUMBA_NUM_THREADS` sets the number of threads; no other environment
variable is needed.

## Environment

* The stored results were produced with Python 3.13.9 (Anaconda) and numpy
  2.1.3, networkx 3.4.2, pandas 2.2.3, scipy 1.15.3, matplotlib 3.10.0 and
  numba 0.61.0 on macOS (Apple M1).
* numba is needed for Fig C, Fig G and Fig I (`weak_selection_sim_fast.py`,
  `r3_strategy_coupling.py`). numba writes compiled-code cache files into
  `__pycache__/`, which `.gitignore` excludes.
* `r3_naoki_revisions.py` (Fig A, Fig B, Table A) needs matplotlib < 3.11,
  because it calls `boxplot(labels=...)`, an argument name that
  matplotlib 3.11 removes.
* The networkx version matters for the six-node analyses (Fig A, Fig B,
  Fig C and the six-node columns of Table A), whose graph indices refer to
  the order of `networkx.graph_atlas_g()`. The five network pairs of Fig C
  are given by such indices; `r3_strategy_coupling_ic.py` records the edge
  lists of these graphs so that the indices can be checked with other
  networkx versions.
* The plots of Fig G and Fig I import `legend_utils.py` from this folder.
* The three `r3_larger_networks_*` scripts find `data/empirical_networks/`
  relative to their own file, and `julia_ba_loader.py` finds
  `graphs_ba_15.json` in the same way. The other inputs are read from the
  working directory, and all outputs are written to it, so run the commands
  from this folder.
* Julia is needed only for the two exporters `export_ba_n15.jl` and
  `export_ba_perms.jl`, which no SI item uses.

## Python files

| File | Type | Role |
|---|---|---|
| `ba_julia.py` | shared module | BA graphs grown by `nx.barabasi_albert_graph` from the star on m + 1 nodes centred at node m (the construction of Julia's `Graphs.barabasi_albert`), and `shuffle_layer2`, a seeded random relabelling of layer 2. Used by `fig4_style_random.py` and the `r3_larger_networks_*` scripts. |
| `fig4_style_random.py` | entry script | Fig H: (b/c, r) region maps for one two-layer ER and one two-layer BA network of size N (argument `100` for Fig H). |
| `julia_ba_loader.py` | shared module | Loader for `graphs_ba_15.json` (BA networks with N = 15 and layer-2 permutations of the main-text Julia run). Not imported by any script in this folder. |
| `larger_networks_solver.py` | shared module | Sparse solver for theta_n and phi_{n,m} of a two-layer network under dB-dB or dB-Bd; `bc_star_two_layer` returns them with `(b/c)*_single` and `(b/c)*_two`. Used for Figs A, B, C, H, I and Tables A, B. |
| `legend_utils.py` | shared module | `draw_uniform_legend`, which draws the bottom legend of Fig G and Fig I with equally spaced markers. |
| `point_b_mc_overlay.py` | entry script | Fig I: fixation probability of cooperation, theory versus simulation, on ER and BA networks. |
| `r3_larger_networks_partA.py` | entry script | Table B: `(b/c)*_single` for each initial node versus node centrality on the larger single-layer networks. Its `get_network` and constants are also used by the part B scripts. |
| `r3_larger_networks_partB.py` | shared module | Construction of the larger two-layer networks and the per-initial-condition solver for `\|d(b/c)*/dr\|`, used by `r3_larger_networks_partB_extended.py`. Run on its own it writes `r3_part_B_*.csv` (not part of S1 Text). |
| `r3_larger_networks_partB_extended.py` | entry script | Table A, larger column: correlation of `\|d(b/c)*/dr\|` with seven metrics of each layer. |
| `r3_naoki_6x6_heatmap.py` | entry script | Fig A: 5 x 5 Pearson correlation matrix read from `r3_naoki_per_graph.csv`. |
| `r3_naoki_revisions.py` | entry script | Fig B, and the per-graph six-node quantities (`r3_naoki_per_graph.csv`) and partial correlations of Text E.1. |
| `r3_naoki_slope_bilayer_table.py` | entry script | Table A, six-node columns, from `r3_n6_exhaustive_results.csv` and `r3_naoki_per_graph.csv`. |
| `r3_strategy_coupling.py` | shared module | Simulation kernels for Fig C (Pearson correlation between the layer states after each update step). Also contains an earlier stand-alone pipeline that is not used for any SI item. |
| `r3_strategy_coupling_ic.py` | entry script | Fig C: strategy-type correlation between the two layers. |
| `r3_topology_correlation_n6_exhaustive.py` | entry script | First step of the Table A six-node columns: the dB-dB scan over all 12,544 ordered pairs of six-node graphs, written to `r3_n6_exhaustive_results.csv`. |
| `run_ws_validation_ring_baba.py` | entry script | Fig G: fixation probability of cooperation, theory versus simulation, on a ring with N = 10 and a BA network with N = 15. |
| `weak_selection_sim_fast.py` | shared module | numba-compiled Monte Carlo simulator of the two-layer dynamics under dB-dB and dB-Bd; `run_sim` is used by Fig G and Fig I, and its update kernels by Fig C. |
| `weak_selection_validation.py` | shared module | Dense NumPy solver for theta_n and phi_{n,m} (`analytical_summary`, used for the theory of Fig G) and a pure-NumPy reference simulator. |

Running `larger_networks_solver.py`, `weak_selection_validation.py`,
`weak_selection_sim_fast.py`, `ba_julia.py` or `julia_ba_loader.py` directly
runs a check or demonstration that prints to the terminal and is not used
for any SI item (see the module docstrings).

## Other files

Julia exporters and their output (kept for reference; not used for any SI
item):

| File | Role |
|---|---|
| `export_ba_n15.jl` | Exports the BA networks with N = 15 and the layer-2 shuffle seeds of the main-text Julia run from Julia-serialized files to JSON. Its input `.dat` files are not included. `export_ba_perms.jl` supersedes it. |
| `export_ba_perms.jl` | Same export, plus the layer-2 permutations generated by Julia's `MersenneTwister` from the shuffle seeds; writes `graphs_ba_15.json` and `ba_theta_fingerprint.json` (theta and phi of one BA pair from the main-text Julia run; not included). Its input `.dat` files are not included. |
| `graphs_ba_15.json` | Output of `export_ba_perms.jl`, read by `julia_ba_loader.py`. |

Stored results (written by the scripts above):

| File | Written by | Content |
|---|---|---|
| `r3_naoki_per_graph.csv` | `r3_naoki_revisions.py` | 112 rows, one per six-node graph: `theta_diff_mean`, `k_coop`, mean degree, mean closeness, mean betweenness, clustering |
| `r3_partial_correlations.csv` | `r3_naoki_revisions.py` | partial correlations with the mean degree partialled out (Text E.1) |
| `r3_per_node_centrality_results.csv`, `r3_per_node_centrality_summary.csv` | `r3_naoki_revisions.py` | correlation coefficients plotted in Fig B, and their number, mean and standard deviation per centrality |
| `r3_pairwise_5x5_R.csv`, `r3_pairwise_5x5_p.csv` | `r3_naoki_6x6_heatmap.py` | correlation coefficients drawn in Fig A, and their p-values |
| `r3_n6_exhaustive_results.csv` | `r3_topology_correlation_n6_exhaustive.py` | 12,544 ordered pairs of six-node graphs: averaged theta_1 - theta_3, theta_2, phi_{2,0}, thresholds and structural properties of both layers |
| `r3_part_4_bilayer_slope_table.csv`, `.txt` | `r3_naoki_slope_bilayer_table.py` | six-node columns of Table A |
| `r3_part_A_table.csv`, `r3_part_A_per_network.csv`, `r3_part_A_per_node.csv` | `r3_larger_networks_partA.py` | Table B, per-network correlations, and per-node `(b/c)*_single` and centralities |
| `r3_part_B_extended_correlations.csv`, `r3_part_B_extended_per_network.csv` | `r3_larger_networks_partB_extended.py` | larger column of Table A (rows "layer 1") and per-network `\|d(b/c)*/dr\|` and metrics |
| `r3_part_B_correlations.csv`, `r3_part_B_per_network.csv` | `r3_larger_networks_partB.py` | four-metric version of the same correlations (not part of S1 Text) |
| `r3_strategy_coupling_ic_results_r4.csv` | `r3_strategy_coupling_ic.py` | correlation trajectories and `(b/c)*` values of Fig C (panels `top` and `bot_ii`) and of the variant `bot_i` |
| `ws_validation_ring_baba_results.csv` | `run_ws_validation_ring_baba.py` | 80 rows: theory inputs, predictions and simulation results; read by the plot step of Fig G |
| `fig4_random_N100_results.csv` | `fig4_style_random.py` | 4 rows: theory inputs of the four panels of Fig H |
| `point_b_mc_results.csv` | `point_b_mc_overlay.py` | 128 rows: theory inputs, predictions and simulation results; the N = 100 rows are read by the plot step of Fig I |

## Published figures

`published_figures/` holds the figure files exactly as they appear in
S1 Text.

| SI figure | File in `published_figures/` | Written by (output file name) | MD5 |
|---|---|---|---|
| Fig A | `fig1_R3_pairwise_5x5.png` | `r3_naoki_6x6_heatmap.py` (`fig1_R3_pairwise_5x5.png`) | 4b648d6ff9ca761985d31e1a560a3a2d |
| Fig B | `fig5_R3_per_node_centrality.png` | `r3_naoki_revisions.py` (`fig5_R3_per_node_centrality.png`) | cccbff1179f6353c317da1a103c7f9ca |
| Fig C | `FigC_strategy_coupling_r4.png` | `r3_strategy_coupling_ic.py` with `SC_R=4` (`r3_sc_versionA_conv_ii_r4.png`) | 364bf9e17bb6209217038e93178f3b18 |
| Fig G | `ws_validation_ring_baba.png` | `run_ws_validation_ring_baba.py`, `plot()` of the stored CSV (`ws_validation_ring_baba.png`) | 021dc8ad278803b2513177f9ac75979f |
| Fig H | `fig4_random_N100.png` | `fig4_style_random.py 100` (`fig4_random_N100.png`) | e6b7fd1d4abc26049dd2b3ebd11436b0 |
| Fig I | `point_b_mc_overlay.png` | `point_b_mc_overlay.py`, `plot()` of the stored CSV (`point_b_mc_overlay.png`) | a2bcf8cd3fe43a2bae35f71f7c17f7d6 |

Tables A and B were typeset from the CSV files named above and have no image
file.

With the environment above, the following outputs are regenerated byte for
byte: Fig A, Fig B and their CSV files; Fig C and its CSV (full run and plot
only); Fig H and its CSV (full run and redraw); the CSV files of Table B, of
the larger column of Table A and of `r3_larger_networks_partB.py`; and the
six-node table of Table A when computed from the stored
`r3_n6_exhaustive_results.csv`. A fresh exhaustive scan gives the same
`r3_n6_exhaustive_results.csv` except in the `g1_alg_conn` and `g2_alg_conn`
columns, which differ by at most 4e-15 because `nx.algebraic_connectivity`
uses an iterative eigensolver with a random start; the text table
`r3_part_4_bilayer_slope_table.txt` computed from a fresh scan is identical.
Fig G and Fig I are regenerated byte for byte from the stored CSV files by
their plot steps (matplotlib 3.10.0, default macOS backend).

## Data: `data/empirical_networks/`

* `edges.csv`: the Vickers and Chan 7th graders network (VC7), a multilayer
  network of nominations among seventh-grade students of a school in
  Victoria, Australia. Original source: M. Vickers and S. Chan,
  "Representing classroom social structure", Victoria Institute of
  Secondary Education, Melbourne (1981).
* `law_edges.csv`: the Lazega law firm network (LLF), a multilayer network of
  relationships among the lawyers of a corporate law firm. Original source:
  E. Lazega, "The Collegial Phenomenon: The Social Mechanisms of Cooperation
  Among Peers in a Corporate Law Partnership", Oxford University Press
  (2001).

Both files were obtained as the edge files (`edges.csv`) of the CSV downloads
of the Netzschleuder network catalogue, entries `7th_graders`
(https://networks.skewed.de/net/7th_graders) and `law_firm`
(https://networks.skewed.de/net/law_firm), which give
https://manliodedomenico.com/data.php as the upstream source. Each file
starts with the comment line `# source, target, weight, layer`, followed by
one row per edge with 0-based node ids, the weight and an integer layer id.

`r3_larger_networks_partA.py` reads the rows with layer id 1 of each file.
`r3_larger_networks_partB.py` reads the rows with layer ids 1 and 2 as
layer 1 and layer 2 of the two-layer network. The loaders drop self-loops,
merge duplicate and reciprocal rows into undirected, unweighted edges, and
relabel the nodes 0, ..., n - 1, so node labels in the output CSV files are
not the node ids of the edge files.
