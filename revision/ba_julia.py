"""
Barabasi-Albert (BA) graphs built with the node-labelling convention of
Julia's Graphs.jl, and a seeded relabelling of layer 2.

The BA networks of the main text were generated in Julia with
Graphs.jl barabasi_albert(N, m) (keyword complete=false, the default). Once
m + 1 nodes exist, that construction has the star on m + 1 nodes centred at
node m (0-based; leaves 0, ..., m-1); preferential attachment then adds
nodes m + 1, ..., N - 1, each with m edges. ba_julia reproduces this by
passing that star to nx.barabasi_albert_graph as the initial graph.
nx.barabasi_albert_graph with its default initial graph (nx.star_graph(m))
labels the nodes differently, so for the same seed the two give different
labelled graphs.

Functions
---------
ba_julia_initial(m)          the star on m + 1 nodes centred at node m
ba_julia(N, m, seed)         nx.barabasi_albert_graph(N, m, seed=seed,
                             initial_graph=ba_julia_initial(m))
ba_julia_first_connected(N, m, rng)
                             ba_julia with seeds drawn from rng; returns the
                             first accepted graph and its seed
apply_perm(G, perm)          relabel nodes by an explicit permutation
shuffle_layer2(G2, seed)     relabel nodes by a uniformly random permutation
                             drawn from np.random.default_rng(seed)

Use in this folder
------------------
ba_julia is imported by fig4_style_random.py (Fig H) and
r3_larger_networks_partA.py (Table B; its get_network is also used by
r3_larger_networks_partB.py). shuffle_layer2 is imported by
fig4_style_random.py and r3_larger_networks_partB.py (the larger column of
Table A, through r3_larger_networks_partB_extended.py). ba_julia_initial is
called by ba_julia. ba_julia_first_connected and apply_perm are not called
by any script in this folder. apply_perm, given a permutation from
julia_ba_loader.load_julia_shuffle_perm, gives the layer-2 relabelling of
the main-text Julia run; numpy cannot produce Julia's permutation from the
same seed, because the random number generators differ.

Seeds: the graphs are drawn by networkx (Python's random module) and the
permutations by numpy, and both random number generators differ from
Julia's, so a given seed gives different graphs and permutations in Python
and in Julia. Julia seeds are signed Int64; ba_julia_first_connected draws
non-negative seeds, and np.random.default_rng (used by shuffle_layer2)
accepts only non-negative seeds.

Running this file
-----------------
    python ba_julia.py
runs smoke tests (edges of ba_julia_initial(2), a small ba_julia graph, and
the seed determinism of ba_julia_first_connected and shuffle_layer2) and
prints "All smoke tests passed." Inputs: none. Output: stdout only.
Runtime: under 1 s.

Dependencies: networkx, numpy.
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def ba_julia_initial(m: int) -> nx.Graph:
    """Star graph on m + 1 nodes centred at node m.

    Parameters
    ----------
    m : number of edges per new node in the BA construction (m >= 1,
        otherwise ValueError).

    Returns
    -------
    networkx.Graph with nodes 0, ..., m and edges (m, 0), ..., (m, m-1).
    This is the graph that Julia's Graphs.jl barabasi_albert(N, m) (default
    complete=false) has once m + 1 nodes exist: node m is attached to each
    of the m nodes 0, ..., m-1. ba_julia passes it to nx.barabasi_albert_graph
    as initial_graph, so that preferential attachment continues from it.
    For m = 1 it is the single edge (1, 0). The scripts in this folder use
    m = 2 (edges (2, 0) and (2, 1)).
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    G = nx.empty_graph(m)
    G.add_node(m)
    G.add_edges_from([(m, i) for i in range(m)])
    return G


def ba_julia(N: int, m: int, seed: int) -> nx.Graph:
    """BA graph with N nodes and m edges per new node, Julia labelling.

    Returns nx.barabasi_albert_graph(N, m, seed=seed,
    initial_graph=ba_julia_initial(m)): starting from the star on m + 1
    nodes centred at node m, nodes m + 1, ..., N - 1 are added in order,
    each linked to m distinct existing nodes chosen with probability
    proportional to their degree. The graph has N nodes (0, ..., N-1) and
    m (N - m) edges. Raises ValueError if N < m + 1.
    """
    if N < m + 1:
        raise ValueError(f"N={N} too small for m={m} (need N >= m+1)")
    return nx.barabasi_albert_graph(
        N, m, seed=seed, initial_graph=ba_julia_initial(m),
    )


def ba_julia_first_connected(
    N: int, m: int, rng: np.random.Generator, max_tries: int = 1000,
) -> tuple[nx.Graph, int]:
    """ba_julia(N, m, s) for seeds s drawn from rng; returns (G, s).

    Parameters
    ----------
    N, m      : as in ba_julia.
    rng       : numpy Generator; each draw is s = rng.integers(0, 2**31 - 1).
    max_tries : maximum number of draws.

    Returns the graph ba_julia(N, m, seed=s) and its seed s for the first
    draw accepted by the test in the loop below; raises RuntimeError if no
    draw is accepted within max_tries. The same rng state gives the same
    result. Not called by any script in this folder.
    """
    for _ in range(max_tries):
        s = int(rng.integers(0, 2**31 - 1))
        G = ba_julia(N, m, seed=s)
        if nx.is_connected(G):
            return G, s
    raise RuntimeError(f"could not generate connected BA(N={N}, m={m}) in {max_tries} tries")


def apply_perm(G: nx.Graph, perm: list[int]) -> nx.Graph:
    """Relabel the nodes of G by an explicit permutation: old label i -> perm[i].

    Parameters
    ----------
    G    : graph with nodes 0, ..., N-1.
    perm : sequence of length N containing 0, ..., N-1 exactly once
           (otherwise ValueError).

    Returns a new graph in which the edge (u, v) of G becomes
    (perm[u], perm[v]). This is the direction of the endpoint relabelling
    edge_seq[:, col] = [perm[x] for x in edge_seq[:, col]] in the Julia
    code of the main text. With julia_ba_loader.load_julia_shuffle_perm, it
    can apply the layer-2 permutations of the main-text Julia run, which are
    stored in graphs_ba_15.json because numpy cannot produce Julia's
    MersenneTwister permutation from the shuffle seed. Not called by any
    script in this folder.
    """
    N = G.number_of_nodes()
    if sorted(perm) != list(range(N)):
        raise ValueError(f"perm is not a permutation of 0..{N-1}")
    mapping = {i: int(perm[i]) for i in range(N)}
    return nx.relabel_nodes(G, mapping)


def shuffle_layer2(G2: nx.Graph, seed: int) -> tuple[nx.Graph, list[int]]:
    """Relabel the nodes of G2 by a uniformly random permutation.

    Parameters
    ----------
    G2   : graph with nodes 0, ..., N-1 (layer 2 of a two-layer network).
    seed : seed of np.random.default_rng; perm = rng.permutation(N).

    Returns
    -------
    (G2_relabelled, perm)
    G2_relabelled : the graph with old label i replaced by perm[i].
    perm          : the permutation as a list of ints; node i of the input
                    becomes node perm[i], which can be used to follow a
                    given node (such as the initial mutant) through the
                    relabelling.

    The relabelling follows the Julia code of the main text
        perm = shuffle(rng, collect(1:N))
        edge_seq[:, j] = [perm[x] for x in edge_seq[:, j]]
    (1-based there, 0-based here), but with numpy's random number generator,
    so a given seed does not give Julia's permutation. The permutation
    depends only on `seed`, so each relabelled layer can be regenerated from
    its seed alone, independently of other random draws.
    """
    N = G2.number_of_nodes()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    mapping = {i: int(perm[i]) for i in range(N)}
    return nx.relabel_nodes(G2, mapping), perm.tolist()


if __name__ == "__main__":
    # Smoke tests: initial star, a small ba_julia graph, and determinism of
    # ba_julia_first_connected and shuffle_layer2 for fixed seeds.
    print("=== ba_julia_initial(m=2) ===")
    G = ba_julia_initial(2)
    print(f"  N={G.number_of_nodes()}, E={G.number_of_edges()}")
    print(f"  edges: {sorted(G.edges())}")
    assert sorted(G.edges()) == [(0, 2), (1, 2)], "should be star centred at node m=2"

    print("\n=== ba_julia(N=5, m=2, seed=42) ===")
    G = ba_julia(5, 2, seed=42)
    print(f"  N={G.number_of_nodes()}, E={G.number_of_edges()}")
    print(f"  edges: {sorted(G.edges())}")

    print("\n=== ba_julia_first_connected(N=15, m=2) determinism ===")
    rng1 = np.random.default_rng(123)
    G_a, s_a = ba_julia_first_connected(15, 2, rng1)
    rng2 = np.random.default_rng(123)
    G_b, s_b = ba_julia_first_connected(15, 2, rng2)
    print(f"  G_a edges == G_b edges: {sorted(G_a.edges()) == sorted(G_b.edges())}")
    print(f"  seed_a == seed_b:       {s_a == s_b}")
    assert sorted(G_a.edges()) == sorted(G_b.edges())

    print("\n=== shuffle_layer2 determinism ===")
    G = ba_julia(15, 2, seed=42)
    G_sh1, perm1 = shuffle_layer2(G, seed=999)
    G_sh2, perm2 = shuffle_layer2(G, seed=999)
    print(f"  shuffled edges identical: {sorted(G_sh1.edges()) == sorted(G_sh2.edges())}")
    print(f"  perms identical:          {perm1 == perm2}")
    assert perm1 == perm2

    print("\nAll smoke tests passed.")
