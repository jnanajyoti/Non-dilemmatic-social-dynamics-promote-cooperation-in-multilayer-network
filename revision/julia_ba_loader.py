"""
Loader for the BA networks with N = 15 of the main-text Julia run and for
the layer-2 permutations used to pair them into two-layer networks.

Data file
---------
graphs_ba_15.json (in the same folder as this module; path
_DEFAULT_JSON_PATH) was written by export_ba_perms.jl from the Julia files
graphs_ba_15.dat and shuffle_seeds_ba_15_dB_{dB,Bd}_shuffled.dat of the
main-text run (these .dat files are not included). It contains
    graphs["k=K"]          for K = 1, ..., 5: a list of 5 instances, each
                           {"N": 15, "seed": Julia seed, "edges": 0-based
                           edge list} of Graphs.jl barabasi_albert(15, K);
    shuffle_seeds_dB_dB,   625 rows each, one per ordered pair of the 25
    shuffle_seeds_dB_Bd    instances (an instance may be paired with
                           itself), (k1, seed1) for layer 1 and (k2, seed2)
                           for layer 2, with the Julia shuffle seed (decimal
                           string, since it is a UInt64) and "perm", the
                           layer-2 permutation (0-based,
                           new_label = perm[old_label]);
    meta                   provenance of the file.
According to meta["rule_provenance"], the main-text theta and phi
computations used the dB-dB shuffle-seed table for both update rules, so
rule="dB-dB" gives the main-text two-layer networks under either rule.

The permutation is derived in Julia as
shuffle(MersenneTwister(shuffle_seed), collect(1:N)); numpy cannot produce
the same permutation from the seed, so load_julia_shuffle_perm (which
returns the stored permutation) is the function to use for the main-text
two-layer networks, with ba_julia.apply_perm.

Use in this folder
------------------
No script in this folder imports this module. It is kept with
graphs_ba_15.json and the two Julia exporters for reference.

Running this file
-----------------
    python julia_ba_loader.py
runs smoke tests (loads instance 0 for K = 2, lists the K = 2 seeds, looks up
one shuffle seed, and checks the edge count of all 25 instances) and prints
"All smoke tests passed." Input: graphs_ba_15.json. Output: stdout only.
Runtime: under 1 s.

Dependencies: networkx (and the standard library json, os, functools).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import networkx as nx


# graphs_ba_15.json next to this file (independent of the working directory).
_DEFAULT_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "graphs_ba_15.json"
)


@lru_cache(maxsize=1)
def _load_json_once(path: str = _DEFAULT_JSON_PATH) -> dict:
    """Parse the JSON file at `path` and return it as a dict.

    The result for the most recent path is cached, so repeated calls with
    the same path do not read the file again.
    """
    with open(path) as f:
        return json.load(f)


def load_julia_ba_n15(k: int, instance_idx: int = 0,
                       path: str = _DEFAULT_JSON_PATH) -> tuple[nx.Graph, int]:
    """One saved BA network with N = 15 and k edges per new node.

    Parameters
    ----------
    k            : BA parameter m, 1 to 5.
    instance_idx : which of the saved instances for this k, 0 to 4
                   (otherwise IndexError).
    path         : JSON file (default graphs_ba_15.json next to this module).

    Returns
    -------
    (G, julia_seed): a networkx.Graph with nodes 0, ..., 14 and the stored
    edge list, and the Julia seed of the instance as an int.
    """
    data = _load_json_once(path)
    entries = data["graphs"][f"k={k}"]
    if not 0 <= instance_idx < len(entries):
        raise IndexError(
            f"instance_idx={instance_idx} out of range; "
            f"have {len(entries)} instances for k={k}"
        )
    entry = entries[instance_idx]
    G = nx.Graph()
    G.add_nodes_from(range(entry["N"]))
    G.add_edges_from(tuple(e) for e in entry["edges"])
    return G, int(entry["seed"])


def load_julia_shuffle_seed(
    k1: int, k2: int, seed1: int, seed2: int,
    rule: str = "dB-dB",
    path: str = _DEFAULT_JSON_PATH,
) -> int:
    """Julia shuffle seed of layer 2 for one pair of saved instances.

    Parameters
    ----------
    k1, seed1 : BA parameter and Julia seed of the layer-1 instance.
    k2, seed2 : BA parameter and Julia seed of the layer-2 instance.
    rule      : "dB-dB" or "dB-Bd", selecting the table shuffle_seeds_dB_dB
                or shuffle_seeds_dB_Bd (otherwise ValueError).
    path      : JSON file (default graphs_ba_15.json next to this module).

    Returns the shuffle seed as a Python int (stored as a decimal string in
    the JSON because a UInt64 can exceed 2^53, the largest integer that many
    JSON readers store exactly). Raises
    KeyError if the pair is not in the table. The seed alone does not give
    the permutation in Python; see load_julia_shuffle_perm.
    """
    data = _load_json_once(path)
    if rule == "dB-dB":
        table = data["shuffle_seeds_dB_dB"]
    elif rule == "dB-Bd":
        table = data["shuffle_seeds_dB_Bd"]
    else:
        raise ValueError(f"unknown rule: {rule!r}")
    for row in table:
        if (row["k1"] == k1 and row["k2"] == k2
                and row["seed1"] == seed1 and row["seed2"] == seed2):
            return int(row["shuffle_seed"])
    raise KeyError(
        f"no shuffle seed for (k1={k1}, k2={k2}, seed1={seed1}, seed2={seed2}, rule={rule!r})"
    )


def load_julia_shuffle_perm(
    k1: int, k2: int, seed1: int, seed2: int,
    rule: str = "dB-dB",
    path: str = _DEFAULT_JSON_PATH,
) -> list[int]:
    """Layer-2 permutation of the Julia run for one pair of saved instances.

    Parameters are as in load_julia_shuffle_seed.

    Returns
    -------
    perm : list of 15 ints, 0-based; perm[i] is the new label of the node
           with old label i (the direction of the Julia code's perm[x]
           endpoint relabelling). ba_julia.apply_perm(G2, perm) applies it.

    The permutation was computed in Julia by export_ba_perms.jl as
    shuffle(MersenneTwister(shuffle_seed), collect(1:N)) and stored in the
    JSON. Raises ValueError for an unknown rule, and KeyError if the pair is
    not in the table or the row has no "perm" field (a JSON written by
    export_ba_n15.jl has none).
    """
    data = _load_json_once(path)
    if rule == "dB-dB":
        table = data["shuffle_seeds_dB_dB"]
    elif rule == "dB-Bd":
        table = data["shuffle_seeds_dB_Bd"]
    else:
        raise ValueError(f"unknown rule: {rule!r}")
    for row in table:
        if (row["k1"] == k1 and row["k2"] == k2
                and row["seed1"] == seed1 and row["seed2"] == seed2):
            if "perm" not in row:
                raise KeyError(
                    "JSON has no 'perm' field — regenerate it with "
                    "export_ba_perms.jl (export_ba_n15.jl output lacks perms)"
                )
            return [int(p) for p in row["perm"]]
    raise KeyError(
        f"no shuffle entry for (k1={k1}, k2={k2}, seed1={seed1}, seed2={seed2}, rule={rule!r})"
    )


def list_julia_seeds(k: int, path: str = _DEFAULT_JSON_PATH) -> list[int]:
    """Julia seeds of the saved instances for BA parameter k, as ints, in
    instance order (index 0 to 4)."""
    data = _load_json_once(path)
    return [int(e["seed"]) for e in data["graphs"][f"k={k}"]]


if __name__ == "__main__":
    # Smoke tests of the loader functions on graphs_ba_15.json.
    print("=== load_julia_ba_n15(k=2, instance_idx=0) ===")
    G, seed = load_julia_ba_n15(2, 0)
    print(f"  N={G.number_of_nodes()}, E={G.number_of_edges()}, seed={seed}")
    print(f"  is_connected: {nx.is_connected(G)}")
    assert G.number_of_nodes() == 15
    assert G.number_of_edges() == 26   # k (N - k) = 2 x 13 edges
    assert nx.is_connected(G)

    print("\n=== seeds at k=2 ===")
    seeds = list_julia_seeds(2)
    print(f"  {seeds}")

    print("\n=== Julia shuffle seed for (k=2, inst 0, k=2, inst 1) under dB-dB ===")
    G1, s1 = load_julia_ba_n15(2, 0)
    G2, s2 = load_julia_ba_n15(2, 1)
    shuffle_seed = load_julia_shuffle_seed(2, 2, s1, s2, rule="dB-dB")
    print(f"  s1={s1}, s2={s2}, shuffle_seed={shuffle_seed} (Python int)")

    # Edge count k (N - k) of every saved instance (5 values of k x 5 instances).
    print("\n=== edge-count sanity for all 25 saved instances ===")
    expected_edges = {1: 14, 2: 26, 3: 36, 4: 44, 5: 50}
    for k in range(1, 6):
        for idx in range(5):
            G, _ = load_julia_ba_n15(k, idx)
            ok = G.number_of_edges() == expected_edges[k]
            print(f"  k={k}, idx={idx}: E={G.number_of_edges()} (expected {expected_edges[k]}): {'OK' if ok else 'FAIL'}")
            assert ok

    print("\nAll smoke tests passed.")
