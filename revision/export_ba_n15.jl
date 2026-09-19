# export_ba_n15.jl
#
# One-time exporter: load the saved Julia BA(N=15) graphs and the per-pair
# layer-2 shuffle seeds, and write them as JSON so the Python pipeline can
# load the EXACT same physical (G1, G2, layer-2 perm) triples that backed
# the manuscript's main-text BA experiments.
#
# Inputs (on disk in this same directory):
#   graphs_ba_15.dat                            : Dict{Int, Vector{(SimpleGraph, Int64)}}
#   shuffle_seeds_ba_15_dB_dB_shuffled.dat      : Dict{NTuple{4,Int64}, UInt64}
#   shuffle_seeds_ba_15_dB_Bd_shuffled.dat      : same shape, for dB-Bd rule
#
# Output:
#   graphs_ba_15.json   : single JSON with both graph data and shuffle-seed lookup
#
# Output JSON structure:
# {
#   "graphs": {
#     "k=1": [{"seed": Int64, "N": 15, "edges": [[u,v], ...]} × 5],
#     ... k=2..5
#   },
#   "shuffle_seeds_dB_dB": [
#     {"k1": Int, "k2": Int, "seed1": Int64, "seed2": Int64, "shuffle_seed": UInt64},
#     ... × 625
#   ],
#   "shuffle_seeds_dB_Bd": [ ... × 625 ]
# }
#
# Node indices are 0-indexed in the JSON output (Python convention).
# Run with:   julia export_ba_n15.jl
# cd(@__DIR__) below makes the script read its inputs from, and write its
# output to, the folder that contains this file. The input .dat files are
# not included in this repository.

using Serialization
using Graphs
using JSON

cd(@__DIR__)   # relative paths resolve to the folder of this script

# ---- graphs ----
graphs = deserialize("graphs_ba_15.dat")  # Dict{Int, Vector{Tuple{SimpleGraph{Int}, Int64}}}
graphs_out = Dict{String, Any}()
for k in sort(collect(keys(graphs)))
    instances = []
    for (g, seed) in graphs[k]
        edges_0idx = [[src(e)-1, dst(e)-1] for e in edges(g)]   # 1-idx Julia -> 0-idx Python
        push!(instances, Dict("seed" => seed, "N" => nv(g), "edges" => edges_0idx))
    end
    graphs_out["k=$k"] = instances
end

# ---- shuffle seeds (both rules) ----
function load_shuffle_seeds(path::String)
    sh = deserialize(path)  # Dict{NTuple{4,Int64}, UInt64}
    out = []
    for (k, v) in sh
        k1, k2, seed1, seed2 = k
        # JSON cannot represent UInt64 > 2^53 reliably as a number, so emit as
        # a string and let the Python loader cast back to int.
        push!(out, Dict(
            "k1"            => k1,
            "k2"            => k2,
            "seed1"         => seed1,
            "seed2"         => seed2,
            "shuffle_seed"  => string(v),    # UInt64 as string for safe JSON
        ))
    end
    return out
end

shuffle_dBdB = load_shuffle_seeds("shuffle_seeds_ba_15_dB_dB_shuffled.dat")
shuffle_dBBd = load_shuffle_seeds("shuffle_seeds_ba_15_dB_Bd_shuffled.dat")

# ---- combine and write ----
out = Dict{String, Any}(
    "graphs"             => graphs_out,
    "shuffle_seeds_dB_dB" => shuffle_dBdB,
    "shuffle_seeds_dB_Bd" => shuffle_dBBd,
    "meta" => Dict(
        "source_graphs_file"        => "graphs_ba_15.dat",
        "source_shuffle_dBdB_file"  => "shuffle_seeds_ba_15_dB_dB_shuffled.dat",
        "source_shuffle_dBBd_file"  => "shuffle_seeds_ba_15_dB_Bd_shuffled.dat",
        "node_indexing"             => "0-indexed (Python convention)",
        "uint64_handling"           => "shuffle_seed stored as decimal string",
        "exporter_script"           => "export_ba_n15.jl",
        "generated_at_julia_version" => string(VERSION),
    ),
)

open("graphs_ba_15.json", "w") do io
    JSON.print(io, out, 2)
end

println("# Wrote graphs_ba_15.json")
println("#   graphs: ", sum(length(v) for v in values(graphs_out)), " total instances (5 per k, k=1..5)")
println("#   shuffle_seeds_dB_dB: ", length(shuffle_dBdB), " entries (expected 625 = 5^4)")
println("#   shuffle_seeds_dB_Bd: ", length(shuffle_dBBd), " entries (expected 625 = 5^4)")
