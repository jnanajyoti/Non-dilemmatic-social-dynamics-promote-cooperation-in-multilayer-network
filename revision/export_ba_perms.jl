# export_ba_perms.jl
#
# Second-stage exporter (supplements export_ba_n15.jl):
#
# 1. Re-derives the EXACT layer-2 permutation for every (k1, k2, seed1, seed2)
#    entry in both shuffle-seed files, using the same derivation as the
#    manuscript's `rebuild_edge_seq_pair` (Barabasi Albert-ff.ipynb cell 34):
#        rng  = MersenneTwister(shuffle_seed)
#        perm = shuffle(rng, collect(1:N))        # new_id = perm[old_id]
#    and rewrites graphs_ba_15.json with a "perm" field (0-indexed) added to
#    every shuffle row. This is necessary because numpy cannot reproduce
#    Julia's MersenneTwister permutation from the seed alone, so exporting the
#    seed is not enough to rebuild the same two-layer network in Python.
#
# 2. Exports the saved θ/ϕ values of the main-text Julia run for the pair
#    (k1=2 inst 0, k2=2 inst 1), all 15×15 (i, j) initial conditions, to
#    ba_theta_fingerprint.json: from theta_phi_ba_15_dB_{dB,Bd}_shuffled.dat
#    (pf goods scheme) and theta_phi_ba_15_dB_{dB,Bd}_shuffled_ff.dat (ff
#    goods scheme). With the exported perm, the same two-layer network can be
#    rebuilt in Python and θ/ϕ recomputed for comparison with these values.
#
# 3. Exports the shuffled layer-2 edge list for that pair (perm applied),
#    0-indexed, as a direct structural fingerprint.
#
# Provenance note (verified from the notebooks): the FINAL theta pipeline
# (cells 34/36/45 in 'Barabasi Albert-ff.ipynb', and cells 26/27/39 in
# 'Barabasi Albert.ipynb') loaded SHUFFLE_SEEDS_FILE =
# "shuffle_seeds_ba_15_dB_dB_shuffled.dat" and reused that same dict for the
# dB-Bd theta run. So the dB-dB seed file backs the bilayers of BOTH update
# rules in the manuscript data. The "_dB_Bd_" seed file was used only by the
# superseded Mar 2025 inline classification runs.
#
# Run with:   julia export_ba_perms.jl
# cd(@__DIR__) below makes the script read its inputs from, and write its
# outputs to, the folder that contains this file. The input .dat files are
# not included in this repository.

using Serialization
using Random
using Graphs
using JSON

cd(@__DIR__)

const N_BA = 15

# ---- load inputs ----
graphs = deserialize("graphs_ba_15.dat")  # Dict{Int, Vector{Tuple{SimpleGraph{Int}, Int64}}}
sh_dBdB = deserialize("shuffle_seeds_ba_15_dB_dB_shuffled.dat")  # Dict{NTuple{4,Int64}, UInt64}
sh_dBBd = deserialize("shuffle_seeds_ba_15_dB_Bd_shuffled.dat")

derive_perm(sseed::UInt64, N::Int) = shuffle(MersenneTwister(sseed), collect(1:N))

# ---- rebuild graphs section (same as export_ba_n15.jl) ----
graphs_out = Dict{String, Any}()
for k in sort(collect(keys(graphs)))
    instances = []
    for (g, seed) in graphs[k]
        edges_0idx = [[src(e)-1, dst(e)-1] for e in edges(g)]
        push!(instances, Dict("seed" => seed, "N" => nv(g), "edges" => edges_0idx))
    end
    graphs_out["k=$k"] = instances
end

# ---- shuffle tables with exact perms ----
function shuffle_table_with_perms(sh::Dict)
    out = []
    for (key, sseed) in sh
        k1, k2, seed1, seed2 = key
        perm = derive_perm(sseed, N_BA)
        push!(out, Dict(
            "k1"           => k1,
            "k2"           => k2,
            "seed1"        => seed1,
            "seed2"        => seed2,
            "shuffle_seed" => string(sseed),       # UInt64 as decimal string
            "perm"         => perm .- 1,           # 0-indexed: new0[i0] = perm[i0+1]-1
        ))
    end
    return out
end

shuffle_dBdB_out = shuffle_table_with_perms(sh_dBdB)
shuffle_dBBd_out = shuffle_table_with_perms(sh_dBBd)

# ---- pair (k=2 inst 0, k=2 inst 1): structural + theta fingerprints ----
(g1, s1) = graphs[2][1]   # inst 0 at k=2
(g2, s2) = graphs[2][2]   # inst 1 at k=2
pair_key = (2, 2, s1, s2)
@assert haskey(sh_dBdB, pair_key) "missing dB-dB shuffle seed for the ws pair"
sseed_pair = sh_dBdB[pair_key]
perm_pair = derive_perm(sseed_pair, N_BA)

# shuffled layer-2 edge list, exactly as rebuild_edge_seq_pair does it:
# relabel each endpoint x -> perm[x], then 0-index
shuffled_edges_0idx = [[perm_pair[src(e)]-1, perm_pair[dst(e)]-1] for e in edges(g2)]

# The files without "_ff" use the pf goods scheme of the main text
# (plain-M1 convention). The "_ff" files use the ff goods scheme of S1 Text,
# Text M (transpose-M1 convention); the two schemes give the same (b/c)* on
# regular graphs and different values on non-regular graphs. Both are
# exported.
theta_dBdB    = deserialize("theta_phi_ba_15_dB_dB_shuffled.dat")     # pf (main text)
theta_dBBd    = deserialize("theta_phi_ba_15_dB_Bd_shuffled.dat")     # pf (main text)
theta_dBdB_ff = deserialize("theta_phi_ba_15_dB_dB_shuffled_ff.dat")  # ff (SI alternative)
theta_dBBd_ff = deserialize("theta_phi_ba_15_dB_Bd_shuffled_ff.dat")  # ff (SI alternative)

function fingerprint_rows(theta_dict, k1, k2, seed1, seed2)
    rows = []
    nmissing = 0
    for i in 1:N_BA, j in 1:N_BA
        key = (Int64(k1), Int64(k2), Int64(i), Int64(j), Int64(seed1), Int64(seed2))
        if haskey(theta_dict, key)
            m = theta_dict[key]   # 4 x 2: col 1 = theta0..3, col 2 = phi00,phi01,phi20,phi21
            push!(rows, Dict(
                "i" => i - 1, "j" => j - 1,                    # 0-indexed
                "theta" => [m[1,1], m[2,1], m[3,1], m[4,1]],   # theta0, theta1, theta2, theta3
                "phi"   => [m[1,2], m[2,2], m[3,2], m[4,2]],   # phi00, phi01, phi20, phi21
            ))
        else
            nmissing += 1
        end
    end
    return rows, nmissing
end

rows_dBdB, miss_dBdB = fingerprint_rows(theta_dBdB, 2, 2, s1, s2)
rows_dBBd, miss_dBBd = fingerprint_rows(theta_dBBd, 2, 2, s1, s2)
rows_dBdB_ff, miss_dBdB_ff = fingerprint_rows(theta_dBdB_ff, 2, 2, s1, s2)
rows_dBBd_ff, miss_dBBd_ff = fingerprint_rows(theta_dBBd_ff, 2, 2, s1, s2)

fingerprint = Dict{String, Any}(
    "pair" => Dict(
        "k1" => 2, "k2" => 2,
        "seed1" => s1, "seed2" => s2,
        "instance_idx1" => 0, "instance_idx2" => 1,
        "shuffle_seed" => string(sseed_pair),
        "perm" => perm_pair .- 1,
    ),
    "shuffled_layer2_edges" => shuffled_edges_0idx,
    "theta_phi_dB_dB"    => rows_dBdB,      # pf (main text)
    "theta_phi_dB_Bd"    => rows_dBBd,      # pf (main text)
    "theta_phi_dB_dB_ff" => rows_dBdB_ff,   # ff (SI alternative goods scheme)
    "theta_phi_dB_Bd_ff" => rows_dBBd_ff,   # ff (SI alternative goods scheme)
    "meta" => Dict(
        "theta_source_dB_dB"    => "theta_phi_ba_15_dB_dB_shuffled.dat (pf, main text)",
        "theta_source_dB_Bd"    => "theta_phi_ba_15_dB_Bd_shuffled.dat (pf, main text)",
        "theta_source_dB_dB_ff" => "theta_phi_ba_15_dB_dB_shuffled_ff.dat (ff, SI alternative)",
        "theta_source_dB_Bd_ff" => "theta_phi_ba_15_dB_Bd_shuffled_ff.dat (ff, SI alternative)",
        "goods_scheme_note" => "pf and ff are two utility/goods schemes; identical (b/c)* on regular graphs, different on non-regular. Main text = pf (plain-M1, matches Allen 2017 + MC). SI = ff (transpose-M1). Revision figures validate pf.",
        "theta_matrix_layout" => "col1 = theta0..theta3, col2 = phi00, phi01, phi20, phi21",
        "indexing" => "i, j, perm, edges all 0-indexed",
        "i_meaning" => "layer-1 initial cooperator node",
        "j_meaning" => "layer-2 initial mutant node (label in the SHUFFLED layer-2 labelling)",
    ),
)

# ---- write graphs_ba_15.json (v2 schema: + perm fields, + provenance meta) ----
out = Dict{String, Any}(
    "graphs"              => graphs_out,
    "shuffle_seeds_dB_dB" => shuffle_dBdB_out,
    "shuffle_seeds_dB_Bd" => shuffle_dBBd_out,
    "meta" => Dict(
        "source_graphs_file"        => "graphs_ba_15.dat",
        "source_shuffle_dBdB_file"  => "shuffle_seeds_ba_15_dB_dB_shuffled.dat",
        "source_shuffle_dBBd_file"  => "shuffle_seeds_ba_15_dB_Bd_shuffled.dat",
        "node_indexing"             => "0-indexed (Python convention)",
        "uint64_handling"           => "shuffle_seed stored as decimal string",
        "perm_field"                => "exact Julia permutation, 0-indexed: new_label = perm[old_label]; derived via shuffle(MersenneTwister(shuffle_seed), collect(1:15)) exactly as rebuild_edge_seq_pair in 'Barabasi Albert-ff.ipynb' cell 34",
        "rule_provenance"           => "FINAL manuscript theta pipeline (theta_phi_ba_15_*_shuffled_ff.dat) used the dB-dB shuffle-seed file for BOTH update rules; the dB-Bd seed table here was used only by the superseded Mar 2025 classification runs",
        "exporter_script"           => "export_ba_perms.jl",
        "generated_at_julia_version" => string(VERSION),
    ),
)

open("graphs_ba_15.json", "w") do io
    JSON.print(io, out, 2)
end
open("ba_theta_fingerprint.json", "w") do io
    JSON.print(io, fingerprint, 2)
end

println("# Wrote graphs_ba_15.json (with exact perms)")
println("#   shuffle_seeds_dB_dB: ", length(shuffle_dBdB_out), " entries")
println("#   shuffle_seeds_dB_Bd: ", length(shuffle_dBBd_out), " entries")
println("# Wrote ba_theta_fingerprint.json")
println("#   ws pair: k=2 insts 0/1, seed1=$s1, seed2=$s2, shuffle_seed=$sseed_pair")
println("#   perm (1-indexed Julia) = ", perm_pair)
println("#   theta rows dB-dB pf: ", length(rows_dBdB), " (missing: $miss_dBdB)")
println("#   theta rows dB-Bd pf: ", length(rows_dBBd), " (missing: $miss_dBBd)")
println("#   theta rows dB-dB ff: ", length(rows_dBdB_ff), " (missing: $miss_dBdB_ff)")
println("#   theta rows dB-Bd ff: ", length(rows_dBBd_ff), " (missing: $miss_dBBd_ff)")
