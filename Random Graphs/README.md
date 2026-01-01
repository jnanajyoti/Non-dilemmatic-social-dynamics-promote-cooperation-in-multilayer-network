
---

## 🎲 Random Graph Analysis

This section covers analysis on random graph ensembles: **Erdős-Rényi (ER)** and **Barabási-Albert (BA)** networks.


### Configuration

The notebook supports four analysis modes via configuration flags:

```julia
GRAPH_TYPE = :ER   # :ER (Erdős-Rényi) or :BA (Barabási-Albert)
UPDATE_RULE = :dB  # :dB (death-Birth) or :ff (fitness-proportional)
```

| Mode | Graph Model | Update Rule | Parameter Sweep |
|------|-------------|-------------|-----------------|
| ER-dB | Erdős-Rényi | Death-Birth | p ∈ [0.1, 0.5] |
| ER-ff | Erdős-Rényi | Fitness-proportional | p ∈ [0.1, 0.5] |
| BA-dB | Barabási-Albert | Death-Birth | k ∈ [1, 5] |
| BA-ff | Barabási-Albert | Fitness-proportional | k ∈ [1, 5] |

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `N` | Network size | 15 |
| `ER_PROBS` | ER connection probabilities | [0.1, 0.2, 0.3, 0.4, 0.5] |
| `BA_K_VALUES` | BA edges per new node | [1, 2, 3, 4, 5] |
| `NUM_SEEDS` | Graph instances per parameter | 5 |

### Output Format

Results are stored as serialized Julia dictionaries:

```julia
# Key format
(param1, param2, coop_node_i, coop_node_j, seed1, seed2) => θφ_matrix

# Load and access
using Serialization
data = deserialize("theta_er_dB_15.dat")
θφ = data[(0.2, 0.3, 1, 5, seed1, seed2)]  # 4×2 matrix
```

### Classification

Each configuration is classified by whether cooperation is favored:

| Region | Condition | Result |
|--------|-----------|--------|
| R1 | θ₁ − θ₃ > 0 | ✓ Favored |
| R2 | θ₁ − θ₃ < 0, φ₂₀ ≥ 0 | ✓ Favored |
| R3 | θ₁ − θ₃ < 0, φ₂₀ < 0, expr > 0 | ✓ Favored |
| R4 | θ₁ ≈ θ₃ (boundary) | Depends |
| F | Various | ✗ Not favored |

### Quick Start

```julia
# 1. Set configuration
GRAPH_TYPE = :ER
UPDATE_RULE = :dB

# 2. Run all cells

# 3. View summary
analyze_results(results)
```

---
