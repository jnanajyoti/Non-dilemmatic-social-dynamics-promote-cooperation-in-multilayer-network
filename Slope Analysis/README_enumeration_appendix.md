
---

## 📊 Complete Enumeration (N=6)

This section covers exhaustive enumeration of all non-isomorphic two-layer multiplex configurations on N=6 nodes.

### Additional Files

```
├── Enumeration_N6.ipynb    # Julia: Generate all unique configurations
├── Slope_Analysis.ipynb    # Python: Statistical analysis & visualization
└── Data files (generated):
    ├── N6_enumeration.csv      # Raw enumeration results
    ├── N6_enumeration.parquet  # Compressed format (faster loads)
    ├── sign_classification.csv # Sign analysis results
    └── theta_aggregation.csv   # Aggregated statistics
```

### Workflow

```
┌─────────────────────┐     ┌─────────────────────┐
│  Enumeration_N6.ipynb │ ──► │  N6_enumeration.csv  │
│       (Julia)        │     │    (~2.7M rows)      │
└─────────────────────┘     └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │ Slope_Analysis.ipynb │
                            │      (Python)        │
                            └──────────┬──────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
            ┌───────────┐      ┌───────────┐      ┌───────────┐
            │ Heatmaps  │      │ Bar plots │      │  Tables   │
            └───────────┘      └───────────┘      └───────────┘
```

### Enumeration (Julia)

**Prerequisites**:
- Julia 1.11+
- nauty (`brew install nauty` or `apt install nauty`)
- Packages: `Graphs`, `IterativeSolvers`, `SparseArrays`, `Combinatorics`

**What it computes**:
1. All 112 non-isomorphic connected graphs on 6 vertices
2. Rooted configurations (graph + mutant position)
3. All unique two-layer multiplex pairs with layer alignments
4. θ,φ coefficients for each configuration

**Output format** (CSV):
| Column | Description |
|--------|-------------|
| `canon_id` | Canonical identifier (unique per configuration) |
| `theta1` | θ₁: Focal-recipient correlation |
| `theta2` | θ₂: Focal-payoff correlation |
| `theta3` | θ₃: Donor-recipient correlation |
| `phi01` | φ₀₁: Inter-layer correlation |
| `phi20` | φ₂₀: Inter-layer payoff correlation |
| `phi21` | φ₂₁: Inter-layer recipient correlation |

### Analysis (Python)

**Prerequisites**:
```bash
pip install polars duckdb matplotlib numpy
```

**Key analyses**:

1. **Sign Classification**: Distribution by sign(θ₁−θ₃) × sign(φ₀₁−φ₂₁)
   - d = θ₁ − θ₃ > 0 → Cooperation directly favored
   - d < 0 & q > 0 → Cooperation conditionally favored

2. **Selection Gradients**:
   - ∂r*/∂b = φ₂₀/θ₂ (sensitivity to benefit)
   - ∂r*/∂c = (φ₀₁−φ₂₁)/θ₂ (sensitivity to cost)

3. **Visualizations**:
   - 2D histogram of selection gradients
   - Sign classification bar plot
   - θ vs φ scatter plots

### Quick Start

```julia
# 1. Run enumeration (Julia) - takes several hours
include("Enumeration_N6.ipynb")

# Output: N6_enumeration.csv
```

```python
# 2. Run analysis (Python)
DATA_FILE = "N6_enumeration.csv"
# Execute all cells in Slope_Analysis.ipynb
```

### Mathematical Background

The selection condition for cooperation under weak selection:

```
cθ₂ + bθ₁ − bθ₃ + (r−1)(−φ₂₀) > 0
```

Rearranging for critical r*:

```
r* = 1 + (cθ₂ + b(θ₁−θ₃)) / φ₂₀
```

The partial derivatives measure how r* changes with game parameters:
- ∂r*/∂b = (θ₁−θ₃)/φ₂₀ (but we use φ₂₀/θ₂ for normalized comparison)
- ∂r*/∂c = θ₂/φ₂₀ (sensitivity to cost)

### Expected Results

For N=6 enumeration (~2.7 million configurations):

| Sign Pattern | Interpretation | Typical % |
|--------------|----------------|-----------|
| θ₁−θ₃ > 0 | Cooperation favored | ~17% |
| θ₁−θ₃ < 0, φ₀₁−φ₂₁ > 0 | Conditional | ~35% |
| θ₁−θ₃ < 0, φ₀₁−φ₂₁ < 0 | Not favored | ~46% |
| θ₁−θ₃ = 0 | Boundary | ~2% |

---
