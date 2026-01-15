# Code for: Non-dilemmatic social dynamics promote cooperation in multilayer networks

Code associated with Non-dilemmatic social dynamics promote cooperation in multilayer network 
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)

This repository contains the simulation code and analysis tools accompanying the following paper:  
[Jnanajyoti Bhaumik, Naoki Masuda.  
Non-dilemmatic social dynamics promote cooperation in multilayer networks.  
Preprint arXiv:2601.00460](https://arxiv.org/abs/2601.00460):

---

## 📋 Overview

We study evolutionary dynamics on multilayer networks where individuals participate in two coupled processes:

1. **Game Layer**: A social dilemma (donation game) where cooperators pay a cost to benefit others
2. **Constant Selection Layer**: A non-dilemmatic process where types differ in reproductive fitness


### Model summary

| Layer | Strategies | Payoff Structure |
|-------|-----------|------------------|
| Layer 1 (Game) | Cooperator (C) vs Defector (D) | Donation game: $u_i = -x_i + b \cdot \frac{\sum_j w_{ij} x_j}{k_i}$ |
| Layer 2 (Constant) | Mutant (M) vs Resident (R) | Constant selection: $u_i = r \cdot x_i + (1-x_i)$ |

**Combined Fitness**: $F_i = 1 + \delta(u_i^{(1)} + u_i^{(2)})$

where $\delta$ is the selection intensity and $b/c$ is the benefit-to-cost ratio.

---

## 🗂️ Repository Structure

```
├── README.md                    # This file
├── multilayer_egt.ipynb         # Main analysis: fixation probabilities & phase diagrams
├── Fixation_Time.ipynb          # Fixation time simulations
└── Random Graphs/            # Random Graphs
    ├── Random_graphs.ipynb       # Code for analysing random graphs
    ├── Figures_Random_graphs.ipynb       # Code for plotting random graphs figures
    └── *.png, *.pdf             # Publication figures
└── Figures and Data/            # Generated figures and precomputed data
    ├── heatmap*_dB_dB.npz       # Heatmap data for dB-dB update rule
    ├── heatmap*_dB_Bd.npz       # Heatmap data for dB-Bd update rule
    └── *.png, *.pdf             # Publication figures
```

---

## 🔬 Methods

### death-Birth Moran process

We simulate evolutionary dynamics using the **death-Birth (dB) update rule**:

1. **Death**: Select a random individual to be replaced
2. **Birth**: A neighbor is chosen proportionally to fitness to reproduce
3. **Copying**: The offspring inherits the parent's strategy

The process runs until the population reaches an **absorbing state** (fixation of one strategy on each layer).

### Absorbing states

| State | Layer 1 | Layer 2 | Interpretation |
|-------|---------|---------|----------------|
| AA | All-C | All-M | Cooperation and the mutant type fixate |
| AB | All-C | All-R | Cooperation and the resident type fixate |
| BA | All-D | All-M | Defection and the mutant type fixate |
| BB | All-D | All-R | Defection and the resident type fixate |

### Update rule variants

- **dB-dB**: death-Birth on both layers
- **dB-Bd**: death-Birth on layer 1, Birth-death on layer 2

---

## Getting Started

### Prerequisites

```bash
# Core dependencies
pip install numpy networkx matplotlib pandas

# For parallel simulations
pip install multiprocessing  # (included in Python standard library)
```

### Running simulations

#### 1. Fixation probability analysis

```python
# Open multilayer_egt.ipynb
# This notebook computes fixation probabilities across parameter space (b/c, r)
```

#### 2. Fixation time analysis

```python
# Open Fixation_Time.ipynb
# This notebook estimates mean fixation times via Monte Carlo simulation
```

### Quick example

```python
import numpy as np
import networkx as nx

# Create a multilayer network (same nodes, different edges)
G1 = nx.complete_graph(20)  # Layer 1: Complete graph
G2 = nx.cycle_graph(20)     # Layer 2: Ring/cycle graph

# Parameters
b = 2.0      # Benefit in donation game (cost c=1)
r = 1.1      # Relative fitness of mutant type
delta = 0.01 # Weak selection intensity
num_runs = 1000

# Run simulation (see notebooks for full implementation)
results = simulate_multilayer_db(G1, G2, b, r, delta, num_runs)
```

---

## 📊 Results


The heatmaps show selection outcomes across the $(b/c, r)$ parameter space:

| Color | Selection Outcome |
|-------|-------------------|
| 🟢 **Dark green** | Both cooperators and mutants favored |
| 🟢 **Light green** | Cooperators favored, mutants neutral/disfavored |
| 🟠 **Light coral** | Mutants favored, cooperators neutral/disfavored |
| 🔴 **Dark red** | Neither favored |

### Networks studied

1. **Ring/cycle graph**: Regular network with degree 2
2. **Heterogeneous network**: Heterogeneous degree distribution
3. **Complete graph**: All-to-all connectivity
4. **Bipartite graph**: Two-group structure

---

## 📁 Data Files

### Heatmap data format (.npz)

Each `.npz` file contains:
- `x_values`: Selection coefficient for cooperators ($\rho_C - 1/N$)
- `y_values`: Selection coefficient for mutants ($\rho_M - 1/N$)

Positive values indicate that the strategy is favored by selection.

### Loading data

```python
import numpy as np

data = np.load("Figures and Data/heatmap1_dB_dB.npz")
coop_selection = data["x_values"]  # Shape: (n_r, n_b)
mutant_selection = data["y_values"]
```

---

## 🛠️ Customization

### Adding new networks

```python
import networkx as nx

# Create your custom network
G_custom = nx.watts_strogatz_graph(n=100, k=4, p=0.1)

# Use in simulation
results = simulate_multilayer_db(G_custom, G_custom, b=2.0, r=1.1, delta=0.01, num_runs=1000)
```

### Parameter sweeps

```python
# Example: Sweep over benefit values
b_values = np.linspace(1.0, 5.0, 50)
r_values = np.linspace(0.5, 2.0, 50)

for b in b_values:
    for r in r_values:
        # Run simulation and store results
        pass
```

---

<!-- ## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{multilayer_cooperation2025,
  title={Non-dilemmatic social dynamics promote cooperation in multilayer networks},
  author={[Authors]},
  journal={[Journal]},
  year={2025},
  doi={[DOI]}
}
``` -->

---

## 📚 References

1. Moran, P. A. P. (1958). Random processes in genetics. *Mathematical Proceedings of the Cambridge Philosophical Society*, 54(1), 60-71.

2. https://github.com/qisu1991/MultilayerPopulations

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

---

## Figure Generation

The following table maps each figure in the manuscript to its corresponding notebook:

| Figure | Notebook | Description |
|--------|----------|-------------|
| Fig. 1 | `Figures and Data/Schematics.ipynb` | Schematic diagrams |
| Fig. 2 | `Figures and Data/Schematics.ipynb` | Schematic diagrams |
| Fig. 3 | `Slope Analysis/Enumeration_N6.ipynb`, `Slope Analysis/Slope_Analysis.ipynb` | Slope analysis results |
| Fig. 4 | `Figures and Data/Figures.ipynb` | Main results |
| Fig. 5 | `Random Graphs/Figures_Random_graphs.ipynb` | Random graph results |
| Fig. S1 | `Slope Analysis/Enumeration_N6.ipynb`, `Slope Analysis/Slope_Analysis.ipynb` | Supplementary slope analysis |
| Fig. S2 | `Figures and Data/Schematics.ipynb` | Supplementary schematics |
| Fig. S3 | `Random Graphs/Figures_Random_graphs.ipynb` | Supplementary random graph results |
| Fig. S4 | `Figures and Data/Figures.ipynb` | Supplementary main results |
| Fig. S5 | `Random Graphs/Figures_Random_graphs.ipynb` | Supplementary random graph results |
| Fig. S6 | `Random Graphs/Figures_Random_graphs.ipynb` | Supplementary random graph results |
| Fig. S8 | `Figures and Data/Fixation-time-plots.ipynb` | Fixation time plots |

## 📧 Contact

For questions or collaborations, please open an issue or contact the authors.

---

<p align="center">
  <i>Understanding cooperation through the lens of multilayer networks</i>
</p>
