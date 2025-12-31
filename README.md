# Non-dilemmatic-social-dynamics-promote-cooperation-in-multilayer-network
Code associated with Non-dilemmatic social dynamics promote cooperation in multilayer network 
# Non-dilemmatic Social Dynamics Promote Cooperation in Multilayer Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the simulation code and analysis tools accompanying the paper:

> **"Non-dilemmatic social dynamics promote cooperation in multilayer networks"**

---

## 📋 Overview

We study evolutionary dynamics on **multilayer networks** where individuals participate in two coupled processes:

1. **Game Layer**: A social dilemma (donation game) where cooperators pay a cost to benefit others
2. **Constant Selection Layer**: A non-dilemmatic process where types differ in reproductive fitness


### Model Summary

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
└── Figures and Data/            # Generated figures and precomputed data
    ├── heatmap*_dB_dB.npz       # Heatmap data for dB-dB update rule
    ├── heatmap*_dB_Bd.npz       # Heatmap data for dB-Bd update rule
    └── *.png, *.pdf             # Publication figures
```

---

## 🔬 Methods

### Death-Birth Moran Process

We simulate evolutionary dynamics using the **death-birth (dB) update rule**:

1. **Death**: Select a random individual to be replaced
2. **Birth**: A neighbor is chosen proportionally to fitness to reproduce
3. **Inheritance**: The offspring inherits the parent's strategy

The process runs until the population reaches an **absorbing state** (fixation of one strategy on each layer).

### Absorbing States

| State | Layer 1 | Layer 2 | Interpretation |
|-------|---------|---------|----------------|
| AA | All-C | All-M | Cooperation and mutant type both fixate |
| AB | All-C | All-R | Cooperation fixates, resident type fixates |
| BA | All-D | All-M | Defection fixates, mutant type fixates |
| BB | All-D | All-R | Defection and resident type both fixate |

### Update Rule Variants

- **dB-dB**: Death-birth on both layers
- **dB-Bd**: Death-birth on Layer 1, birth-death on Layer 2

---

## Getting Started

### Prerequisites

```bash
# Core dependencies
pip install numpy networkx matplotlib pandas

# For parallel simulations
pip install multiprocessing  # (included in Python standard library)
```

### Running Simulations

#### 1. Fixation Probability Analysis

```python
# Open multilayer_egt.ipynb
# This notebook computes fixation probabilities across parameter space (b/c, r)
```

#### 2. Fixation Time Analysis

```python
# Open Fixation_Time.ipynb
# This notebook estimates mean fixation times via Monte Carlo simulation
```

### Quick Example

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

## 📊 Key Results

### Phase Diagrams

The heatmaps show selection outcomes across the $(b/c, r)$ parameter space:

| Color | Selection Outcome |
|-------|-------------------|
| 🟢 **Dark Green** | Both cooperators AND mutants favored |
| 🟢 **Light Green** | Cooperators favored, mutants neutral/disfavored |
| 🟠 **Light Coral** | Mutants favored, cooperators neutral/disfavored |
| 🔴 **Dark Red** | Neither favored |

### Network Topologies Studied

1. **Ring/Cycle Graph**: Regular structure with degree 2
2. **Heterogeneous Network**: Variable degree distribution
3. **Complete Graph**: All-to-all connectivity
4. **Bipartite Graph**: Two-group structure

---

## 📁 Data Files

### Heatmap Data Format (.npz)

Each `.npz` file contains:
- `x_values`: Selection coefficient for cooperators ($\rho_C - 1/N$)
- `y_values`: Selection coefficient for mutants ($\rho_M - 1/N$)

Positive values indicate the strategy is **favored by selection**.

### Loading Data

```python
import numpy as np

data = np.load("Figures and Data/heatmap1_dB_dB.npz")
coop_selection = data["x_values"]  # Shape: (n_r, n_b)
mutant_selection = data["y_values"]
```

---

## 🛠️ Customization

### Adding New Network Topologies

```python
import networkx as nx

# Create your custom network
G_custom = nx.watts_strogatz_graph(n=100, k=4, p=0.1)

# Use in simulation
results = simulate_multilayer_db(G_custom, G_custom, b=2.0, r=1.1, delta=0.01, num_runs=1000)
```

### Parameter Sweeps

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

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{multilayer_cooperation2025,
  title={Non-dilemmatic social dynamics promote cooperation in multilayer networks},
  author={[Authors]},
  journal={[Journal]},
  year={2025},
  doi={[DOI]}
}
```

---

## 📚 References

1. Moran, P. A. P. (1958). Random processes in genetics. *Mathematical Proceedings of the Cambridge Philosophical Society*, 54(1), 60-71.

2. Nowak, M. A., Tarnita, C. E., & Antal, T. (2010). Evolutionary dynamics in structured populations. *Philosophical Transactions of the Royal Society B*, 365(1537), 19-30.

3. Allen, B., et al. (2017). Evolutionary dynamics on any population structure. *Nature*, 544(7649), 227-230.

4. Gómez-Gardeñes, J., et al. (2012). Evolution of cooperation in multiplex networks. *Scientific Reports*, 2, 620.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or collaborations, please open an issue or contact the authors.

---

<p align="center">
  <i>Understanding cooperation through the lens of multilayer networks</i>
</p>
