# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mab-mpc-optimizer.git
cd mab-mpc-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run the Interactive App

```bash
streamlit run app/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## Basic Usage

### 1. Run a Simple Comparison

```python
from src.models.bioreactor import Bioreactor
from src.models.parameters import get_scale_parameters
from src.control.fixed_recipe import FixedRecipeLibrary
from src.control.exponential_feeding import ExponentialRecipeLibrary
from src.control.mpc_controller import SimplifiedMPCController
from src.models.cho_kinetics import CHOCellModel
import numpy as np

# Setup
params = get_scale_parameters("production")  # 2000L
reactor = Bioreactor(params=params)
t_span = (0, 336)  # 14 days
t_eval = np.linspace(0, 336, 500)

# Fixed Recipe
fixed = FixedRecipeLibrary.typical_industry(params.V_max)
results_fixed = reactor.simulate(t_span, fixed, t_eval=t_eval)

# MPC
model = CHOCellModel(params)
mpc = SimplifiedMPCController(model)
results_mpc = reactor.simulate(t_span, mpc, t_eval=t_eval)

# Compare
print(f"Fixed Recipe: {results_fixed.final_titer:.2f} g/L")
print(f"MPC:          {results_mpc.final_titer:.2f} g/L")
print(f"Improvement:  {(results_mpc.final_titer - results_fixed.final_titer)/results_fixed.final_titer * 100:.1f}%")
```

### 2. Visualize Results

```python
from src.visualization.plots import plot_comparison_dashboard

results_dict = {
    'Fixed Recipe': results_fixed,
    'MPC': results_mpc
}

fig = plot_comparison_dashboard(results_dict, save_path='comparison.png')
```

### 3. Economic Analysis

```python
from src.utils.economics import EconomicAnalyzer

analyzer = EconomicAnalyzer(
    mab_value_per_g=20000.0,
    batch_cost=100000.0,
    batches_per_year=20
)

report = analyzer.format_comparison(
    results_fixed,
    results_mpc,
    strategy_names=('Fixed Recipe', 'MPC')
)

print(report)
```

## Next Steps

- **Explore notebooks:** `notebooks/01_model_development.ipynb`
- **Read documentation:** `docs/mpc_theory.md`
- **Try different scenarios:** Modify parameters, add disturbances
- **Tune the MPC:** Adjust weights, horizons, constraints

## Troubleshooting

**Simulation is slow:**
- Reduce prediction horizon in MPC config
- Use coarser time points (`t_eval`)
- Use SimplifiedMPCController instead of full MPCController

**Optimization fails:**
- Check initial conditions (cells > 0, glucose > 0)
- Relax constraints slightly
- Increase max_iterations in MPC config

**Results look wrong:**
- Validate parameters against literature
- Check units (cells/L, mM, mg/L)
- Ensure feed rate is reasonable for scale

## Resources

- **GitHub:** [Repository Link]
- **Blog Series:** [Link to Part 1]
- **LinkedIn:** [Your Profile]
- **Email:** [Your Email]

## Citation

```bibtex
@software{mab_mpc_optimizer,
  author = {Kemal Yaylali},
  title = {Real-Time Fed-Batch Optimizer: MPC for mAb Production},
  year = {2026},
  url = {https://github.com/yourusername/mab-mpc-optimizer}
}
```
