# PetriRAM: A Petri Net–based Simulation Framework for RAM Analysis

### Purpose

PetriRAM is a Python framework designed to model and simulate maintainance dynamics systems using Petri nets with a focus on Reliability, Availability, and Maintainability (RAM) analysis.

The tool enables researchers and practitioners to:

- Represent system components and their operational/maintenance logic through Petri nets.
- Perform stochastic simulations with customizable probability distributions (e.g., Weibull, Lognormal).
- Compute RAM indicators (Availability, MDT, MTBF, MTTF, MLDT, etc.) directly from simulations without post-processing.
- Extend the framework with new distributions, or optimization routines.
- This software provides a reproducible and extensible alternative to commercial tools, making it easier to integrate RAM analysis into scientific workflows.

### Installation

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/cdtm15/petriRAM_simulator.git
cd PetriRAM
```
Dependencies include:
- Python ≥ 3.9
- NumPy
- Pandas
- Matplotlib
- SciPy

### Usage

The software allows the user to configure different system scenarios by specifying component-level transition parameters, simulation settings, and Monte Carlo repetitions.

### 1. Define transition parameters per component
Each component (`C1`, `C2`, …) requires dictionaries that specify time-to-completion, failure, and maintenance behaviors. Supported probability distributions include Normal, Weibull, Lognormal, and Gamma.

# Example: two heterogeneous components
```bash
params_per_component_different = {
    "C1": {
        "Complete": {"loc": 28000, "scale": 3600},   # Normal distribution
        "Failure_tran": {"dist": "weibull", "alpha": 1.5, "scale": 50000},
        "Corrective Maintenance Time": {"low": 7200, "high": 14400},   # Uniform
        "Preventive Maintenance Time": {"low": 3600, "high": 7200}
    },
    "C2": {
        "Complete": {"loc": 84000, "scale": 3600},
        "Failure_tran": {"dist": "weibull", "alpha": 4.5, "scale": 50000},
        "Corrective Maintenance Time": {"low": 21600, "high": 28800},
        "Preventive Maintenance Time": {"low": 10800, "high": 14400}
    }
}

```

### 2. Configure simulation scenario
A scenario specifies the simulation horizon, number of components, parameter set, and the number of preventive maintenance cycles.
```bash
from functools import partial
from petriRAM import simulate_multicolor_petri

scenario = partial(
    simulate_multicolor_petri,
    T_MAX=200000,                     # simulation horizon (model time units)
    num_components=2,                 # system size
    params_per_component=params_per_component_different,
    num_pm_cycles=5                   # preventive maintenance cycles
)
```

### 3. Run Monte Carlo experiments
Monte Carlo execution replicates the scenario multiple times to obtain stable RAM indicators.

```bash
from petriRAM import monte_carlo_indicators

desc, long_format, general, comparison = monte_carlo_indicators(
    scenario,
    n_iterations=1000                 # number of replications
)
```

- `desc`: per-component summary across iterations.
- `long_forma`: detailed results by iteration (long table).
- `general`: overall averages across all components.
- `comparison`: wide-format tables for side-by-side analysis.

### 4. Visualize results
The framework provides built-in visualization functions.

```bash
from petriRAM import plot_boxplots_per_indicator, plot_ram_summary

plot_boxplots_per_indicator(long_format)   # saves 'components_comparison_boxplot.pdf'
plot_ram_summary(long_format)              # timeline and RAM indicator plots
```

# Features

- Stochastic events: Supports Weibull, Lognormal, and Gamma distributions.
- Signals and scheduling: Incorporates preventive and corrective maintenance cycles.
- Scalability: Supports multiple components and scenarios.
- Indicators: Direct computation of RAM metrics during simulation.
- Visualization: Timeline and state evolution plots.

# License
This project is licensed under the MIT License – see the LICENSE file for details.
