# User Guide: Adaptive Multi-Fidelity Simulation-Based Optimization

## Table of Contents

1. [Getting Started](#getting-started)
2. [Aircraft Optimization Tutorial](#aircraft-optimization-tutorial)
3. [Spacecraft Optimization Tutorial](#spacecraft-optimization-tutorial)
4. [Advanced Features](#advanced-features)
5. [Configuration and Customization](#configuration-and-customization)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Getting Started

### Installation and Setup

#### System Requirements
- Python 3.8 or higher
- 8GB RAM (16GB recommended for large problems)
- Multi-core CPU for parallel processing

#### Quick Installation
```bash
# Clone the repository
git clone https://github.com/aerospace-optimization/amf-sbo.git
cd amf-sbo

# Create virtual environment
python -m venv amf_env
source amf_env/bin/activate  # Windows: amf_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.models.aerospace_systems import AircraftOptimizationSystem; print('✓ Installation successful')"
```

#### Project Structure Overview
```
amf-sbo/
├── src/                     # Core framework
│   ├── simulation/         # Multi-fidelity models
│   ├── optimization/       # Algorithms (GA, PSO, BO)
│   ├── models/            # Aerospace systems
│   └── visualization/     # Plotting and dashboards
├── examples/              # Working examples
├── data/                  # Example datasets
└── config.json           # Configuration file
```

### Basic Concepts

#### Multi-Fidelity Approach
The framework uses three computational fidelity levels:

- **Low Fidelity (0.1s)**: Fast analytical models, ±15-20% accuracy
- **Medium Fidelity (2-5s)**: Semi-empirical models, ±8-12% accuracy  
- **High Fidelity (15-25s)**: CFD approximations, ±3-5% accuracy

#### Adaptive Switching
The system automatically selects the appropriate fidelity based on:
- Optimization progress
- Remaining computational budget
- Solution uncertainty
- User-defined strategy

---

## Aircraft Optimization Tutorial

### Tutorial 1: Basic Aircraft Design

Let's optimize a commercial aircraft configuration for maximum lift-to-drag ratio.

#### Step 1: Setup
```python
from src.models.aerospace_systems import AircraftOptimizationSystem
from src.optimization.algorithms import GeneticAlgorithm
from src.visualization.graph_generator import ProfessionalGraphGenerator

# Initialize the aircraft optimization system
aircraft_system = AircraftOptimizationSystem()
print("Aircraft optimization system initialized ✓")
```

#### Step 2: Define Design Parameters
```python
# Define the parameter bounds for aircraft design
parameter_bounds = {
    'wingspan': (30.0, 60.0),        # meters
    'wing_area': (150.0, 350.0),     # square meters
    'aspect_ratio': (7.0, 12.0),     # dimensionless
    'sweep_angle': (15.0, 35.0),     # degrees
    'taper_ratio': (0.3, 0.8),       # dimensionless
    'thickness_ratio': (0.08, 0.15), # dimensionless
    'cruise_altitude': (8000, 12000), # meters
    'cruise_mach': (0.65, 0.85),     # Mach number
    'weight': (40000, 80000)         # kg
}

print(f"Optimizing {len(parameter_bounds)} design parameters")
```

#### Step 3: Define Objective Function
```python
def aircraft_objective(parameters):
    """
    Objective function for aircraft optimization.
    Returns negative L/D ratio (we minimize, so negative for maximization)
    """
    # Evaluate the aircraft design
    result = aircraft_system.evaluate_design(
        parameters, 
        mission_profile='commercial'
    )
    
    # Extract the simulation result
    sim_result = result['simulation_result']
    
    # Return negative L/D ratio (minimize negative = maximize positive)
    return {
        'lift_to_drag_ratio': -sim_result.objectives['lift_to_drag_ratio']
    }

print("Objective function defined ✓")
```

#### Step 4: Setup Optimizer
```python
# Create genetic algorithm optimizer
optimizer = GeneticAlgorithm(
    parameter_bounds=parameter_bounds,
    population_size=50,
    crossover_rate=0.8,
    mutation_rate=0.1,
    tournament_size=3
)

print("Genetic algorithm optimizer configured ✓")
```

#### Step 5: Run Optimization
```python
# Run the optimization
print("Starting optimization...")
result = optimizer.optimize(
    objective_function=aircraft_objective,
    max_evaluations=200,
    convergence_tolerance=1e-6
)

print(f"Optimization completed in {result.total_time:.2f} seconds")
print(f"Best L/D ratio: {-result.best_objectives['lift_to_drag_ratio']:.2f}")
```

#### Step 6: Analyze Results
```python
# Print optimal design parameters
print("\n=== OPTIMAL AIRCRAFT DESIGN ===")
for param, value in result.best_parameters.items():
    print(f"{param}: {value:.2f}")

# Get detailed performance metrics
final_evaluation = aircraft_system.evaluate_design(
    result.best_parameters, 
    'commercial'
)
objectives = final_evaluation['simulation_result'].objectives

print(f"\n=== PERFORMANCE METRICS ===")
print(f"L/D Ratio: {objectives['lift_to_drag_ratio']:.2f}")
print(f"Fuel Efficiency: {objectives['fuel_efficiency']:.2f} kg/km")
print(f"Range: {objectives['range']:.0f} km")
print(f"Payload Capacity: {objectives['payload_capacity']:.0f} kg")
```

#### Step 7: Visualize Results
```python
# Create visualization
graph_generator = ProfessionalGraphGenerator("visualizations/")

# Convergence plot
convergence_plot = graph_generator.create_convergence_plot(
    result.optimization_history,
    "Genetic Algorithm - Aircraft Optimization",
    "aircraft_convergence"
)
print(f"Convergence plot saved: {convergence_plot}")

# Fidelity switching analysis
fidelity_stats = aircraft_system.get_optimization_statistics()
fidelity_plot = graph_generator.create_fidelity_switching_plot(
    fidelity_stats['fidelity_history'],
    "aircraft_fidelity_switching"
)
print(f"Fidelity analysis saved: {fidelity_plot}")
```

### Tutorial 2: Multi-Objective Aircraft Optimization

Optimize for both aerodynamic efficiency and fuel consumption.

#### Step 1: Multi-Objective Function
```python
def multi_objective_aircraft(parameters):
    """
    Multi-objective function returning multiple objectives to optimize
    """
    result = aircraft_system.evaluate_design(parameters, 'commercial')
    sim_result = result['simulation_result']
    
    return {
        'lift_to_drag_ratio': -sim_result.objectives['lift_to_drag_ratio'],  # Maximize
        'fuel_efficiency': sim_result.objectives['fuel_efficiency']  # Minimize
    }
```

#### Step 2: NSGA-II Optimization
```python
from src.optimization.algorithms import NSGA2

# Multi-objective optimizer
optimizer = NSGA2(
    parameter_bounds=parameter_bounds,
    population_size=100,
    crossover_rate=0.9,
    mutation_rate=0.1
)

# Run multi-objective optimization
pareto_result = optimizer.optimize(
    objective_function=multi_objective_aircraft,
    max_evaluations=500
)

print(f"Found {len(pareto_result.pareto_solutions)} Pareto optimal solutions")
```

#### Step 3: Pareto Front Analysis
```python
# Visualize Pareto front
pareto_plot = graph_generator.create_pareto_front_plot(
    pareto_result.pareto_solutions,
    "aircraft_pareto_front"
)

# Print some Pareto solutions
print("\n=== PARETO OPTIMAL SOLUTIONS (top 5) ===")
for i, solution in enumerate(pareto_result.pareto_solutions[:5]):
    ld_ratio = -solution['objectives']['lift_to_drag_ratio']
    fuel_eff = solution['objectives']['fuel_efficiency']
    print(f"Solution {i+1}: L/D={ld_ratio:.2f}, Fuel={fuel_eff:.2f} kg/km")
```

### Tutorial 3: Robust Aircraft Design

Design aircraft that performs well under uncertainty.

#### Step 1: Setup Uncertainty Sources
```python
from src.optimization.robust_optimization import (
    UncertaintyQuantification, 
    UncertaintyDistribution,
    RobustOptimizer
)

# Define uncertainties
uq = UncertaintyQuantification()

# Manufacturing tolerances
uq.add_parameter_uncertainty('wingspan', UncertaintyDistribution(
    'normal', {'mean': 0.0, 'std': 0.5}, bounds=(-2.0, 2.0)
))
uq.add_parameter_uncertainty('wing_area', UncertaintyDistribution(
    'normal', {'mean': 0.0, 'std': 5.0}, bounds=(-20.0, 20.0)
))

# Environmental uncertainties
uq.add_environmental_uncertainty('wind_speed', UncertaintyDistribution(
    'uniform', {'low': -10.0, 'high': 10.0}
))
uq.add_environmental_uncertainty('temperature_deviation', UncertaintyDistribution(
    'normal', {'mean': 0.0, 'std': 5.0}, bounds=(-15.0, 15.0)
))

print("Uncertainty sources defined ✓")
```

#### Step 2: Robust Optimization
```python
# Create robust optimizer
robust_optimizer = RobustOptimizer(uq)

# Run robust optimization
robust_result = robust_optimizer.robust_optimization(
    objective_function=aircraft_objective,
    parameter_bounds=parameter_bounds,
    robustness_measure='mean_std',  # μ - k*σ approach
    n_mc_samples=100,
    optimization_algorithm='genetic_algorithm'
)

print("Robust optimization completed ✓")
print(f"Robust L/D ratio: {-robust_result.robust_objective:.2f}")
print(f"Standard deviation: {robust_result.objective_std:.3f}")
```

#### Step 3: Uncertainty Analysis
```python
# Generate uncertainty propagation analysis
uncertainty_plot = graph_generator.create_uncertainty_propagation_plot(
    robust_result.monte_carlo_results,
    "aircraft_uncertainty_analysis"
)

# Print robustness metrics
print(f"\n=== ROBUSTNESS ANALYSIS ===")
print(f"Mean performance: {-robust_result.mean_objective:.2f}")
print(f"Std deviation: {robust_result.objective_std:.3f}")
print(f"Worst-case performance: {-robust_result.worst_case:.2f}")
print(f"95% confidence interval: [{-robust_result.confidence_intervals[0.95][1]:.2f}, {-robust_result.confidence_intervals[0.95][0]:.2f}]")
```

---

## Spacecraft Optimization Tutorial

### Tutorial 4: Earth Observation Satellite Design

Optimize a satellite for Earth observation missions.

#### Step 1: Setup Spacecraft System
```python
from src.models.aerospace_systems import SpacecraftOptimizationSystem

# Initialize spacecraft system
spacecraft_system = SpacecraftOptimizationSystem()

# Define spacecraft parameters
spacecraft_bounds = {
    'dry_mass': (1000, 8000),         # kg
    'fuel_mass': (5000, 40000),       # kg
    'specific_impulse': (250, 400),   # seconds
    'thrust': (1000, 20000),          # Newtons
    'solar_panel_area': (20.0, 100.0), # m²
    'thermal_mass': (500, 3000),      # kg
    'target_orbit_altitude': (400, 1000), # km
    'mission_duration': (365, 2190)   # days (1-6 years)
}

print("Spacecraft optimization system initialized ✓")
```

#### Step 2: Define Mission Objective
```python
def spacecraft_objective(parameters):
    """
    Optimize spacecraft for Earth observation mission
    Maximize mission success probability and minimize mass
    """
    result = spacecraft_system.evaluate_design(
        parameters, 
        mission_type='earth_observation'
    )
    
    sim_result = result['simulation_result']
    
    # Multi-objective: maximize success probability, minimize total mass
    total_mass = parameters['dry_mass'] + parameters['fuel_mass']
    success_prob = sim_result.objectives['mission_success_probability']
    
    return {
        'total_mass': total_mass,
        'mission_success': -success_prob  # Negative because we minimize
    }
```

#### Step 3: Run Spacecraft Optimization
```python
from src.optimization.algorithms import BayesianOptimization

# Use Bayesian optimization for expensive spacecraft evaluations
optimizer = BayesianOptimization(
    parameter_bounds=spacecraft_bounds,
    acquisition_function='ei',  # Expected improvement
    xi=0.01,                   # Exploration parameter
    kappa=2.576               # UCB parameter
)

# Run optimization
spacecraft_result = optimizer.optimize(
    objective_function=spacecraft_objective,
    max_evaluations=100
)

print("Spacecraft optimization completed ✓")
```

#### Step 4: Analyze Spacecraft Design
```python
# Get optimal spacecraft configuration
optimal_params = spacecraft_result.best_parameters
final_eval = spacecraft_system.evaluate_design(
    optimal_params, 
    'earth_observation'
)

print(f"\n=== OPTIMAL SPACECRAFT DESIGN ===")
print(f"Dry Mass: {optimal_params['dry_mass']:.0f} kg")
print(f"Fuel Mass: {optimal_params['fuel_mass']:.0f} kg")
print(f"Total Mass: {optimal_params['dry_mass'] + optimal_params['fuel_mass']:.0f} kg")
print(f"Solar Panel Area: {optimal_params['solar_panel_area']:.1f} m²")
print(f"Target Altitude: {optimal_params['target_orbit_altitude']:.0f} km")
print(f"Mission Duration: {optimal_params['mission_duration']:.0f} days")

# Performance metrics
objectives = final_eval['simulation_result'].objectives
print(f"\n=== MISSION PERFORMANCE ===")
print(f"Delta-V Capability: {objectives['delta_v_capability']:.0f} m/s")
print(f"Power Efficiency: {objectives['power_efficiency']:.3f}")
print(f"Thermal Stability: {objectives['thermal_stability']:.3f}")
print(f"Mission Success Probability: {objectives['mission_success_probability']:.3f}")
```

### Tutorial 5: Interplanetary Mission Design

Design a spacecraft for deep space exploration.

#### Step 1: Deep Space Mission Setup
```python
# Deep space mission parameters
deep_space_bounds = {
    'dry_mass': (2000, 15000),
    'fuel_mass': (20000, 100000),
    'specific_impulse': (300, 500),    # Higher Isp for deep space
    'thrust': (500, 5000),             # Lower thrust, higher efficiency
    'solar_panel_area': (50.0, 200.0), # Larger panels for distant operations
    'thermal_mass': (1000, 5000),
    'target_orbit_altitude': (0, 0),   # N/A for interplanetary
    'mission_duration': (1825, 7300)   # 5-20 years
}

def deep_space_objective(parameters):
    """Deep space mission optimization"""
    result = spacecraft_system.evaluate_design(
        parameters, 
        mission_type='deep_space'
    )
    
    sim_result = result['simulation_result']
    
    # Prioritize delta-v capability and power efficiency for deep space
    return {
        'delta_v': -sim_result.objectives['delta_v_capability'],  # Maximize
        'power_efficiency': -sim_result.objectives['power_efficiency'],  # Maximize
        'total_mass': parameters['dry_mass'] + parameters['fuel_mass']  # Minimize
    }
```

#### Step 2: Multi-Objective Deep Space Optimization
```python
# Use NSGA-II for multi-objective deep space mission
deep_space_optimizer = NSGA2(
    parameter_bounds=deep_space_bounds,
    population_size=80
)

deep_space_result = deep_space_optimizer.optimize(
    objective_function=deep_space_objective,
    max_evaluations=400
)

# Analyze Pareto solutions for different mission priorities
print(f"Found {len(deep_space_result.pareto_solutions)} deep space configurations")

# Find solution with highest delta-v
best_deltav_solution = max(
    deep_space_result.pareto_solutions,
    key=lambda x: -x['objectives']['delta_v']
)

print(f"Highest delta-v configuration: {-best_deltav_solution['objectives']['delta_v']:.0f} m/s")
```

---

## Advanced Features

### Sensitivity Analysis

#### Global Sensitivity Analysis
```python
from src.optimization.robust_optimization import SensitivityAnalysis

# Create sensitivity analyzer
sensitivity = SensitivityAnalysis()

# Morris screening (fast, qualitative)
morris_results = sensitivity.morris_screening(
    objective_function=aircraft_objective,
    parameter_bounds=parameter_bounds,
    n_trajectories=20,
    n_levels=4
)

print("=== MORRIS SENSITIVITY INDICES ===")
for param, indices in morris_results.items():
    mu_star = indices['mu_star']
    sigma = indices['sigma']
    print(f"{param}: μ*={mu_star:.3f}, σ={sigma:.3f}")
```

#### Sobol Sensitivity Indices
```python
# Sobol indices (quantitative, more expensive)
sobol_results = sensitivity.sobol_indices(
    objective_function=aircraft_objective,
    parameter_bounds=parameter_bounds,
    n_samples=2000
)

print("\n=== SOBOL SENSITIVITY INDICES ===")
for param, indices in sobol_results.items():
    s1 = indices['S1']     # First-order index
    st = indices['ST']     # Total-order index
    print(f"{param}: S1={s1:.3f}, ST={st:.3f}")
```

### Data Management and Export

#### Save and Load Optimization Runs
```python
from src.utilities.data_manager import DataManager

# Initialize data manager
data_manager = DataManager("optimization_results")

# Save optimization run
run_id = data_manager.save_optimization_run(
    run_id="aircraft_ga_commercial_001",
    optimization_result=result,
    algorithm_name="GeneticAlgorithm",
    system_type="aircraft",
    parameters={
        "mission_profile": "commercial",
        "population_size": 50,
        "max_evaluations": 200
    }
)

print(f"Optimization run saved with ID: {run_id}")

# Load previous run
loaded_run = data_manager.load_optimization_run(run_id)
if loaded_run:
    print(f"Loaded run: {loaded_run['metadata']['timestamp']}")
```

#### Export Results
```python
# Export to multiple formats
csv_file = data_manager.export_to_csv([run_id], "aircraft_results.csv")
excel_file = data_manager.export_to_excel([run_id], "aircraft_results.xlsx")
hdf5_file = data_manager.export_to_hdf5([run_id], "aircraft_results.h5")

print(f"Results exported to:")
print(f"  CSV: {csv_file}")
print(f"  Excel: {excel_file}")
print(f"  HDF5: {hdf5_file}")

# Create comparison report
report = data_manager.create_comparison_report(
    [run_id], 
    "Aircraft_Optimization_Report"
)
print(f"Comparison report: {report}")
```

### Advanced Visualization

#### Interactive Dashboard
```python
# Create interactive dashboard
dashboard_file = graph_generator.create_interactive_dashboard(
    optimization_data={
        'optimization_history': result.optimization_history,
        'fidelity_switching': fidelity_stats['fidelity_history'],
        'parameter_evolution': result.parameter_evolution
    },
    save_filename="aircraft_dashboard"
)

print(f"Interactive dashboard created: {dashboard_file}")
```

#### 3D Design Space Exploration
```python
# 3D visualization of design space
design_space_plot = graph_generator.create_3d_design_space_plot(
    optimization_history=result.optimization_history,
    param_names=['wingspan', 'wing_area', 'aspect_ratio'],
    save_filename="aircraft_design_space_3d"
)

print(f"3D design space plot: {design_space_plot}")
```

---

## Configuration and Customization

### Configuration File Setup

#### Basic Configuration
Edit `config.json` to customize framework behavior:

```json
{
    "aircraft": {
        "parameter_bounds": {
            "wingspan": [25.0, 80.0],
            "wing_area": [100.0, 500.0]
        },
        "mission_profiles": {
            "commercial": {
                "max_range_km": 8000,
                "cruise_altitude_m": 11000,
                "payload_kg": 15000
            }
        }
    },
    "optimization": {
        "default_parameters": {
            "genetic_algorithm": {
                "population_size": 50,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1
            }
        },
        "convergence_tolerance": 1e-6,
        "max_evaluations": 1000
    },
    "visualization": {
        "style": "aerospace",
        "color_scheme": "professional",
        "save_format": ["png", "svg", "pdf"]
    }
}
```

#### Loading Custom Configuration
```python
import json

# Load custom configuration
with open('my_custom_config.json', 'r') as f:
    config = json.load(f)

# Apply to aircraft system
aircraft_system = AircraftOptimizationSystem()
aircraft_system.configure(config['aircraft'])

# Apply to optimizer
optimizer = GeneticAlgorithm(
    parameter_bounds,
    **config['optimization']['default_parameters']['genetic_algorithm']
)
```

### Custom Fidelity Strategies

#### Define Custom Strategy
```python
from src.simulation.adaptive_fidelity import FidelitySwitchingStrategy

class CustomFidelityStrategy:
    def select_fidelity(self, parameters, optimization_progress=None, **kwargs):
        """
        Custom fidelity selection logic
        """
        if optimization_progress is None:
            return 1  # Start with low fidelity
        
        evaluations = optimization_progress.get('evaluations', 0)
        improvement_rate = optimization_progress.get('improvement_rate', 1.0)
        
        # Use high fidelity if improvement is slowing and we have budget
        if improvement_rate < 0.01 and evaluations < 150:
            return 3
        elif improvement_rate < 0.05:
            return 2
        else:
            return 1

# Use custom strategy
aircraft_system = AircraftOptimizationSystem(
    fidelity_strategy=CustomFidelityStrategy()
)
```

### Custom Objective Functions

#### Multi-Disciplinary Optimization
```python
def mdo_aircraft_objective(parameters):
    """
    Multi-disciplinary optimization including aerodynamics,
    structures, propulsion, and economics
    """
    # Aerodynamic analysis
    aero_result = aircraft_system.evaluate_design(parameters, 'commercial')
    
    # Add custom structural analysis
    wing_loading = parameters['weight'] / parameters['wing_area']
    structural_margin = 2.5 - (wing_loading / 5000)  # Simple structural model
    
    # Add economic analysis
    manufacturing_cost = (
        parameters['wingspan'] * 50000 +  # Wing cost
        parameters['weight'] * 100 +      # Weight penalty
        1000000                           # Base cost
    )
    
    # Combined objective
    aero_objectives = aero_result['simulation_result'].objectives
    
    return {
        'lift_to_drag_ratio': -aero_objectives['lift_to_drag_ratio'],
        'structural_margin': -structural_margin,
        'manufacturing_cost': manufacturing_cost / 1e6  # Normalize to millions
    }
```

---

## Troubleshooting

### Common Issues and Solutions

#### Installation Problems
**Issue**: Import errors or missing dependencies
```python
# Check Python version
import sys
print(f"Python version: {sys.version}")

# Verify key packages
try:
    import numpy, scipy, matplotlib
    print("✓ Core packages available")
except ImportError as e:
    print(f"✗ Missing package: {e}")
```

**Solution**: Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```

#### Optimization Not Converging
**Issue**: Optimizer runs but doesn't improve

**Diagnostics**:
```python
# Check convergence history
import matplotlib.pyplot as plt

history = result.optimization_history
objectives = [h['best_objective'] for h in history]

plt.plot(objectives)
plt.xlabel('Generation')
plt.ylabel('Objective Value')
plt.title('Convergence History')
plt.show()

# Check if stuck in local optimum
if len(set(objectives[-10:])) == 1:
    print("⚠ Optimization may be stuck in local optimum")
```

**Solutions**:
1. Increase population size or mutation rate
2. Try different optimization algorithm
3. Check parameter bounds
4. Use multi-start optimization

#### Memory Issues
**Issue**: Out of memory errors during large optimizations

**Solution**: Enable memory-efficient mode
```python
# Configure for memory efficiency
aircraft_system = AircraftOptimizationSystem()
aircraft_system.configure({
    'cache_size': 100,        # Limit evaluation cache
    'parallel_workers': 2,    # Reduce parallel processes
    'save_history': False     # Disable detailed history
})
```

#### Slow Performance
**Issue**: Optimization runs very slowly

**Diagnostics**:
```python
import time

# Time a single evaluation
start_time = time.time()
result = aircraft_objective(parameter_bounds)
eval_time = time.time() - start_time

print(f"Single evaluation time: {eval_time:.3f} seconds")

# Check fidelity distribution
fidelity_stats = aircraft_system.get_optimization_statistics()
print(f"Fidelity distribution: {fidelity_stats['fidelity_counts']}")
```

**Solutions**:
1. Adjust fidelity switching strategy
2. Enable parallel processing
3. Use surrogate models for expensive evaluations

### Debug Mode

#### Enable Detailed Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_debug.log'),
        logging.StreamHandler()
    ]
)

# Run optimization with debug info
logger = logging.getLogger('amf_sbo')
logger.info("Starting debug optimization run")

result = optimizer.optimize(
    objective_function=aircraft_objective,
    max_evaluations=50,
    verbose=True
)
```

#### Validation Checks
```python
def validate_parameters(parameters, bounds):
    """Check if parameters are within bounds"""
    for param, value in parameters.items():
        lower, upper = bounds[param]
        if not (lower <= value <= upper):
            print(f"⚠ Parameter {param}={value} outside bounds [{lower}, {upper}]")
            return False
    return True

# Test with sample parameters
test_params = {param: (bounds[0] + bounds[1]) / 2 
               for param, bounds in parameter_bounds.items()}

if validate_parameters(test_params, parameter_bounds):
    print("✓ Parameter validation passed")
    
    # Test objective function
    try:
        test_result = aircraft_objective(test_params)
        print("✓ Objective function test passed")
    except Exception as e:
        print(f"✗ Objective function error: {e}")
```

---

## Best Practices

### Optimization Strategy

#### Parameter Bounds Selection
- **Start wide, then narrow**: Begin with generous bounds, then refine based on initial results
- **Physical constraints**: Ensure bounds reflect realistic design constraints
- **Normalization**: Consider normalizing parameters to similar ranges

```python
# Example of well-chosen bounds
parameter_bounds = {
    # Generous but realistic bounds
    'wingspan': (25.0, 70.0),      # Covers regional to wide-body aircraft
    'wing_area': (120.0, 450.0),   # Appropriate for wingspan range
    'aspect_ratio': (6.0, 14.0),   # From low-speed to high-efficiency designs
    
    # Tighter bounds based on mission requirements
    'cruise_mach': (0.70, 0.85),   # Commercial cruise range
    'cruise_altitude': (9000, 12500) # Typical cruise altitudes
}
```

#### Algorithm Selection Guidelines

| Problem Characteristics | Recommended Algorithm | Settings |
|-------------------------|----------------------|-----------|
| Continuous, smooth | Bayesian Optimization | EI acquisition, 50-100 evaluations |
| Discrete/mixed variables | Genetic Algorithm | Pop size 50-100, high mutation |
| Multi-objective | NSGA-II | Pop size 100-200, 500+ evaluations |
| Expensive evaluations | Bayesian Optimization | Conservative acquisition |
| Many local optima | Particle Swarm | Large swarm, high exploration |

### Performance Optimization

#### Parallel Processing
```python
# Enable parallel evaluation
import multiprocessing

n_cores = multiprocessing.cpu_count()
print(f"Available CPU cores: {n_cores}")

# Configure for parallel processing
optimizer = GeneticAlgorithm(
    parameter_bounds,
    population_size=n_cores * 4,  # 4x cores for good load balancing
    parallel_evaluations=True,
    n_workers=n_cores - 1         # Leave one core free
)
```

#### Efficient Fidelity Strategy
```python
# Configure balanced fidelity strategy
aircraft_system = AircraftOptimizationSystem(
    fidelity_strategy=FidelitySwitchingStrategy.BALANCED
)

# Monitor fidelity usage
def optimization_callback(generation, best_fitness, population):
    if generation % 10 == 0:
        stats = aircraft_system.get_optimization_statistics()
        fidelity_counts = stats['fidelity_counts']
        print(f"Gen {generation}: Fidelity usage {fidelity_counts}")
```

### Result Analysis

#### Statistical Significance
```python
# Run multiple optimization runs for statistical analysis
n_runs = 10
results = []

for run in range(n_runs):
    print(f"Run {run+1}/{n_runs}")
    result = optimizer.optimize(aircraft_objective, max_evaluations=200)
    results.append(-result.best_objectives['lift_to_drag_ratio'])

# Statistical analysis
import numpy as np
mean_performance = np.mean(results)
std_performance = np.std(results)
min_performance = np.min(results)
max_performance = np.max(results)

print(f"\n=== STATISTICAL ANALYSIS ({n_runs} runs) ===")
print(f"Mean L/D: {mean_performance:.2f} ± {std_performance:.2f}")
print(f"Best L/D: {max_performance:.2f}")
print(f"Worst L/D: {min_performance:.2f}")
print(f"Success rate: {np.sum(np.array(results) > 18.0)/n_runs*100:.1f}%")
```

#### Convergence Analysis
```python
# Analyze convergence characteristics
def analyze_convergence(optimization_history):
    objectives = [h['best_objective'] for h in optimization_history]
    
    # Find convergence point (when improvement < 1% for 10 generations)
    convergence_gen = None
    for i in range(10, len(objectives)):
        recent_improvement = (objectives[i-10] - objectives[i]) / abs(objectives[i-10])
        if recent_improvement < 0.01:
            convergence_gen = i
            break
    
    if convergence_gen:
        print(f"Converged at generation {convergence_gen}")
        print(f"Could have stopped early, saving {len(objectives) - convergence_gen} evaluations")
    else:
        print("No clear convergence detected - consider more evaluations")

analyze_convergence(result.optimization_history)
```

### Documentation and Reproducibility

#### Save Complete Run Information
```python
import json
from datetime import datetime

# Create comprehensive run record
run_record = {
    'timestamp': datetime.now().isoformat(),
    'algorithm': 'GeneticAlgorithm',
    'parameters': {
        'population_size': 50,
        'max_evaluations': 200,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1
    },
    'problem_setup': {
        'parameter_bounds': parameter_bounds,
        'mission_profile': 'commercial',
        'fidelity_strategy': 'balanced'
    },
    'results': {
        'best_parameters': result.best_parameters,
        'best_objectives': result.best_objectives,
        'total_evaluations': result.total_evaluations,
        'convergence_achieved': result.convergence_achieved,
        'total_time': result.total_time
    },
    'computational_stats': aircraft_system.get_optimization_statistics()
}

# Save to file
with open(f'optimization_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
    json.dump(run_record, f, indent=2)

print("Complete run record saved ✓")
```

This completes the comprehensive user guide with practical examples and best practices for using the adaptive multi-fidelity optimization framework.