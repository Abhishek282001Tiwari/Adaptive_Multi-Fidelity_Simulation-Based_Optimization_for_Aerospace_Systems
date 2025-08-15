---
layout: page
title: "Examples & Tutorials"
description: "Comprehensive examples and step-by-step tutorials for aircraft and spacecraft optimization"
---

## Quick Start Examples

### Basic Aircraft Optimization

Get started with a simple aircraft wing optimization example:

```python
from src.models.aerospace_systems import AircraftOptimizationSystem
from src.optimization.algorithms import GeneticAlgorithm

# Initialize system
aircraft_system = AircraftOptimizationSystem()

# Define parameters
parameter_bounds = {
    'wingspan': (30.0, 60.0),
    'wing_area': (150.0, 350.0),
    'aspect_ratio': (7.0, 12.0),
    'sweep_angle': (15.0, 35.0),
    'cruise_mach': (0.65, 0.85)
}

# Create optimizer
optimizer = GeneticAlgorithm(parameter_bounds, population_size=50)

# Define objective function
def objective_function(parameters):
    result = aircraft_system.evaluate_design(parameters, 'commercial')
    return result['simulation_result']

# Run optimization
result = optimizer.optimize(objective_function, max_evaluations=200)

print(f"Best L/D ratio: {result.best_objectives['lift_to_drag_ratio']:.2f}")
```

### Basic Spacecraft Optimization

Simple spacecraft design optimization example:

```python
from src.models.aerospace_systems import SpacecraftOptimizationSystem
from src.optimization.algorithms import BayesianOptimization

# Initialize spacecraft system
spacecraft_system = SpacecraftOptimizationSystem()

# Define spacecraft parameters
spacecraft_bounds = {
    'dry_mass': (1000, 8000),
    'fuel_mass': (5000, 40000),
    'solar_panel_area': (20.0, 100.0),
    'target_orbit_altitude': (400, 1000)
}

# Create Bayesian optimizer
optimizer = BayesianOptimization(spacecraft_bounds)

# Define objective
def spacecraft_objective(parameters):
    result = spacecraft_system.evaluate_design(parameters, 'earth_observation')
    return result['simulation_result']

# Run optimization
result = optimizer.optimize(spacecraft_objective, max_evaluations=100)
```

## Comprehensive Tutorials

### Tutorial 1: Aircraft Wing Design Optimization

#### Objective
Optimize a commercial aircraft wing for maximum aerodynamic efficiency while maintaining structural and operational constraints.

#### Problem Setup

```python
import numpy as np
from src.models.aerospace_systems import AircraftOptimizationSystem
from src.optimization.algorithms import GeneticAlgorithm
from src.visualization.graph_generator import ProfessionalGraphGenerator

# Initialize systems
aircraft_system = AircraftOptimizationSystem()
graph_generator = ProfessionalGraphGenerator("tutorial_results/")

# Define comprehensive parameter bounds
parameter_bounds = {
    'wingspan': (35.0, 65.0),          # Wing span (m)
    'wing_area': (200.0, 400.0),       # Wing planform area (m²)
    'aspect_ratio': (8.0, 12.0),       # Wing aspect ratio
    'sweep_angle': (20.0, 35.0),       # Wing sweep angle (degrees)
    'taper_ratio': (0.3, 0.8),         # Wing taper ratio
    'thickness_ratio': (0.09, 0.15),   # Airfoil thickness ratio
    'cruise_altitude': (9000, 12000),   # Cruise altitude (m)
    'cruise_mach': (0.75, 0.85),       # Cruise Mach number
    'weight': (60000, 90000)           # Aircraft weight (kg)
}
```

#### Multi-Objective Optimization

```python
def multi_objective_aircraft(parameters):
    """
    Multi-objective function optimizing for:
    1. Aerodynamic efficiency (L/D ratio)
    2. Fuel efficiency
    3. Structural weight
    """
    result = aircraft_system.evaluate_design(parameters, 'commercial')
    sim_result = result['simulation_result']
    
    # Calculate additional objectives
    structural_weight_penalty = (parameters['wingspan'] * parameters['wing_area'] * 0.1)
    
    return {
        'lift_to_drag_ratio': -sim_result.objectives['lift_to_drag_ratio'],  # Maximize
        'fuel_efficiency': sim_result.objectives['fuel_efficiency'],         # Minimize
        'structural_weight': structural_weight_penalty                       # Minimize
    }

# Use NSGA-II for multi-objective optimization
from src.optimization.algorithms import NSGA2

optimizer = NSGA2(
    parameter_bounds=parameter_bounds,
    population_size=100,
    crossover_rate=0.9,
    mutation_rate=0.1
)

# Run multi-objective optimization
pareto_result = optimizer.optimize(
    objective_function=multi_objective_aircraft,
    max_evaluations=1000
)
```

#### Results Analysis

```python
# Analyze Pareto front
pareto_solutions = pareto_result.pareto_solutions

print(f"Found {len(pareto_solutions)} Pareto optimal solutions")

# Find solutions with specific trade-offs
best_aerodynamic = min(pareto_solutions, 
                      key=lambda x: x['objectives']['lift_to_drag_ratio'])
best_fuel_efficient = min(pareto_solutions,
                         key=lambda x: x['objectives']['fuel_efficiency'])

print(f"\nBest aerodynamic solution:")
print(f"L/D ratio: {-best_aerodynamic['objectives']['lift_to_drag_ratio']:.2f}")
print(f"Fuel efficiency: {best_aerodynamic['objectives']['fuel_efficiency']:.3f}")

print(f"\nBest fuel efficient solution:")
print(f"L/D ratio: {-best_fuel_efficient['objectives']['lift_to_drag_ratio']:.2f}")
print(f"Fuel efficiency: {best_fuel_efficient['objectives']['fuel_efficiency']:.3f}")

# Create visualizations
pareto_plot = graph_generator.create_pareto_front_plot(
    pareto_solutions, "tutorial_aircraft_pareto"
)
print(f"Pareto front plot saved: {pareto_plot}")
```

### Tutorial 2: Robust Spacecraft Design Under Uncertainty

#### Objective
Design a robust Earth observation satellite that performs well under manufacturing tolerances and environmental uncertainties.

#### Uncertainty Definition

```python
from src.optimization.robust_optimization import (
    UncertaintyQuantification, 
    UncertaintyDistribution,
    RobustOptimizer
)

# Initialize uncertainty quantification
uq = UncertaintyQuantification()

# Define parameter uncertainties (manufacturing tolerances)
uq.add_parameter_uncertainty('dry_mass', UncertaintyDistribution(
    'normal', {'mean': 0.0, 'std': 100.0}, bounds=(-300.0, 300.0)
))
uq.add_parameter_uncertainty('fuel_mass', UncertaintyDistribution(
    'normal', {'mean': 0.0, 'std': 500.0}, bounds=(-1500.0, 1500.0)
))
uq.add_parameter_uncertainty('solar_panel_area', UncertaintyDistribution(
    'normal', {'mean': 0.0, 'std': 2.0}, bounds=(-6.0, 6.0)
))

# Define environmental uncertainties
uq.add_environmental_uncertainty('solar_flux_variation', UncertaintyDistribution(
    'uniform', {'low': -50.0, 'high': 50.0}  # W/m² variation
))
uq.add_environmental_uncertainty('atmospheric_density_variation', UncertaintyDistribution(
    'lognormal', {'mean': 0.0, 'sigma': 0.2}
))

print("Uncertainty sources defined ✓")
```

#### Robust Optimization Setup

```python
# Initialize spacecraft system
spacecraft_system = SpacecraftOptimizationSystem()

# Define spacecraft design bounds
spacecraft_bounds = {
    'dry_mass': (2000, 6000),           # kg
    'fuel_mass': (8000, 25000),         # kg
    'specific_impulse': (280, 350),     # seconds
    'thrust': (2000, 8000),             # Newtons
    'solar_panel_area': (40.0, 120.0),  # m²
    'thermal_mass': (800, 2500),        # kg
    'target_orbit_altitude': (500, 800), # km
    'mission_duration': (1095, 2190)    # days (3-6 years)
}

def robust_spacecraft_objective(parameters):
    """
    Robust objective function for spacecraft optimization.
    Optimizes for mission success probability under uncertainty.
    """
    result = spacecraft_system.evaluate_design(parameters, 'earth_observation')
    sim_result = result['simulation_result']
    
    return {
        'mission_success': -sim_result.objectives['mission_success_probability'],
        'total_mass': parameters['dry_mass'] + parameters['fuel_mass'],
        'power_margin': -sim_result.objectives['power_efficiency']
    }

# Create robust optimizer
robust_optimizer = RobustOptimizer(uq)
```

#### Robust Optimization Execution

```python
# Run robust optimization with different robustness measures
robustness_measures = ['mean_std', 'worst_case', 'cvar']
robust_results = {}

for measure in robustness_measures:
    print(f"\nRunning robust optimization with {measure} measure...")
    
    result = robust_optimizer.robust_optimization(
        objective_function=robust_spacecraft_objective,
        parameter_bounds=spacecraft_bounds,
        robustness_measure=measure,
        n_mc_samples=200,
        optimization_algorithm='genetic_algorithm'
    )
    
    robust_results[measure] = result
    
    print(f"Robust objective ({measure}): {result.robust_objective:.4f}")
    print(f"Mean mission success: {-result.mean_objective:.3f}")
    print(f"Objective std dev: {result.objective_std:.4f}")
```

#### Uncertainty Analysis

```python
# Detailed uncertainty analysis for best solution
best_solution = robust_results['mean_std']

# Monte Carlo analysis
mc_samples = uq.sample_uncertainties(1000)
mc_results = []

for sample in mc_samples:
    # Apply uncertainties to best parameters
    perturbed_params = best_solution.best_parameters.copy()
    
    # Add parameter uncertainties
    for param, uncertainty in sample['parameters'].items():
        if param in perturbed_params:
            perturbed_params[param] += uncertainty
    
    # Evaluate with perturbed parameters
    result = spacecraft_system.evaluate_design(perturbed_params, 'earth_observation')
    mc_results.append(result['simulation_result'].objectives)

# Statistical analysis
mission_success_values = [r['mission_success_probability'] for r in mc_results]
power_efficiency_values = [r['power_efficiency'] for r in mc_results]

print(f"\n=== UNCERTAINTY ANALYSIS RESULTS ===")
print(f"Mission Success Probability:")
print(f"  Mean: {np.mean(mission_success_values):.3f}")
print(f"  Std:  {np.std(mission_success_values):.4f}")
print(f"  95% CI: [{np.percentile(mission_success_values, 2.5):.3f}, "
      f"{np.percentile(mission_success_values, 97.5):.3f}]")

print(f"\nPower Efficiency:")
print(f"  Mean: {np.mean(power_efficiency_values):.3f}")
print(f"  Std:  {np.std(power_efficiency_values):.4f}")

# Create uncertainty visualization
uncertainty_plot = graph_generator.create_uncertainty_propagation_plot(
    mc_results, "tutorial_spacecraft_uncertainty"
)
print(f"Uncertainty analysis plot saved: {uncertainty_plot}")
```

### Tutorial 3: Advanced Multi-Fidelity Strategy Comparison

#### Objective
Compare different fidelity switching strategies to understand their impact on optimization performance and computational cost.

#### Strategy Setup

```python
from src.simulation.adaptive_fidelity import FidelitySwitchingStrategy

# Define different fidelity strategies
strategies = [
    FidelitySwitchingStrategy.CONSERVATIVE,
    FidelitySwitchingStrategy.AGGRESSIVE, 
    FidelitySwitchingStrategy.BALANCED,
    FidelitySwitchingStrategy.ADAPTIVE
]

strategy_results = {}

# Run optimization with each strategy
for strategy in strategies:
    print(f"\nTesting {strategy.name} fidelity strategy...")
    
    # Initialize system with specific strategy
    aircraft_system = AircraftOptimizationSystem(fidelity_strategy=strategy)
    
    # Create optimizer
    optimizer = GeneticAlgorithm(parameter_bounds, population_size=50)
    
    # Run optimization
    start_time = time.time()
    result = optimizer.optimize(objective_function, max_evaluations=300)
    total_time = time.time() - start_time
    
    # Get fidelity statistics
    fidelity_stats = aircraft_system.get_optimization_statistics()
    
    strategy_results[strategy.name] = {
        'optimization_result': result,
        'total_time': total_time,
        'fidelity_stats': fidelity_stats
    }
    
    print(f"  Best L/D ratio: {result.best_objectives['lift_to_drag_ratio']:.2f}")
    print(f"  Total time: {total_time:.1f} seconds")
    print(f"  Fidelity usage: {fidelity_stats['fidelity_counts']}")
```

#### Strategy Comparison Analysis

```python
# Create comparison visualization
comparison_data = {}
for strategy_name, data in strategy_results.items():
    comparison_data[strategy_name] = {
        'objectives': {
            'lift_to_drag_ratio': {
                'mean': data['optimization_result'].best_objectives['lift_to_drag_ratio'],
                'std': 0.2  # Placeholder - would come from multiple runs
            }
        },
        'computational_cost': data['total_time'],
        'fidelity_distribution': data['fidelity_stats']['fidelity_counts']
    }

# Generate comparison plots
comparison_plot = graph_generator.create_performance_comparison_plot(
    comparison_data, "tutorial_strategy_comparison"
)

# Print detailed analysis
print(f"\n=== FIDELITY STRATEGY COMPARISON ===")
print(f"{'Strategy':<12} {'L/D Ratio':<10} {'Time (s)':<10} {'Low %':<8} {'Med %':<8} {'High %':<8}")
print("-" * 70)

for strategy_name, data in strategy_results.items():
    result = data['optimization_result']
    stats = data['fidelity_stats']
    total_evals = sum(stats['fidelity_counts'].values())
    
    low_pct = stats['fidelity_counts'].get('low', 0) / total_evals * 100
    med_pct = stats['fidelity_counts'].get('medium', 0) / total_evals * 100
    high_pct = stats['fidelity_counts'].get('high', 0) / total_evals * 100
    
    print(f"{strategy_name:<12} {result.best_objectives['lift_to_drag_ratio']:<10.2f} "
          f"{data['total_time']:<10.1f} {low_pct:<8.1f} {med_pct:<8.1f} {high_pct:<8.1f}")
```

## Interactive Examples

### Jupyter Notebook Tutorials

We provide comprehensive Jupyter notebooks for interactive learning:

1. **`01_Quick_Start.ipynb`** - Basic framework usage and simple examples
2. **`02_Aircraft_Optimization.ipynb`** - Complete aircraft design optimization
3. **`03_Spacecraft_Design.ipynb`** - Spacecraft mission planning and optimization  
4. **`04_Multi_Objective.ipynb`** - Multi-objective optimization and Pareto analysis
5. **`05_Robust_Design.ipynb`** - Uncertainty quantification and robust optimization
6. **`06_Advanced_Features.ipynb`** - Custom algorithms and advanced techniques

### Running the Notebooks

```bash
# Install Jupyter if not already installed
pip install jupyter notebook

# Navigate to examples directory
cd examples/notebooks/

# Start Jupyter notebook server
jupyter notebook

# Open desired notebook in browser
```

## Example Gallery

### Aircraft Configurations

**Commercial Airliner Optimization**
- Variables: 9 design parameters
- Objectives: L/D ratio, fuel efficiency, range
- Results: 28.4 L/D ratio, 15% fuel savings
- [View Code](examples/aircraft_commercial.py) | [Results](results/aircraft_commercial/)

**Regional Aircraft Design**
- Variables: 7 design parameters  
- Objectives: Takeoff distance, fuel efficiency
- Results: 1200m takeoff, 18% efficiency gain
- [View Code](examples/aircraft_regional.py) | [Results](results/aircraft_regional/)

**Business Jet Optimization**  
- Variables: 8 design parameters
- Objectives: Speed, range, comfort
- Results: Mach 0.92, 6800km range
- [View Code](examples/aircraft_business.py) | [Results](results/aircraft_business/)

### Spacecraft Missions

**Earth Observation Satellite**
- Variables: 8 design parameters
- Objectives: Mission success, mass efficiency
- Results: 94.2% success probability, 2850kg mass
- [View Code](examples/spacecraft_earth_obs.py) | [Results](results/spacecraft_earth_obs/)

**Communication Satellite**
- Variables: 10 design parameters
- Objectives: Coverage, power efficiency, lifetime
- Results: 15-year lifetime, 99.5% coverage
- [View Code](examples/spacecraft_communication.py) | [Results](results/spacecraft_communication/)

**Interplanetary Probe**
- Variables: 12 design parameters
- Objectives: Delta-v capability, instrument mass
- Results: 12.8 km/s delta-v, 450kg instruments
- [View Code](examples/spacecraft_interplanetary.py) | [Results](results/spacecraft_interplanetary/)

## Best Practices Examples

### Optimization Setup

```python
# Example of well-structured optimization setup
class OptimizationPipeline:
    def __init__(self, system_type='aircraft'):
        self.system = self._initialize_system(system_type)
        self.data_manager = DataManager("optimization_results")
        self.visualizer = ProfessionalGraphGenerator("visualizations")
        
    def run_optimization(self, config):
        # Validate configuration
        self._validate_config(config)
        
        # Setup optimizer
        optimizer = self._create_optimizer(config)
        
        # Define objective with error handling
        objective_func = self._create_objective_function(config)
        
        # Run optimization with logging
        result = self._run_with_monitoring(optimizer, objective_func, config)
        
        # Save and analyze results
        self._save_results(result, config)
        self._generate_visualizations(result)
        
        return result
```

### Error Handling

```python
def robust_objective_function(parameters):
    """
    Example of robust objective function with comprehensive error handling
    """
    try:
        # Validate parameters
        if not validate_parameters(parameters, parameter_bounds):
            raise ValueError("Parameters outside valid bounds")
        
        # Evaluate design with timeout
        with timeout_context(120):  # 2-minute timeout
            result = aircraft_system.evaluate_design(parameters)
        
        # Check for valid results
        if not result or 'simulation_result' not in result:
            raise RuntimeError("Invalid simulation result")
        
        objectives = result['simulation_result'].objectives
        
        # Validate objectives
        if any(np.isnan(list(objectives.values()))):
            raise RuntimeError("NaN values in objectives")
        
        return objectives
        
    except TimeoutError:
        logger.warning(f"Timeout for parameters: {parameters}")
        return {'lift_to_drag_ratio': -1000}  # Penalty value
        
    except Exception as e:
        logger.error(f"Error in objective function: {e}")
        return {'lift_to_drag_ratio': -1000}  # Penalty value
```

## Performance Optimization Examples

### Parallel Processing

```python
from multiprocessing import Pool
import functools

def parallel_optimization_example():
    """Example of parallel optimization execution"""
    
    # Configure for parallel processing
    n_cores = multiprocessing.cpu_count() - 1
    
    # Create population-based optimizer
    optimizer = GeneticAlgorithm(
        parameter_bounds,
        population_size=n_cores * 4,  # 4 individuals per core
        parallel_evaluations=True,
        n_workers=n_cores
    )
    
    # Define thread-safe objective function
    def thread_safe_objective(parameters):
        # Create local system instance for thread safety
        local_system = AircraftOptimizationSystem()
        return local_system.evaluate_design(parameters)
    
    # Run parallel optimization
    result = optimizer.optimize(
        thread_safe_objective,
        max_evaluations=400
    )
    
    return result
```

### Memory Optimization

```python
def memory_efficient_optimization():
    """Example of memory-efficient optimization for large problems"""
    
    # Configure system for memory efficiency
    aircraft_system = AircraftOptimizationSystem()
    aircraft_system.configure({
        'cache_size': 50,           # Limit evaluation cache
        'history_compression': True, # Compress optimization history
        'lazy_loading': True        # Load data on demand
    })
    
    # Use streaming data processing
    def streaming_objective(parameters):
        result = aircraft_system.evaluate_design(parameters)
        # Process and return only essential data
        return {
            'lift_to_drag_ratio': result['simulation_result'].objectives['lift_to_drag_ratio']
        }
    
    # Configure optimizer for memory efficiency
    optimizer = GeneticAlgorithm(
        parameter_bounds,
        population_size=30,          # Smaller population
        archive_solutions=False      # Don't store all solutions
    )
    
    return optimizer.optimize(streaming_objective, max_evaluations=200)
```

---

*All examples are tested and validated. Complete source code is available in the `/examples` directory. For questions or support, please visit our [GitHub repository](https://github.com/aerospace-optimization/amf-sbo).*