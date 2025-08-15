# API Reference

## Table of Contents

1. [Simulation Framework](#simulation-framework)
2. [Optimization Algorithms](#optimization-algorithms)
3. [Aerospace Models](#aerospace-models)
4. [Uncertainty Quantification](#uncertainty-quantification)
5. [Data Management](#data-management)
6. [Visualization](#visualization)

---

## Simulation Framework

### BaseSimulation

Abstract base class for all simulation models.

```python
class BaseSimulation(ABC):
    def __init__(self, name: str, fidelity_level: FidelityLevel)
```

**Parameters:**
- `name` (str): Name identifier for the simulation
- `fidelity_level` (FidelityLevel): Computational fidelity level

**Abstract Methods:**
- `evaluate(parameters: Dict[str, float]) -> SimulationResult`
- `get_computational_cost() -> float`
- `validate_parameters(parameters: Dict[str, float]) -> bool`

**Methods:**
- `set_parameter_bounds(bounds: Dict[str, Tuple[float, float]])`: Set parameter bounds
- `get_parameter_bounds() -> Dict[str, Tuple[float, float]]`: Get parameter bounds
- `get_expected_runtime(parameters: Dict[str, float]) -> float`: Estimate runtime

### SimulationResult

Container for simulation results.

```python
class SimulationResult:
    def __init__(self, 
                 objectives: Dict[str, float],
                 constraints: Optional[Dict[str, float]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 fidelity_level: FidelityLevel = FidelityLevel.LOW,
                 computation_time: float = 0.0,
                 uncertainty: Optional[Dict[str, float]] = None)
```

**Attributes:**
- `objectives` (Dict[str, float]): Objective function values
- `constraints` (Dict[str, float]): Constraint values
- `metadata` (Dict[str, Any]): Additional metadata
- `fidelity_level` (FidelityLevel): Fidelity level used
- `computation_time` (float): Computation time in seconds
- `uncertainty` (Dict[str, float]): Uncertainty estimates
- `timestamp` (float): Creation timestamp

### MultiFidelitySimulation

Manager for multiple fidelity levels.

```python
class MultiFidelitySimulation:
    def __init__(self, name: str)
```

**Methods:**
- `add_simulation(simulation: BaseSimulation)`: Add fidelity level
- `evaluate(parameters: Dict[str, float], force_fidelity: Optional[FidelityLevel] = None) -> SimulationResult`: Evaluate with adaptive or forced fidelity
- `get_fidelity_statistics() -> Dict[str, Any]`: Get usage statistics
- `reset_statistics()`: Reset all statistics

### AdaptiveFidelityManager

Intelligent fidelity switching management.

```python
class AdaptiveFidelityManager:
    def __init__(self, strategy: FidelitySwitchingStrategy = FidelitySwitchingStrategy.ADAPTIVE)
```

**Parameters:**
- `strategy` (FidelitySwitchingStrategy): Switching strategy

**Methods:**
- `select_fidelity(parameters: Dict[str, float], optimization_progress: Optional[Dict[str, Any]] = None, multi_fidelity_sim: Optional[MultiFidelitySimulation] = None) -> FidelityLevel`: Select optimal fidelity
- `update_evaluation_history(parameters: Dict[str, float], result: SimulationResult, computation_time: float)`: Update history
- `get_fidelity_statistics() -> Dict[str, Any]`: Get statistics
- `reset()`: Reset manager state

---

## Optimization Algorithms

### BaseOptimizer

Abstract base class for optimization algorithms.

```python
class BaseOptimizer(ABC):
    def __init__(self, name: str, parameter_bounds: Dict[str, Tuple[float, float]])
```

**Abstract Methods:**
- `optimize(objective_function: Callable, max_evaluations: int, **kwargs) -> OptimizationResult`

**Methods:**
- `_normalize_parameters(parameters: Dict[str, float]) -> Dict[str, float]`: Normalize to [0,1]
- `_denormalize_parameters(normalized_params: Dict[str, float]) -> Dict[str, float]`: Denormalize from [0,1]
- `_ensure_bounds(parameters: Dict[str, float]) -> Dict[str, float]`: Enforce bounds

### GeneticAlgorithm

Genetic Algorithm implementation.

```python
class GeneticAlgorithm(BaseOptimizer):
    def __init__(self, 
                 parameter_bounds: Dict[str, Tuple[float, float]], 
                 population_size: int = 50, 
                 crossover_rate: float = 0.8, 
                 mutation_rate: float = 0.1, 
                 tournament_size: int = 3)
```

**Parameters:**
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Variable bounds
- `population_size` (int): Population size (default: 50)
- `crossover_rate` (float): Crossover probability (default: 0.8)
- `mutation_rate` (float): Mutation probability (default: 0.1)
- `tournament_size` (int): Tournament selection size (default: 3)

**Methods:**
- `optimize(objective_function: Callable, max_evaluations: int, **kwargs) -> OptimizationResult`: Run optimization

### ParticleSwarmOptimization

Particle Swarm Optimization implementation.

```python
class ParticleSwarmOptimization(BaseOptimizer):
    def __init__(self, 
                 parameter_bounds: Dict[str, Tuple[float, float]], 
                 swarm_size: int = 30, 
                 w: float = 0.729, 
                 c1: float = 1.49445, 
                 c2: float = 1.49445)
```

**Parameters:**
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Variable bounds
- `swarm_size` (int): Swarm size (default: 30)
- `w` (float): Inertia weight (default: 0.729)
- `c1` (float): Cognitive parameter (default: 1.49445)
- `c2` (float): Social parameter (default: 1.49445)

### BayesianOptimization

Bayesian Optimization with Gaussian Processes.

```python
class BayesianOptimization(BaseOptimizer):
    def __init__(self, 
                 parameter_bounds: Dict[str, Tuple[float, float]], 
                 acquisition_function: str = 'ei', 
                 xi: float = 0.01, 
                 kappa: float = 2.576)
```

**Parameters:**
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Variable bounds
- `acquisition_function` (str): Acquisition function ('ei', 'ucb', 'poi')
- `xi` (float): Exploration parameter for EI (default: 0.01)
- `kappa` (float): Exploration parameter for UCB (default: 2.576)

### OptimizationResult

Container for optimization results.

```python
@dataclass
class OptimizationResult:
    best_parameters: Dict[str, float]
    best_objectives: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    convergence_achieved: bool
    total_time: float
    algorithm_name: str
    metadata: Dict[str, Any]
```

---

## Aerospace Models

### AircraftOptimizationSystem

Complete aircraft optimization system.

```python
class AircraftOptimizationSystem:
    def __init__(self, fidelity_strategy: FidelitySwitchingStrategy = FidelitySwitchingStrategy.ADAPTIVE)
```

**Methods:**
- `evaluate_design(parameters: Dict[str, float], mission_profile: str = 'commercial', optimization_progress: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`: Evaluate aircraft design
- `get_optimization_statistics() -> Dict[str, Any]`: Get statistics

**Mission Profiles:**
- `'commercial'`: Long-range commercial airliner
- `'regional'`: Short to medium-range regional aircraft  
- `'business_jet'`: High-performance business jet

### SpacecraftOptimizationSystem

Complete spacecraft optimization system.

```python
class SpacecraftOptimizationSystem:
    def __init__(self, fidelity_strategy: FidelitySwitchingStrategy = FidelitySwitchingStrategy.ADAPTIVE)
```

**Methods:**
- `evaluate_design(parameters: Dict[str, float], mission_type: str = 'earth_observation', optimization_progress: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`: Evaluate spacecraft design
- `get_optimization_statistics() -> Dict[str, Any]`: Get statistics

**Mission Types:**
- `'earth_observation'`: Low Earth orbit observation satellite
- `'communication'`: Geostationary communication satellite
- `'deep_space'`: Interplanetary exploration probe

### LowFidelityAerodynamics

Fast analytical aerodynamics model.

```python
class LowFidelityAerodynamics(BaseSimulation):
    def __init__(self)
```

**Computational Cost:** ~0.1 seconds
**Accuracy:** ±15-20%
**Parameters:**
- `wingspan` (10.0-80.0 m): Wing span
- `wing_area` (20.0-500.0 m²): Wing planform area
- `aspect_ratio` (5.0-15.0): Wing aspect ratio
- `sweep_angle` (0.0-45.0 deg): Wing sweep angle
- `taper_ratio` (0.3-1.0): Wing taper ratio
- `thickness_ratio` (0.08-0.18): Airfoil thickness ratio
- `cruise_altitude` (1000.0-15000.0 m): Cruise altitude
- `cruise_mach` (0.1-0.9): Cruise Mach number
- `weight` (5000.0-100000.0 kg): Aircraft weight

**Objectives:**
- `lift_to_drag_ratio`: Aerodynamic efficiency
- `fuel_efficiency`: Fuel consumption efficiency
- `range`: Maximum range (km)
- `payload_capacity`: Payload capacity (kg)

### HighFidelityAerodynamics

Detailed CFD approximation model.

```python
class HighFidelityAerodynamics(BaseSimulation):
    def __init__(self)
```

**Computational Cost:** ~15-25 seconds
**Accuracy:** ±3-5%
**Additional Features:**
- Viscous effects modeling
- Compressibility corrections
- Reynolds number dependencies
- Flutter analysis
- Detailed structural margins

---

## Uncertainty Quantification

### UncertaintyQuantification

Uncertainty modeling and sampling.

```python
class UncertaintyQuantification:
    def __init__(self)
```

**Methods:**
- `add_parameter_uncertainty(parameter_name: str, uncertainty: UncertaintyDistribution)`: Add parameter uncertainty
- `add_environmental_uncertainty(env_name: str, uncertainty: UncertaintyDistribution)`: Add environmental uncertainty
- `add_model_uncertainty(model_name: str, uncertainty: UncertaintyDistribution)`: Add model uncertainty
- `sample_uncertainties(n_samples: int) -> List[Dict[str, Dict[str, float]]]`: Generate samples
- `compute_statistics(samples: List[float]) -> Dict[str, float]`: Compute statistics
- `compute_confidence_intervals(samples: List[float], confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Tuple[float, float]]`: Compute intervals

### UncertaintyDistribution

Uncertainty distribution specification.

```python
@dataclass
class UncertaintyDistribution:
    distribution_type: str
    parameters: Dict[str, float]
    bounds: Optional[Tuple[float, float]] = None
```

**Supported Distributions:**
- `'normal'`: Normal distribution (mean, std)
- `'uniform'`: Uniform distribution (low, high)
- `'lognormal'`: Log-normal distribution (mean, sigma)
- `'beta'`: Beta distribution (alpha, beta)
- `'triangular'`: Triangular distribution (left, mode, right)

### RobustOptimizer

Robust optimization under uncertainty.

```python
class RobustOptimizer:
    def __init__(self, uncertainty_quantification: UncertaintyQuantification)
```

**Methods:**
- `robust_optimization(objective_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], robustness_measure: str = 'mean_std', n_mc_samples: int = 100, optimization_algorithm: str = 'nelder_mead') -> RobustOptimizationResult`: Run robust optimization

**Robustness Measures:**
- `'mean_std'`: μ(f) - k×σ(f)
- `'worst_case'`: min(f) over uncertainty range
- `'cvar'`: Conditional Value at Risk
- `'mean'`: Expected value

### SensitivityAnalysis

Global sensitivity analysis.

```python
class SensitivityAnalysis:
    def __init__(self)
```

**Methods:**
- `morris_screening(objective_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], n_trajectories: int = 10, n_levels: int = 4) -> Dict[str, Dict[str, float]]`: Morris screening method
- `sobol_indices(objective_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], n_samples: int = 1000) -> Dict[str, Dict[str, float]]`: Sobol indices

---

## Data Management

### DataManager

Comprehensive data management system.

```python
class DataManager:
    def __init__(self, base_path: str = "data")
```

**Methods:**
- `save_optimization_run(run_id: str, optimization_result: Any, algorithm_name: str, system_type: str, parameters: Dict[str, Any]) -> str`: Save optimization run
- `load_optimization_run(run_id: str) -> Optional[Dict[str, Any]]`: Load optimization run
- `export_to_csv(run_ids: List[str], filename: str) -> str`: Export to CSV
- `export_to_excel(run_ids: List[str], filename: str) -> str`: Export to Excel
- `export_to_hdf5(run_ids: List[str], filename: str) -> str`: Export to HDF5
- `create_comparison_report(run_ids: List[str], report_name: str) -> str`: Create comparison report
- `get_run_statistics() -> Dict[str, Any]`: Get run statistics
- `cleanup_old_results(days_old: int = 30)`: Cleanup old results

### ResultsAnalyzer

Results analysis and comparison.

```python
class ResultsAnalyzer:
    def __init__(self, data_manager: DataManager)
```

**Methods:**
- `analyze_algorithm_performance(algorithm_name: str) -> Dict[str, Any]`: Analyze algorithm performance

---

## Visualization

### ProfessionalGraphGenerator

Professional graph generation system.

```python
class ProfessionalGraphGenerator:
    def __init__(self, output_dir: str = "visualizations", style: str = "aerospace")
```

**Methods:**
- `create_convergence_plot(optimization_history: List[Dict[str, Any]], algorithm_name: str, save_filename: str) -> str`: Create convergence plot
- `create_pareto_front_plot(optimization_results: List[Dict[str, Any]], save_filename: str) -> str`: Create Pareto front plot
- `create_fidelity_switching_plot(fidelity_history: List[Dict[str, Any]], save_filename: str) -> str`: Create fidelity switching plot
- `create_uncertainty_propagation_plot(monte_carlo_results: List[Dict[str, Any]], save_filename: str) -> str`: Create uncertainty plot
- `create_performance_comparison_plot(comparison_data: Dict[str, Any], save_filename: str) -> str`: Create comparison plot
- `create_3d_design_space_plot(optimization_history: List[Dict[str, Any]], param_names: List[str], save_filename: str) -> str`: Create 3D plot
- `create_interactive_dashboard(optimization_data: Dict[str, Any], save_filename: str) -> str`: Create interactive dashboard
- `create_statistical_distribution_plot(data: Dict[str, List[float]], save_filename: str) -> str`: Create distribution plot
- `generate_comprehensive_report(optimization_results: List[Dict[str, Any]], report_name: str) -> Dict[str, str]`: Generate comprehensive report

---

## Usage Examples

### Basic Aircraft Optimization

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

# Define objective
def objective_function(parameters):
    result = aircraft_system.evaluate_design(parameters, 'commercial')
    return result['simulation_result']

# Run optimization
result = optimizer.optimize(objective_function, max_evaluations=200)
```

### Robust Optimization Example

```python
from src.optimization.robust_optimization import UncertaintyQuantification, UncertaintyDistribution, RobustOptimizer

# Setup uncertainties
uq = UncertaintyQuantification()
uq.add_parameter_uncertainty('wingspan', UncertaintyDistribution(
    'normal', {'mean': 0.0, 'std': 0.5}, bounds=(-2.0, 2.0)
))

# Create robust optimizer
robust_optimizer = RobustOptimizer(uq)

# Run robust optimization
robust_result = robust_optimizer.robust_optimization(
    objective_function=objective_function,
    parameter_bounds=parameter_bounds,
    robustness_measure='mean_std',
    n_mc_samples=100
)
```

### Visualization Example

```python
from src.visualization.graph_generator import ProfessionalGraphGenerator

# Create visualizations
graph_generator = ProfessionalGraphGenerator("visualizations/")

# Generate convergence plot
graph_generator.create_convergence_plot(
    result.optimization_history,
    "Genetic Algorithm",
    "aircraft_convergence"
)

# Generate comparison plot
comparison_data = {
    'GA': {'objectives': {'lift_to_drag_ratio': {'mean': 18.5, 'std': 1.2}}},
    'PSO': {'objectives': {'lift_to_drag_ratio': {'mean': 17.8, 'std': 0.9}}}
}
graph_generator.create_performance_comparison_plot(
    comparison_data,
    "algorithm_comparison"
)
```

---

## Error Handling

All methods include comprehensive error handling:

- **Parameter Validation**: Invalid parameters raise `ValueError`
- **Bound Checking**: Out-of-bounds parameters are automatically clipped
- **Convergence Issues**: Non-convergent optimizations return partial results
- **File I/O Errors**: Data management operations include retry logic
- **Numerical Stability**: All computations include numerical safeguards

## Performance Considerations

- **Parallel Processing**: Most algorithms support parallel evaluation
- **Caching**: Expensive evaluations are cached automatically  
- **Memory Management**: Large datasets use efficient storage formats
- **Scalability**: Framework scales from single-core to HPC clusters

## Configuration

Global configuration is managed through `config.json`:

```python
import json
with open('config.json', 'r') as f:
    config = json.load(f)
    
# Access configuration
aircraft_bounds = config['aircraft']['parameter_bounds']
optimization_params = config['optimization']['default_parameters']
```