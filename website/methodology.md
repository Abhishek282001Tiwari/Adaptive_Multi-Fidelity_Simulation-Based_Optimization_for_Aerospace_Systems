---
layout: page
title: "Methodology"
description: "Detailed explanation of the multi-fidelity approach and optimization algorithms"
---

## Multi-Fidelity Simulation Framework

### Fidelity Levels and Models

Our framework employs three distinct fidelity levels, each offering different trade-offs between computational cost and accuracy:

#### Low Fidelity Models
- **Computational Cost**: 0.05-0.1 seconds per evaluation
- **Accuracy**: ±15-20% typical error
- **Applications**: Initial design space exploration, rapid prototyping
- **Aircraft Models**: Simplified lift and drag equations, basic atmospheric models
- **Spacecraft Models**: Analytical orbital mechanics, simplified thermal models

#### Medium Fidelity Models  
- **Computational Cost**: 1-5 seconds per evaluation
- **Accuracy**: ±8-12% typical error
- **Applications**: Design refinement, constraint verification
- **Enhanced Physics**: Viscous effects, compressibility corrections, detailed geometry

#### High Fidelity Models
- **Computational Cost**: 15-25 seconds per evaluation  
- **Accuracy**: ±3-5% typical error
- **Applications**: Final design validation, critical analysis
- **Aircraft Models**: CFD approximations, detailed Reynolds number effects
- **Spacecraft Models**: Orbital perturbations, detailed thermal analysis

### Adaptive Fidelity Switching

The system automatically selects the optimal fidelity level using multiple criteria:

#### Decision Factors
1. **Convergence Status**: Switch to higher fidelity as optimization converges
2. **Uncertainty Levels**: Use high fidelity in regions with high uncertainty
3. **Computational Budget**: Balance accuracy needs with available resources
4. **Expected Improvement**: Prioritize high fidelity where improvement is likely

#### Switching Strategies
- **Conservative**: Prefer low fidelity, switch only when confident
- **Aggressive**: Use high fidelity early and frequently  
- **Balanced**: Moderate approach balancing cost and accuracy
- **Adaptive**: Machine learning-based dynamic strategy selection

## Optimization Algorithms

### Genetic Algorithm (GA)

#### Algorithm Parameters
- **Population Size**: 50-100 individuals
- **Selection Method**: Tournament selection with size 3
- **Crossover Rate**: 0.8 (blend crossover)
- **Mutation Rate**: 0.1 (Gaussian mutation)
- **Elite Preservation**: Top 10% carried forward

#### Key Features
- Multi-objective capability using NSGA-II framework
- Adaptive parameter tuning based on convergence
- Constraint handling through penalty functions
- Parallel evaluation of population members

### Particle Swarm Optimization (PSO)

#### Algorithm Parameters  
- **Swarm Size**: 30-50 particles
- **Inertia Weight**: 0.729 (linearly decreasing)
- **Cognitive Parameter**: 1.49445
- **Social Parameter**: 1.49445
- **Velocity Clamping**: ±20% of search space

#### Key Features
- Global and local best tracking
- Adaptive velocity updates
- Boundary constraint handling
- Convergence acceleration techniques

### Bayesian Optimization

#### Algorithm Components
- **Surrogate Model**: Gaussian Process with Matérn kernel
- **Acquisition Function**: Expected Improvement (EI)
- **Initial Sampling**: Latin Hypercube Design (10-20 points)
- **Kernel Parameters**: Automatic relevance determination

#### Key Features
- Efficient for expensive function evaluations
- Built-in uncertainty quantification
- Active learning for optimal sampling
- Multiple acquisition function options (EI, UCB, PI)

## Uncertainty Quantification

### Sources of Uncertainty

#### Design Parameter Uncertainty
- Manufacturing tolerances (±1-5%)
- Material property variations
- Assembly and installation errors
- Operational wear and degradation

#### Environmental Uncertainty  
- Atmospheric condition variations
- Temperature and pressure fluctuations
- Wind and turbulence effects
- Space environment variations (radiation, debris)

#### Model Uncertainty
- Simplified physics assumptions
- Numerical approximation errors
- Calibration and validation uncertainties
- Scale-up and extrapolation errors

### Uncertainty Analysis Methods

#### Monte Carlo Sampling
- **Sample Size**: 100-1000 evaluations per design point
- **Sampling Method**: Latin Hypercube or Sobol sequences
- **Distribution Types**: Normal, uniform, lognormal, beta
- **Output Statistics**: Mean, variance, percentiles, confidence intervals

#### Sensitivity Analysis
- **Morris Screening**: Efficient factor ranking method
- **Sobol Indices**: Variance-based sensitivity measures  
- **Parameter Ranking**: Identify most influential variables
- **Interaction Effects**: Detect parameter coupling

#### Polynomial Chaos Expansion
- **Basis Functions**: Orthogonal polynomials matched to input distributions
- **Expansion Order**: 2nd-4th order depending on problem complexity
- **Coefficient Estimation**: Least squares regression
- **Uncertainty Propagation**: Analytical moments calculation

## Robust Optimization

### Robustness Measures

#### Mean-Standard Deviation
- Minimize: μ(f) - k×σ(f)
- Balance between performance and reliability
- Tunable risk parameter k (typically 1-3)

#### Worst-Case Optimization  
- Minimize: max(f) over uncertainty range
- Conservative approach for safety-critical applications
- Guarantees performance under all scenarios

#### Conditional Value at Risk (CVaR)
- Minimize: Expected value of worst α% outcomes
- Risk-aware optimization for financial applications
- Flexible risk tolerance through α parameter

### Multi-Objective Formulation

#### Objective Functions
- **Performance**: Maximize primary design metrics
- **Robustness**: Minimize sensitivity to uncertainties  
- **Reliability**: Maximize constraint satisfaction probability
- **Cost**: Minimize computational or manufacturing cost

#### Pareto Front Analysis
- Trade-off visualization between objectives
- Decision support for design selection
- Sensitivity analysis of Pareto optimal solutions

## Aerospace Performance Models

### Aircraft Models

#### Aerodynamic Performance
- **Lift Coefficient**: Function of angle of attack, Mach number, Reynolds number
- **Drag Coefficient**: Parasitic drag + induced drag + compressibility effects
- **Lift-to-Drag Ratio**: Primary performance metric for efficiency
- **Stall Characteristics**: Critical angle of attack and stall speed

#### Mission Performance
- **Range**: Breguet range equation with realistic fuel fractions
- **Fuel Efficiency**: Thrust-specific fuel consumption models
- **Payload Capacity**: Weight and balance constraints
- **Takeoff/Landing**: Field length requirements

### Spacecraft Models

#### Orbital Mechanics
- **Delta-V Requirements**: Hohmann transfers, plane changes, orbit maintenance
- **Propellant Mass**: Rocket equation with realistic specific impulse
- **Mission Duration**: Orbital decay, eclipse effects, aging
- **Attitude Control**: Reaction wheel sizing and fuel requirements

#### Subsystem Performance  
- **Power Generation**: Solar array efficiency, degradation, eclipse effects
- **Thermal Management**: Heat generation, dissipation, temperature control
- **Communications**: Link budgets, antenna sizing, power requirements
- **Structural Design**: Launch loads, thermal stress, fatigue life

## Implementation Details

### Software Architecture

#### Object-Oriented Design
- Base classes for simulations, optimizers, and uncertainty models
- Polymorphic interfaces for algorithm interchangeability  
- Modular design for easy extension and customization
- Comprehensive error handling and logging

#### Performance Optimization
- Parallel evaluation of objective functions
- Efficient memory management for large populations
- Caching of expensive function evaluations
- Adaptive convergence criteria

#### Data Management
- Automatic saving of optimization progress
- Multiple export formats (CSV, Excel, HDF5)
- Comprehensive metadata tracking
- Result comparison and analysis tools

### Validation and Verification

#### Benchmark Problems
- Standard optimization test functions
- Published aerospace design problems
- Comparison with literature results
- Cross-validation between algorithms

#### Code Quality
- Unit testing for all major components
- Integration testing of complete workflows
- Performance profiling and optimization
- Documentation and code review processes

This methodology provides a comprehensive framework for aerospace optimization that balances accuracy, computational efficiency, and design robustness while providing detailed uncertainty analysis and decision support capabilities.