# Contributing to Adaptive Multi-Fidelity Simulation-Based Optimization

Thank you for your interest in contributing to the AMF-SBO framework! This guide will help you get started with contributing to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Contributing Guidelines](#contributing-guidelines)
4. [Code Standards](#code-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Community Guidelines](#community-guidelines)

---

## Getting Started

### Ways to Contribute

We welcome contributions in many forms:

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new capabilities or improvements
- **Code Contributions**: Implement new features or fix bugs
- **Documentation**: Improve guides, tutorials, and API documentation
- **Examples**: Create new example problems and use cases
- **Testing**: Add test cases and improve test coverage
- **Performance**: Optimize algorithms and computational efficiency

### Before You Start

1. **Check existing issues** on GitHub to see if your idea or bug has already been reported
2. **Discuss major changes** by opening an issue before starting implementation
3. **Read this guide** completely to understand our development process
4. **Review the codebase** to understand the architecture and coding style

---

## Development Setup

### Prerequisites

- **Python 3.8+**: Ensure you have Python 3.8 or newer
- **Git**: For version control and collaboration
- **Text Editor/IDE**: We recommend VS Code with Python extensions

### Environment Setup

#### 1. Fork and Clone the Repository

```bash
# Fork the repository on GitHub first, then clone your fork
git clone https://github.com/YOUR_USERNAME/amf-sbo.git
cd amf-sbo

# Add the original repository as upstream
git remote add upstream https://github.com/aerospace-optimization/amf-sbo.git
```

#### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

#### 3. Verify Setup

```bash
# Run tests to ensure everything works
python -m pytest tests/

# Run a simple example
python examples/aircraft_optimization_example.py
```

### Development Dependencies

The development environment includes additional tools:

```text
# Testing
pytest>=6.0.0
pytest-cov>=2.10.0
pytest-mock>=3.0.0

# Code Quality
black>=21.0.0
flake8>=3.8.0
isort>=5.0.0
mypy>=0.800

# Documentation
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0
nbsphinx>=0.8.0

# Pre-commit Hooks
pre-commit>=2.15.0
```

### Pre-commit Hooks

Set up pre-commit hooks to automatically check code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks on all files (optional)
pre-commit run --all-files
```

---

## Contributing Guidelines

### Issue Reporting

#### Bug Reports

When reporting bugs, please include:

```markdown
**Bug Description**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Set parameters to '...'
2. Run optimization with '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened, including error messages.

**Environment**
- OS: [e.g., Ubuntu 20.04, macOS 11.6, Windows 10]
- Python version: [e.g., 3.8.10]
- Package version: [e.g., 1.0.0]

**Additional Context**
- Configuration files used
- Data files (if applicable)
- Screenshots or plots showing the issue
```

#### Feature Requests

For feature requests, please provide:

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed Implementation**
If you have ideas about how to implement this, please share.

**Alternatives**
Any alternative solutions or workarounds you've considered.

**Additional Context**
Any other context, research papers, or examples.
```

### Development Workflow

#### 1. Choose an Issue

- Look for issues labeled `good first issue` for beginner-friendly tasks
- Check issues labeled `help wanted` for areas where we need assistance
- For major features, discuss the approach in the issue before coding

#### 2. Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

#### 3. Make Changes

- Write code following our [code standards](#code-standards)
- Add tests for new functionality
- Update documentation as needed
- Commit changes with descriptive messages

#### 4. Test Your Changes

```bash
# Run the full test suite
python -m pytest tests/

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Test specific modules
python -m pytest tests/test_optimization_algorithms.py

# Run examples to ensure they still work
python examples/aircraft_optimization_example.py
python examples/spacecraft_optimization_example.py
```

#### 5. Submit Pull Request

- Push your branch to your fork
- Open a pull request with a clear description
- Link to any related issues
- Be responsive to code review feedback

---

## Code Standards

### Code Style

We use **Black** for code formatting and **flake8** for linting:

```bash
# Format code
black src/ tests/ examples/

# Check linting
flake8 src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/
```

### Python Style Guidelines

#### General Principles

1. **Readability**: Code should be clear and self-documenting
2. **Consistency**: Follow established patterns in the codebase
3. **Simplicity**: Prefer simple, straightforward solutions
4. **Performance**: Consider computational efficiency for optimization code

#### Naming Conventions

```python
# Variables and functions: snake_case
parameter_bounds = {...}
def evaluate_design(parameters):
    pass

# Classes: PascalCase
class AircraftOptimizationSystem:
    pass

# Constants: UPPER_SNAKE_CASE
DEFAULT_POPULATION_SIZE = 50
MAX_ITERATIONS = 1000

# Private methods: leading underscore
def _validate_parameters(self, params):
    pass
```

#### Function Documentation

Use **Google-style docstrings**:

```python
def optimize_aircraft(parameters: Dict[str, float], 
                     mission_profile: str = 'commercial') -> OptimizationResult:
    """
    Optimize aircraft design for given mission profile.
    
    Args:
        parameters: Dictionary of design parameters with bounds
        mission_profile: Mission type ('commercial', 'regional', 'business_jet')
        
    Returns:
        OptimizationResult containing best design and performance metrics
        
    Raises:
        ValueError: If parameters are outside valid bounds
        RuntimeError: If optimization fails to converge
        
    Example:
        >>> bounds = {'wingspan': (30, 60), 'wing_area': (150, 350)}
        >>> result = optimize_aircraft(bounds, 'commercial')
        >>> print(f"Best L/D ratio: {result.best_objectives['lift_to_drag_ratio']:.2f}")
    """
```

#### Type Hints

Use type hints for better code clarity:

```python
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

def process_results(results: List[OptimizationResult], 
                   confidence_level: float = 0.95) -> Dict[str, float]:
    """Process optimization results with statistical analysis."""
    pass

# For complex types, create type aliases
ParameterBounds = Dict[str, Tuple[float, float]]
ObjectiveValues = Dict[str, float]
```

### Architecture Guidelines

#### Class Design

1. **Single Responsibility**: Each class should have one clear purpose
2. **Composition over Inheritance**: Prefer composition when possible
3. **Abstract Base Classes**: Use ABCs for interface definition

```python
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms."""
    
    def __init__(self, parameter_bounds: ParameterBounds):
        self.parameter_bounds = parameter_bounds
        self.evaluation_count = 0
    
    @abstractmethod
    def optimize(self, objective_function: Callable, 
                max_evaluations: int) -> OptimizationResult:
        """Run optimization algorithm."""
        pass
    
    def _validate_bounds(self, parameters: Dict[str, float]) -> bool:
        """Validate parameters against bounds."""
        # Implementation here
        pass
```

#### Error Handling

```python
# Use specific exception types
class OptimizationError(Exception):
    """Base exception for optimization errors."""
    pass

class ConvergenceError(OptimizationError):
    """Raised when optimization fails to converge."""
    pass

# Provide helpful error messages
def validate_parameters(parameters, bounds):
    for param, value in parameters.items():
        if param not in bounds:
            raise ValueError(
                f"Unknown parameter '{param}'. "
                f"Valid parameters: {list(bounds.keys())}"
            )
        
        lower, upper = bounds[param]
        if not (lower <= value <= upper):
            raise ValueError(
                f"Parameter '{param}' = {value} is outside bounds "
                f"[{lower}, {upper}]"
            )
```

### Performance Guidelines

#### Computational Efficiency

```python
# Use NumPy for numerical computations
import numpy as np

# Good: Vectorized operations
def compute_fitness(population):
    return np.sum(population**2, axis=1)

# Avoid: Python loops for large arrays
def compute_fitness_slow(population):
    fitness = []
    for individual in population:
        fitness.append(sum(x**2 for x in individual))
    return fitness

# Use appropriate data structures
from collections import defaultdict, deque

# For caching expensive computations
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_computation(parameters_tuple):
    # Convert tuple back to dict for processing
    parameters = dict(parameters_tuple)
    # ... expensive computation
    return result
```

---

## Testing Guidelines

### Test Structure

Organize tests to mirror the source code structure:

```
tests/
├── test_simulation_framework.py
├── test_optimization_algorithms.py
├── test_aerospace_models.py
├── test_uncertainty_quantification.py
├── test_data_management.py
├── test_visualization.py
├── integration/
│   ├── test_aircraft_optimization.py
│   └── test_spacecraft_optimization.py
└── benchmarks/
    ├── test_performance.py
    └── test_validation.py
```

### Unit Tests

Write comprehensive unit tests:

```python
import pytest
import numpy as np
from src.optimization.algorithms import GeneticAlgorithm

class TestGeneticAlgorithm:
    """Test suite for Genetic Algorithm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parameter_bounds = {
            'x1': (-5.0, 5.0),
            'x2': (-5.0, 5.0)
        }
        self.optimizer = GeneticAlgorithm(
            self.parameter_bounds,
            population_size=20
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.population_size == 20
        assert len(self.optimizer.parameter_bounds) == 2
    
    def test_sphere_function_optimization(self):
        """Test optimization on simple sphere function."""
        def sphere_function(params):
            return {'objective': params['x1']**2 + params['x2']**2}
        
        result = self.optimizer.optimize(
            sphere_function, 
            max_evaluations=100
        )
        
        # Check convergence to global minimum
        assert result.best_objectives['objective'] < 0.1
        assert abs(result.best_parameters['x1']) < 0.5
        assert abs(result.best_parameters['x2']) < 0.5
    
    def test_invalid_bounds(self):
        """Test handling of invalid parameter bounds."""
        with pytest.raises(ValueError):
            GeneticAlgorithm({'x1': (5.0, -5.0)})  # Invalid: lower > upper
    
    def test_convergence_callback(self):
        """Test convergence callback functionality."""
        callback_calls = []
        
        def callback(generation, best_fitness, population):
            callback_calls.append((generation, best_fitness))
        
        def simple_function(params):
            return {'objective': params['x1']**2}
        
        self.optimizer.optimize(
            simple_function,
            max_evaluations=50,
            callback=callback
        )
        
        assert len(callback_calls) > 0
        assert callback_calls[-1][1] <= callback_calls[0][1]  # Fitness should improve
```

### Integration Tests

Test complete workflows:

```python
def test_aircraft_optimization_workflow():
    """Test complete aircraft optimization workflow."""
    from src.models.aerospace_systems import AircraftOptimizationSystem
    from src.optimization.algorithms import GeneticAlgorithm
    
    # Setup system
    aircraft_system = AircraftOptimizationSystem()
    
    parameter_bounds = {
        'wingspan': (30.0, 60.0),
        'wing_area': (150.0, 350.0),
        'aspect_ratio': (7.0, 12.0)
    }
    
    optimizer = GeneticAlgorithm(parameter_bounds, population_size=20)
    
    def objective(params):
        result = aircraft_system.evaluate_design(params, 'commercial')
        return result['simulation_result']
    
    # Run optimization
    result = optimizer.optimize(objective, max_evaluations=50)
    
    # Verify results
    assert result.convergence_achieved or result.total_evaluations >= 50
    assert 'lift_to_drag_ratio' in result.best_objectives
    assert result.best_objectives['lift_to_drag_ratio'] > 0
```

### Test Data and Fixtures

Use pytest fixtures for common test data:

```python
@pytest.fixture
def sample_aircraft_parameters():
    """Sample aircraft parameters for testing."""
    return {
        'wingspan': 45.0,
        'wing_area': 250.0,
        'aspect_ratio': 9.0,
        'sweep_angle': 25.0,
        'cruise_mach': 0.78
    }

@pytest.fixture
def optimization_result():
    """Sample optimization result for testing."""
    from src.optimization.base import OptimizationResult
    return OptimizationResult(
        best_parameters={'x': 1.0, 'y': 2.0},
        best_objectives={'objective': 0.5},
        optimization_history=[],
        total_evaluations=100,
        convergence_achieved=True,
        total_time=10.5,
        algorithm_name="TestAlgorithm",
        metadata={}
    )
```

### Performance Tests

Include performance benchmarks:

```python
import time
import pytest

class TestPerformance:
    """Performance tests for optimization algorithms."""
    
    @pytest.mark.slow
    def test_genetic_algorithm_performance(self):
        """Test GA performance on standard benchmarks."""
        from src.optimization.algorithms import GeneticAlgorithm
        
        bounds = {f'x{i}': (-5.0, 5.0) for i in range(10)}
        optimizer = GeneticAlgorithm(bounds, population_size=50)
        
        def rosenbrock(params):
            x = [params[f'x{i}'] for i in range(10)]
            result = sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
                        for i in range(9))
            return {'objective': result}
        
        start_time = time.time()
        result = optimizer.optimize(rosenbrock, max_evaluations=1000)
        elapsed_time = time.time() - start_time
        
        # Performance assertions
        assert elapsed_time < 30.0  # Should complete within 30 seconds
        assert result.best_objectives['objective'] < 100.0
        
    def test_evaluation_caching(self):
        """Test that evaluation caching improves performance."""
        # Implementation of caching performance test
        pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run specific test file
pytest tests/test_optimization_algorithms.py

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto  # Requires pytest-xdist
```

---

## Documentation

### Documentation Types

1. **API Documentation**: Docstrings in code (auto-generated)
2. **User Guide**: Step-by-step tutorials and examples
3. **Technical Documentation**: Mathematical formulations and algorithms
4. **Contributing Guide**: This document
5. **README**: Project overview and quick start

### Writing Documentation

#### Docstring Standards

```python
class OptimizationResult:
    """
    Container for optimization results.
    
    This class stores the results of an optimization run, including
    the best parameters found, objective values, and metadata about
    the optimization process.
    
    Attributes:
        best_parameters (Dict[str, float]): Best parameter values found
        best_objectives (Dict[str, float]): Corresponding objective values
        optimization_history (List[Dict]): History of optimization progress
        total_evaluations (int): Total number of function evaluations
        convergence_achieved (bool): Whether optimization converged
        total_time (float): Total optimization time in seconds
        algorithm_name (str): Name of optimization algorithm used
        metadata (Dict[str, Any]): Additional algorithm-specific data
    
    Example:
        >>> result = OptimizationResult(
        ...     best_parameters={'x': 1.0, 'y': 2.0},
        ...     best_objectives={'f': 0.5},
        ...     optimization_history=[],
        ...     total_evaluations=100,
        ...     convergence_achieved=True,
        ...     total_time=10.5,
        ...     algorithm_name="GeneticAlgorithm",
        ...     metadata={}
        ... )
        >>> print(f"Best objective: {result.best_objectives['f']}")
        Best objective: 0.5
    """
```

#### Example Documentation

Create comprehensive examples:

```python
"""
Aircraft Optimization Example
=============================

This example demonstrates how to optimize an aircraft design for
maximum lift-to-drag ratio using the genetic algorithm.

The optimization considers multiple design parameters including
wingspan, wing area, aspect ratio, and operational conditions.
"""

import numpy as np
from src.models.aerospace_systems import AircraftOptimizationSystem
from src.optimization.algorithms import GeneticAlgorithm

# Setup optimization problem
print("Setting up aircraft optimization problem...")

aircraft_system = AircraftOptimizationSystem()
parameter_bounds = {
    'wingspan': (30.0, 60.0),        # Wing span in meters
    'wing_area': (150.0, 350.0),     # Wing area in square meters
    'aspect_ratio': (7.0, 12.0),     # Wing aspect ratio
    'sweep_angle': (15.0, 35.0),     # Wing sweep angle in degrees
    'cruise_mach': (0.65, 0.85)      # Cruise Mach number
}

# Define objective function
def maximize_lift_to_drag(parameters):
    """
    Objective function to maximize lift-to-drag ratio.
    
    Args:
        parameters: Dictionary of aircraft design parameters
        
    Returns:
        Dictionary with 'lift_to_drag_ratio' key (negative for minimization)
    """
    result = aircraft_system.evaluate_design(parameters, 'commercial')
    sim_result = result['simulation_result']
    
    # Return negative value since we minimize
    return {'lift_to_drag_ratio': -sim_result.objectives['lift_to_drag_ratio']}

# Rest of example...
```

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme nbsphinx

# Build HTML documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

---

## Submitting Changes

### Pull Request Process

#### 1. Prepare Your Pull Request

```bash
# Ensure your branch is up to date
git checkout main
git pull upstream main
git checkout your-feature-branch
git rebase main

# Run all checks
black src/ tests/ examples/
flake8 src/ tests/ examples/
python -m pytest tests/

# Push changes
git push origin your-feature-branch
```

#### 2. Create Pull Request

**Pull Request Template:**

```markdown
## Description
Brief description of the changes and why they're needed.

Fixes #(issue_number)

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested the changes on realistic optimization problems

## Documentation
- [ ] I have updated the documentation accordingly
- [ ] I have added docstrings to new functions/classes
- [ ] I have updated examples if needed

## Checklist
- [ ] My code follows the code style of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published
```

#### 3. Code Review Process

- **Automated Checks**: CI/CD will run tests and code quality checks
- **Maintainer Review**: Core maintainers will review your code
- **Feedback**: Address any feedback promptly and professionally
- **Approval**: Once approved, maintainers will merge your PR

### Commit Message Guidelines

Use clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add Bayesian optimization algorithm with GP surrogate model"
git commit -m "Fix convergence criteria bug in genetic algorithm"
git commit -m "Update aircraft model to include compressibility effects"
git commit -m "Add comprehensive tests for uncertainty quantification module"

# Poor commit messages (avoid these)
git commit -m "Fixed bug"
git commit -m "Updated code"
git commit -m "Changes"
```

### Release Process

For maintainers releasing new versions:

1. **Version Bump**: Update version in `setup.py` and `__init__.py`
2. **Changelog**: Update `CHANGELOG.md` with new features and fixes
3. **Documentation**: Ensure all documentation is up to date
4. **Testing**: Run comprehensive test suite including performance tests
5. **Tag Release**: Create git tag and GitHub release
6. **Distribution**: Upload to PyPI (when ready)

---

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- **Be respectful** in all interactions
- **Be collaborative** and help others learn
- **Be constructive** when providing feedback
- **Be patient** with newcomers and questions

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Pull Requests**: Code reviews and technical discussions
- **Email**: For private concerns or security issues

### Recognition

We value all contributions and recognize contributors in:

- **CONTRIBUTORS.md**: List of all project contributors
- **Release Notes**: Major contributors acknowledged in releases
- **Documentation**: Authors credited in relevant sections

### Getting Help

If you need help:

1. **Check Documentation**: Start with README, User Guide, and API docs
2. **Search Issues**: See if someone else had the same question
3. **Ask Questions**: Open a GitHub Discussion or issue
4. **Join Community**: Participate in project discussions

### Mentorship

New contributors can:

- **Start with Issues**: Look for `good first issue` labels
- **Ask for Guidance**: Maintainers are happy to help new contributors
- **Pair Programming**: Senior contributors may be available for pair programming sessions

---

## Advanced Contributing Topics

### Adding New Optimization Algorithms

To add a new optimization algorithm:

1. **Inherit from BaseOptimizer**:
```python
from src.optimization.base import BaseOptimizer

class YourNewAlgorithm(BaseOptimizer):
    def __init__(self, parameter_bounds, **kwargs):
        super().__init__(parameter_bounds)
        # Your algorithm-specific parameters
    
    def optimize(self, objective_function, max_evaluations, **kwargs):
        # Your algorithm implementation
        pass
```

2. **Add comprehensive tests**
3. **Update documentation and examples**
4. **Add to algorithm factory/registry**

### Adding New Fidelity Models

To add new fidelity models:

1. **Implement BaseSimulation interface**
2. **Define computational cost and accuracy characteristics**
3. **Add validation against known solutions**
4. **Update multi-fidelity correlation models**

### Performance Optimization

When optimizing performance:

1. **Profile First**: Use `cProfile` to identify bottlenecks
2. **Vectorize**: Use NumPy operations where possible
3. **Parallel Processing**: Consider `multiprocessing` or `joblib`
4. **Caching**: Implement intelligent caching strategies
5. **Memory Management**: Monitor memory usage for large problems

### Extending to New Domains

To extend the framework to new application domains:

1. **Define Domain Models**: Create new simulation models
2. **Domain-Specific Constraints**: Implement relevant constraint handling
3. **Validation Data**: Gather benchmark problems and validation cases
4. **Documentation**: Create domain-specific tutorials and examples

---

Thank you for contributing to the Adaptive Multi-Fidelity Simulation-Based Optimization framework! Your contributions help advance the state of the art in computational optimization and benefit the entire aerospace engineering community.

For questions about contributing, please open a GitHub Discussion or contact the maintainers directly.