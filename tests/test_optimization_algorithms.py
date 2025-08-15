#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
from typing import Dict, List

from optimization.algorithms import (
    GeneticAlgorithm, ParticleSwarmOptimization, BayesianOptimization, 
    MultiObjectiveOptimizer, OptimizationResult
)
from simulation.base import SimulationResult, FidelityLevel


class MockObjectiveFunction:
    """Mock objective function for testing optimization algorithms."""
    
    def __init__(self, problem_type: str = "sphere"):
        self.problem_type = problem_type
        self.evaluation_count = 0
    
    def __call__(self, parameters: Dict[str, float]) -> SimulationResult:
        self.evaluation_count += 1
        x = parameters.get('x', 0.0)
        y = parameters.get('y', 0.0)
        
        if self.problem_type == "sphere":
            # Simple sphere function (minimum at origin)
            obj1 = x**2 + y**2
            obj2 = (x-1)**2 + (y-1)**2
        elif self.problem_type == "rosenbrock":
            # Rosenbrock function
            obj1 = 100 * (y - x**2)**2 + (1 - x)**2
            obj2 = obj1 * 0.5  # Secondary objective
        elif self.problem_type == "rastrigin":
            # Rastrigin function (multimodal)
            A = 10
            obj1 = A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))
            obj2 = obj1 * 0.3
        else:
            obj1 = x**2 + y**2
            obj2 = (x-1)**2 + (y-1)**2
        
        objectives = {'objective_1': obj1, 'objective_2': obj2}
        constraints = {'constraint_1': 1.0 - (x**2 + y**2)}  # Circle constraint
        
        return SimulationResult(
            objectives=objectives,
            constraints=constraints,
            fidelity_level=FidelityLevel.LOW,
            computation_time=0.001
        )


class TestGeneticAlgorithm(unittest.TestCase):
    """Test cases for Genetic Algorithm implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameter_bounds = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        }
        self.ga = GeneticAlgorithm(self.parameter_bounds, population_size=20)
        self.objective_function = MockObjectiveFunction("sphere")
    
    def test_ga_initialization(self):
        """Test GA initialization."""
        self.assertEqual(self.ga.name, "GeneticAlgorithm")
        self.assertEqual(self.ga.population_size, 20)
        self.assertEqual(len(self.ga.parameter_bounds), 2)
    
    def test_population_initialization(self):
        """Test population initialization."""
        population = self.ga._initialize_population()
        
        self.assertEqual(len(population), self.ga.population_size)
        
        for individual in population:
            self.assertIn('x', individual)
            self.assertIn('y', individual)
            
            # Check bounds
            self.assertGreaterEqual(individual['x'], -5.0)
            self.assertLessEqual(individual['x'], 5.0)
            self.assertGreaterEqual(individual['y'], -5.0)
            self.assertLessEqual(individual['y'], 5.0)
    
    def test_crossover(self):
        """Test crossover operation."""
        parent1 = {'x': 1.0, 'y': 2.0}
        parent2 = {'x': 3.0, 'y': 4.0}
        
        child1, child2 = self.ga._crossover(parent1, parent2)
        
        self.assertIn('x', child1)
        self.assertIn('y', child1)
        self.assertIn('x', child2)
        self.assertIn('y', child2)
    
    def test_mutation(self):
        """Test mutation operation."""
        individual = {'x': 1.0, 'y': 2.0}
        mutated = self.ga._mutate(individual)
        
        self.assertIn('x', mutated)
        self.assertIn('y', mutated)
        # Values might be the same if no mutation occurred
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        population = [
            {'x': 0.0, 'y': 0.0},  # Good fitness
            {'x': 5.0, 'y': 5.0},  # Poor fitness
            {'x': 1.0, 'y': 1.0}   # Medium fitness
        ]
        fitness_scores = [1.0, 50.0, 2.0]  # Higher is better in our implementation
        
        selected = self.ga._tournament_selection(population, fitness_scores)
        self.assertIn('x', selected)
        self.assertIn('y', selected)
    
    def test_ga_optimization(self):
        """Test complete GA optimization run."""
        result = self.ga.optimize(self.objective_function, max_evaluations=50)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIn('objective_1', result.best_objectives)
        self.assertIn('objective_2', result.best_objectives)
        self.assertLessEqual(result.total_evaluations, 50)
        self.assertGreater(result.total_time, 0.0)
        
        # Check that optimization history exists
        self.assertGreater(len(result.optimization_history), 0)


class TestParticleSwarmOptimization(unittest.TestCase):
    """Test cases for Particle Swarm Optimization implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameter_bounds = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        }
        self.pso = ParticleSwarmOptimization(self.parameter_bounds, swarm_size=15)
        self.objective_function = MockObjectiveFunction("sphere")
    
    def test_pso_initialization(self):
        """Test PSO initialization."""
        self.assertEqual(self.pso.name, "ParticleSwarmOptimization")
        self.assertEqual(self.pso.swarm_size, 15)
        self.assertGreater(self.pso.w, 0)  # Inertia weight
        self.assertGreater(self.pso.c1, 0)  # Cognitive parameter
        self.assertGreater(self.pso.c2, 0)  # Social parameter
    
    def test_swarm_initialization(self):
        """Test swarm initialization."""
        swarm = self.pso._initialize_swarm()
        velocities = self.pso._initialize_velocities()
        
        self.assertEqual(len(swarm), self.pso.swarm_size)
        self.assertEqual(len(velocities), self.pso.swarm_size)
        
        for particle in swarm:
            self.assertIn('x', particle)
            self.assertIn('y', particle)
        
        for velocity in velocities:
            self.assertIn('x', velocity)
            self.assertIn('y', velocity)
    
    def test_pso_optimization(self):
        """Test complete PSO optimization run."""
        result = self.pso.optimize(self.objective_function, max_evaluations=45)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIn('objective_1', result.best_objectives)
        self.assertIn('objective_2', result.best_objectives)
        self.assertLessEqual(result.total_evaluations, 45)
        self.assertGreater(result.total_time, 0.0)
        
        # PSO should have particle information in history
        self.assertGreater(len(result.optimization_history), 0)
        if result.optimization_history:
            self.assertIn('particle', result.optimization_history[0])


class TestBayesianOptimization(unittest.TestCase):
    """Test cases for Bayesian Optimization implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameter_bounds = {
            'x': (-2.0, 2.0),
            'y': (-2.0, 2.0)
        }
        self.bo = BayesianOptimization(self.parameter_bounds)
        self.objective_function = MockObjectiveFunction("sphere")
    
    def test_bo_initialization(self):
        """Test Bayesian Optimization initialization."""
        self.assertEqual(self.bo.name, "BayesianOptimization")
        self.assertEqual(self.bo.acquisition_function, 'ei')
        self.assertIsNone(self.bo.gp)  # GP not initialized until optimization starts
    
    def test_parameter_conversion(self):
        """Test parameter array conversion."""
        params = {'x': 1.0, 'y': -0.5}
        param_array = self.bo._params_to_array(params)
        
        self.assertEqual(len(param_array), 2)
        
        converted_back = self.bo._array_to_params(param_array)
        self.assertAlmostEqual(converted_back['x'], params['x'])
        self.assertAlmostEqual(converted_back['y'], params['y'])
    
    def test_random_sampling(self):
        """Test random parameter sampling."""
        sample = self.bo._random_sample()
        
        self.assertIn('x', sample)
        self.assertIn('y', sample)
        self.assertGreaterEqual(sample['x'], -2.0)
        self.assertLessEqual(sample['x'], 2.0)
        self.assertGreaterEqual(sample['y'], -2.0)
        self.assertLessEqual(sample['y'], 2.0)
    
    def test_bo_optimization(self):
        """Test complete Bayesian optimization run."""
        result = self.bo.optimize(self.objective_function, max_evaluations=25)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIn('objective_1', result.best_objectives)
        self.assertIn('objective_2', result.best_objectives)
        self.assertLessEqual(result.total_evaluations, 25)
        self.assertGreater(result.total_time, 0.0)
        
        # BO should have acquisition information in history
        self.assertGreater(len(result.optimization_history), 0)
        # Later evaluations should have acquisition_type
        if len(result.optimization_history) > 10:
            self.assertIn('acquisition_type', result.optimization_history[-1])


class TestMultiObjectiveOptimizer(unittest.TestCase):
    """Test cases for Multi-Objective Optimization (NSGA-II)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameter_bounds = {
            'x': (-3.0, 3.0),
            'y': (-3.0, 3.0)
        }
        self.moo = MultiObjectiveOptimizer(self.parameter_bounds, population_size=30)
        self.objective_function = MockObjectiveFunction("sphere")
    
    def test_moo_initialization(self):
        """Test multi-objective optimizer initialization."""
        self.assertEqual(self.moo.name, "NSGA-II")
        self.assertEqual(self.moo.population_size, 30)
    
    def test_dominance_relation(self):
        """Test dominance relation for multi-objective optimization."""
        obj1 = [2.0, 3.0]  # Objective values for solution 1
        obj2 = [1.0, 4.0]  # Objective values for solution 2
        obj3 = [3.0, 2.0]  # Objective values for solution 3
        
        # obj1 dominates obj2 (better in first objective, worse in second)
        self.assertFalse(self.moo._dominates(obj1, obj2))
        self.assertFalse(self.moo._dominates(obj2, obj1))
        
        # Test clear dominance
        obj4 = [1.0, 1.0]  # Clearly dominates obj1
        self.assertTrue(self.moo._dominates(obj4, obj1))
        self.assertFalse(self.moo._dominates(obj1, obj4))
    
    def test_non_dominated_sorting(self):
        """Test fast non-dominated sorting."""
        objectives_list = [
            [1.0, 4.0],  # Solution 1
            [2.0, 3.0],  # Solution 2
            [3.0, 2.0],  # Solution 3
            [4.0, 1.0],  # Solution 4
            [2.5, 2.5]   # Solution 5
        ]
        
        fronts = self.moo._fast_non_dominated_sort(objectives_list)
        
        self.assertGreater(len(fronts), 0)
        self.assertGreater(len(fronts[0]), 0)  # First front should not be empty
    
    def test_crowding_distance(self):
        """Test crowding distance calculation."""
        objectives_list = [
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0]
        ]
        
        distances = self.moo._calculate_crowding_distance(objectives_list)
        
        self.assertEqual(len(distances), len(objectives_list))
        self.assertTrue(all(d >= 0 for d in distances))
    
    def test_moo_optimization(self):
        """Test complete multi-objective optimization run."""
        result = self.moo.optimize(self.objective_function, max_evaluations=60)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIn('objective_1', result.best_objectives)
        self.assertIn('objective_2', result.best_objectives)
        self.assertLessEqual(result.total_evaluations, 60)
        self.assertGreater(result.total_time, 0.0)
        
        # Multi-objective should have generation information
        self.assertGreater(len(result.optimization_history), 0)
        if result.optimization_history:
            self.assertIn('generation', result.optimization_history[0])


class TestOptimizationComparison(unittest.TestCase):
    """Test cases for comparing optimization algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameter_bounds = {
            'x': (-2.0, 2.0),
            'y': (-2.0, 2.0)
        }
        
        self.algorithms = {
            'GA': GeneticAlgorithm(self.parameter_bounds, population_size=20),
            'PSO': ParticleSwarmOptimization(self.parameter_bounds, swarm_size=20),
            'BO': BayesianOptimization(self.parameter_bounds)
        }
        
        self.objective_function = MockObjectiveFunction("sphere")
    
    def test_algorithm_convergence(self):
        """Test that all algorithms can converge on simple problem."""
        max_evaluations = 40
        results = {}
        
        for name, algorithm in self.algorithms.items():
            result = algorithm.optimize(self.objective_function, max_evaluations)
            results[name] = result
            
            # All algorithms should find reasonable solutions for sphere function
            best_obj1 = result.best_objectives['objective_1']
            self.assertLess(best_obj1, 10.0)  # Should be much better than random
        
        # Compare convergence speeds
        for name, result in results.items():
            self.assertGreater(len(result.optimization_history), 0)
            self.assertLessEqual(result.total_evaluations, max_evaluations)
    
    def test_algorithm_consistency(self):
        """Test algorithm consistency across multiple runs."""
        # Test GA consistency
        ga_results = []
        for _ in range(3):
            self.objective_function.evaluation_count = 0  # Reset counter
            result = self.algorithms['GA'].optimize(self.objective_function, 30)
            ga_results.append(result.best_objectives['objective_1'])
        
        # Results should be somewhat consistent (not wildly different)
        ga_std = np.std(ga_results)
        self.assertLess(ga_std, 5.0)  # Reasonable variation for stochastic algorithm


class TestOptimizationEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for optimization algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameter_bounds = {
            'x': (-1.0, 1.0),
            'y': (-1.0, 1.0)
        }
    
    def test_small_evaluation_budget(self):
        """Test optimization with very small evaluation budget."""
        ga = GeneticAlgorithm(self.parameter_bounds, population_size=10)
        objective_function = MockObjectiveFunction("sphere")
        
        result = ga.optimize(objective_function, max_evaluations=5)
        
        self.assertLessEqual(result.total_evaluations, 5)
        self.assertGreater(len(result.optimization_history), 0)
    
    def test_parameter_bounds_enforcement(self):
        """Test that parameter bounds are properly enforced."""
        ga = GeneticAlgorithm(self.parameter_bounds)
        
        # Test bounds enforcement
        test_params = {'x': 2.0, 'y': -2.0}  # Outside bounds
        bounded_params = ga._ensure_bounds(test_params)
        
        self.assertLessEqual(bounded_params['x'], 1.0)
        self.assertGreaterEqual(bounded_params['x'], -1.0)
        self.assertLessEqual(bounded_params['y'], 1.0)
        self.assertGreaterEqual(bounded_params['y'], -1.0)
    
    def test_empty_parameter_bounds(self):
        """Test handling of empty parameter bounds."""
        with self.assertRaises((ValueError, KeyError)):
            ga = GeneticAlgorithm({})
            objective_function = MockObjectiveFunction("sphere")
            ga.optimize(objective_function, 10)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestGeneticAlgorithm))
    test_suite.addTest(unittest.makeSuite(TestParticleSwarmOptimization))
    test_suite.addTest(unittest.makeSuite(TestBayesianOptimization))
    test_suite.addTest(unittest.makeSuite(TestMultiObjectiveOptimizer))
    test_suite.addTest(unittest.makeSuite(TestOptimizationComparison))
    test_suite.addTest(unittest.makeSuite(TestOptimizationEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with proper code
    sys.exit(0 if result.wasSuccessful() else 1)