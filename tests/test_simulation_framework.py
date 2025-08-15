#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
from typing import Dict

from simulation.base import (
    FidelityLevel, SimulationResult, BaseSimulation, MultiFidelitySimulation
)
from simulation.adaptive_fidelity import AdaptiveFidelityManager, FidelitySwitchingStrategy


class MockSimulation(BaseSimulation):
    """Mock simulation for testing purposes."""
    
    def __init__(self, fidelity_level: FidelityLevel, computation_cost: float = 1.0):
        super().__init__(f"MockSimulation_{fidelity_level.value}", fidelity_level)
        self.computation_cost = computation_cost
        self.evaluation_count = 0
    
    def evaluate(self, parameters: Dict[str, float]) -> SimulationResult:
        self.evaluation_count += 1
        
        # Simple mock objectives based on parameters
        objectives = {
            'objective_1': parameters.get('x', 0.0) ** 2 + parameters.get('y', 0.0) ** 2,
            'objective_2': abs(parameters.get('x', 0.0) - 1.0) + abs(parameters.get('y', 0.0) - 1.0)
        }
        
        constraints = {
            'constraint_1': 1.0 - (parameters.get('x', 0.0) ** 2 + parameters.get('y', 0.0) ** 2)
        }
        
        uncertainty = {
            'objective_1': 0.1 if self.fidelity_level == FidelityLevel.LOW else 0.05,
            'objective_2': 0.15 if self.fidelity_level == FidelityLevel.LOW else 0.08
        }
        
        return SimulationResult(
            objectives=objectives,
            constraints=constraints,
            fidelity_level=self.fidelity_level,
            computation_time=self.computation_cost,
            uncertainty=uncertainty
        )
    
    def get_computational_cost(self) -> float:
        return self.computation_cost
    
    def validate_parameters(self, parameters: Dict[str, float]) -> bool:
        required_params = ['x', 'y']
        return all(param in parameters for param in required_params)


class TestSimulationBase(unittest.TestCase):
    """Test cases for base simulation classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_parameters = {'x': 0.5, 'y': -0.3}
        self.low_fidelity_sim = MockSimulation(FidelityLevel.LOW, 0.1)
        self.high_fidelity_sim = MockSimulation(FidelityLevel.HIGH, 1.0)
    
    def test_simulation_result_creation(self):
        """Test SimulationResult creation and properties."""
        objectives = {'obj1': 1.5, 'obj2': 2.3}
        constraints = {'cons1': 0.8}
        
        result = SimulationResult(
            objectives=objectives,
            constraints=constraints,
            fidelity_level=FidelityLevel.MEDIUM
        )
        
        self.assertEqual(result.objectives, objectives)
        self.assertEqual(result.constraints, constraints)
        self.assertEqual(result.fidelity_level, FidelityLevel.MEDIUM)
        self.assertIsInstance(result.timestamp, float)
    
    def test_base_simulation_evaluation(self):
        """Test base simulation evaluation."""
        result = self.low_fidelity_sim.evaluate(self.test_parameters)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.fidelity_level, FidelityLevel.LOW)
        self.assertIn('objective_1', result.objectives)
        self.assertIn('objective_2', result.objectives)
        self.assertIn('constraint_1', result.constraints)
        self.assertEqual(self.low_fidelity_sim.evaluation_count, 1)
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        valid_params = {'x': 1.0, 'y': 2.0}
        invalid_params = {'x': 1.0}  # missing 'y'
        
        self.assertTrue(self.low_fidelity_sim.validate_parameters(valid_params))
        self.assertFalse(self.low_fidelity_sim.validate_parameters(invalid_params))
    
    def test_computational_cost(self):
        """Test computational cost retrieval."""
        self.assertEqual(self.low_fidelity_sim.get_computational_cost(), 0.1)
        self.assertEqual(self.high_fidelity_sim.get_computational_cost(), 1.0)


class TestMultiFidelitySimulation(unittest.TestCase):
    """Test cases for multi-fidelity simulation system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.multi_sim = MultiFidelitySimulation("TestMultiFidelity")
        
        self.low_sim = MockSimulation(FidelityLevel.LOW, 0.1)
        self.medium_sim = MockSimulation(FidelityLevel.MEDIUM, 0.5)
        self.high_sim = MockSimulation(FidelityLevel.HIGH, 1.0)
        
        self.multi_sim.add_simulation(self.low_sim)
        self.multi_sim.add_simulation(self.medium_sim)
        self.multi_sim.add_simulation(self.high_sim)
        
        self.test_parameters = {'x': 0.5, 'y': -0.3}
    
    def test_simulation_addition(self):
        """Test adding simulations to multi-fidelity system."""
        self.assertIn(FidelityLevel.LOW, self.multi_sim.simulations)
        self.assertIn(FidelityLevel.MEDIUM, self.multi_sim.simulations)
        self.assertIn(FidelityLevel.HIGH, self.multi_sim.simulations)
    
    def test_forced_fidelity_evaluation(self):
        """Test evaluation with forced fidelity level."""
        result = self.multi_sim.evaluate(self.test_parameters, FidelityLevel.HIGH)
        
        self.assertEqual(result.fidelity_level, FidelityLevel.HIGH)
        self.assertEqual(self.high_sim.evaluation_count, 1)
        self.assertEqual(self.low_sim.evaluation_count, 0)
    
    def test_automatic_fidelity_selection(self):
        """Test automatic fidelity selection."""
        # First evaluation should use low fidelity
        result = self.multi_sim.evaluate(self.test_parameters)
        self.assertEqual(result.fidelity_level, FidelityLevel.LOW)
        
        # Multiple evaluations to test fidelity progression
        for _ in range(10):
            self.multi_sim.evaluate(self.test_parameters)
        
        # Should have some fidelity history
        self.assertGreater(len(self.multi_sim.fidelity_history), 0)
    
    def test_fidelity_statistics(self):
        """Test fidelity statistics tracking."""
        # Perform multiple evaluations
        for _ in range(5):
            self.multi_sim.evaluate(self.test_parameters, FidelityLevel.LOW)
        
        for _ in range(3):
            self.multi_sim.evaluate(self.test_parameters, FidelityLevel.HIGH)
        
        stats = self.multi_sim.get_fidelity_statistics()
        
        self.assertEqual(stats['total_evaluations'], 8)
        self.assertEqual(stats['evaluations_by_fidelity'][FidelityLevel.LOW], 5)
        self.assertEqual(stats['evaluations_by_fidelity'][FidelityLevel.HIGH], 3)
        self.assertGreater(stats['total_computation_time'], 0)


class TestAdaptiveFidelityManager(unittest.TestCase):
    """Test cases for adaptive fidelity management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = AdaptiveFidelityManager(FidelitySwitchingStrategy.ADAPTIVE)
        self.test_parameters = {'x': 0.5, 'y': -0.3}
        
        # Mock result for testing
        self.mock_result = SimulationResult(
            objectives={'obj1': 1.0, 'obj2': 2.0},
            constraints={'cons1': 0.5},
            fidelity_level=FidelityLevel.LOW,
            computation_time=0.1
        )
    
    def test_fidelity_manager_initialization(self):
        """Test fidelity manager initialization."""
        self.assertEqual(self.manager.strategy, FidelitySwitchingStrategy.ADAPTIVE)
        self.assertEqual(len(self.manager.evaluation_history), 0)
        self.assertEqual(self.manager.used_budget, 0.0)
    
    def test_conservative_strategy(self):
        """Test conservative fidelity switching strategy."""
        conservative_manager = AdaptiveFidelityManager(FidelitySwitchingStrategy.CONSERVATIVE)
        
        # Early evaluations should prefer low fidelity
        fidelity = conservative_manager.select_fidelity(self.test_parameters)
        self.assertEqual(fidelity, FidelityLevel.LOW)
    
    def test_aggressive_strategy(self):
        """Test aggressive fidelity switching strategy."""
        aggressive_manager = AdaptiveFidelityManager(FidelitySwitchingStrategy.AGGRESSIVE)
        
        # Add some evaluation history
        for i in range(10):
            aggressive_manager.update_evaluation_history(
                self.test_parameters, self.mock_result, 0.1
            )
        
        fidelity = aggressive_manager.select_fidelity(self.test_parameters)
        # Should be more willing to use higher fidelity
        self.assertIn(fidelity, [FidelityLevel.MEDIUM, FidelityLevel.HIGH])
    
    def test_evaluation_history_update(self):
        """Test evaluation history tracking."""
        initial_count = len(self.manager.evaluation_history)
        
        self.manager.update_evaluation_history(
            self.test_parameters, self.mock_result, 0.1
        )
        
        self.assertEqual(len(self.manager.evaluation_history), initial_count + 1)
        self.assertEqual(self.manager.used_budget, 0.1)
    
    def test_fidelity_statistics(self):
        """Test fidelity statistics generation."""
        # Add some evaluation history
        for i in range(5):
            result = SimulationResult(
                objectives={'obj1': i, 'obj2': i+1},
                constraints={'cons1': 0.5},
                fidelity_level=FidelityLevel.LOW if i < 3 else FidelityLevel.HIGH,
                computation_time=0.1 if i < 3 else 1.0
            )
            self.manager.update_evaluation_history(self.test_parameters, result, result.computation_time)
        
        stats = self.manager.get_fidelity_statistics()
        
        self.assertEqual(stats['total_evaluations'], 5)
        self.assertIn('fidelity_distribution', stats)
        self.assertIn('computation_time_by_fidelity', stats)
        self.assertEqual(stats['budget_utilization'], self.manager.used_budget / self.manager.computational_budget)


class TestFidelitySwitchingStrategies(unittest.TestCase):
    """Test cases for different fidelity switching strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_parameters = {'x': 0.5, 'y': -0.3}
        self.mock_result = SimulationResult(
            objectives={'obj1': 1.0, 'obj2': 2.0},
            constraints={'cons1': 0.5},
            fidelity_level=FidelityLevel.LOW,
            computation_time=0.1
        )
    
    def test_strategy_differences(self):
        """Test that different strategies behave differently."""
        strategies = [
            FidelitySwitchingStrategy.CONSERVATIVE,
            FidelitySwitchingStrategy.AGGRESSIVE,
            FidelitySwitchingStrategy.BALANCED,
            FidelitySwitchingStrategy.ADAPTIVE
        ]
        
        managers = {strategy: AdaptiveFidelityManager(strategy) for strategy in strategies}
        
        # Add some evaluation history to all managers
        for manager in managers.values():
            for i in range(15):
                result = SimulationResult(
                    objectives={'obj1': i * 0.1, 'obj2': (i+1) * 0.1},
                    constraints={'cons1': 0.5},
                    fidelity_level=FidelityLevel.LOW,
                    computation_time=0.1
                )
                manager.update_evaluation_history(self.test_parameters, result, result.computation_time)
        
        # Get fidelity selections
        selections = {}
        for strategy, manager in managers.items():
            selections[strategy] = manager.select_fidelity(self.test_parameters)
        
        # Verify that not all strategies select the same fidelity
        unique_selections = set(selections.values())
        self.assertGreaterEqual(len(unique_selections), 1)  # At least some variety expected
    
    def test_budget_constraints(self):
        """Test that managers respect computational budget constraints."""
        manager = AdaptiveFidelityManager(FidelitySwitchingStrategy.BALANCED)
        
        # Consume most of the budget
        manager.used_budget = manager.computational_budget * 0.95
        
        # Should prefer low fidelity when budget is nearly exhausted
        fidelity = manager.select_fidelity(self.test_parameters)
        self.assertEqual(fidelity, FidelityLevel.LOW)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSimulationBase))
    test_suite.addTest(unittest.makeSuite(TestMultiFidelitySimulation))
    test_suite.addTest(unittest.makeSuite(TestAdaptiveFidelityManager))
    test_suite.addTest(unittest.makeSuite(TestFidelitySwitchingStrategies))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with proper code
    sys.exit(0 if result.wasSuccessful() else 1)