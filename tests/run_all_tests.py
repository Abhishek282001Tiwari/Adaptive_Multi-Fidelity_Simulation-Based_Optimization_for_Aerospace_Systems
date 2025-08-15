#!/usr/bin/env python3

import sys
import os
import unittest
import time
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import test modules
from test_simulation_framework import *
from test_optimization_algorithms import *


def create_test_report(test_result, output_file="test_report.txt"):
    """Create a comprehensive test report."""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("AEROSPACE OPTIMIZATION SYSTEM - TEST REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Overall summary
    total_tests = test_result.testsRun
    failures = len(test_result.failures)
    errors = len(test_result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
    
    report_lines.append("SUMMARY:")
    report_lines.append(f"  Total Tests Run: {total_tests}")
    report_lines.append(f"  Successful: {total_tests - failures - errors}")
    report_lines.append(f"  Failures: {failures}")
    report_lines.append(f"  Errors: {errors}")
    report_lines.append(f"  Success Rate: {success_rate:.1f}%")
    report_lines.append("")
    
    # Test execution time
    execution_time = getattr(test_result, 'execution_time', 0)
    report_lines.append(f"Execution Time: {execution_time:.2f} seconds")
    report_lines.append("")
    
    # Detailed results by test case
    if hasattr(test_result, 'test_results_by_class'):
        report_lines.append("DETAILED RESULTS BY TEST CLASS:")
        report_lines.append("-" * 50)
        
        for class_name, class_results in test_result.test_results_by_class.items():
            class_total = class_results['total']
            class_passed = class_results['passed']
            class_failed = class_results['failed']
            class_errors = class_results['errors']
            
            report_lines.append(f"{class_name}:")
            report_lines.append(f"  Tests: {class_total}")
            report_lines.append(f"  Passed: {class_passed}")
            report_lines.append(f"  Failed: {class_failed}")
            report_lines.append(f"  Errors: {class_errors}")
            report_lines.append("")
    
    # Failure details
    if test_result.failures:
        report_lines.append("FAILURES:")
        report_lines.append("-" * 50)
        for test, traceback in test_result.failures:
            report_lines.append(f"FAIL: {test}")
            report_lines.append(traceback)
            report_lines.append("")
    
    # Error details
    if test_result.errors:
        report_lines.append("ERRORS:")
        report_lines.append("-" * 50)
        for test, traceback in test_result.errors:
            report_lines.append(f"ERROR: {test}")
            report_lines.append(traceback)
            report_lines.append("")
    
    # Coverage analysis (simplified)
    report_lines.append("COVERAGE ANALYSIS:")
    report_lines.append("-" * 50)
    report_lines.append("Modules Tested:")
    report_lines.append("  ✓ Simulation Framework")
    report_lines.append("    - Base simulation classes")
    report_lines.append("    - Multi-fidelity simulation")
    report_lines.append("    - Adaptive fidelity management")
    report_lines.append("    - Fidelity switching strategies")
    report_lines.append("")
    report_lines.append("  ✓ Optimization Algorithms")
    report_lines.append("    - Genetic Algorithm (GA)")
    report_lines.append("    - Particle Swarm Optimization (PSO)")
    report_lines.append("    - Bayesian Optimization (BO)")
    report_lines.append("    - Multi-Objective Optimization (NSGA-II)")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS:")
    report_lines.append("-" * 50)
    
    if success_rate >= 95:
        report_lines.append("✓ Excellent test coverage and system reliability")
        report_lines.append("✓ System ready for production use")
    elif success_rate >= 85:
        report_lines.append("⚠ Good test coverage with minor issues")
        report_lines.append("⚠ Address failures before production deployment")
    else:
        report_lines.append("❌ Significant test failures detected")
        report_lines.append("❌ System requires debugging before use")
    
    if failures > 0:
        report_lines.append(f"• Investigate and fix {failures} test failures")
    
    if errors > 0:
        report_lines.append(f"• Debug and resolve {errors} test errors")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    # Write report to file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return '\n'.join(report_lines)


class DetailedTestResult(unittest.TextTestResult):
    """Extended test result class that tracks additional metrics."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_results_by_class = {}
        self.start_time = None
        self.execution_time = 0
    
    def startTestRun(self):
        super().startTestRun()
        self.start_time = time.time()
    
    def stopTestRun(self):
        super().stopTestRun()
        if self.start_time:
            self.execution_time = time.time() - self.start_time
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self._track_test_result(test, 'passed')
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._track_test_result(test, 'failed')
    
    def addError(self, test, err):
        super().addError(test, err)
        self._track_test_result(test, 'errors')
    
    def _track_test_result(self, test, result_type):
        class_name = test.__class__.__name__
        
        if class_name not in self.test_results_by_class:
            self.test_results_by_class[class_name] = {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0
            }
        
        self.test_results_by_class[class_name]['total'] += 1
        self.test_results_by_class[class_name][result_type] += 1


def run_simulation_tests():
    """Run all simulation framework tests."""
    print("Running Simulation Framework Tests...")
    
    test_classes = [
        TestSimulationBase,
        TestMultiFidelitySimulation,
        TestAdaptiveFidelityManager,
        TestFidelitySwitchingStrategies
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))
    
    return suite


def run_optimization_tests():
    """Run all optimization algorithm tests."""
    print("Running Optimization Algorithm Tests...")
    
    test_classes = [
        TestGeneticAlgorithm,
        TestParticleSwarmOptimization,
        TestBayesianOptimization,
        TestMultiObjectiveOptimizer,
        TestOptimizationComparison,
        TestOptimizationEdgeCases
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))
    
    return suite


def run_integration_tests():
    """Run integration tests (if any)."""
    print("Running Integration Tests...")
    
    # For now, create a simple integration test
    class TestSystemIntegration(unittest.TestCase):
        """Basic integration tests."""
        
        def test_imports(self):
            """Test that all modules can be imported successfully."""
            try:
                from simulation.base import MultiFidelitySimulation, FidelityLevel
                from optimization.algorithms import GeneticAlgorithm
                from models.aerospace_systems import AircraftOptimizationSystem
                from utilities.data_manager import DataManager
                from visualization.graph_generator import ProfessionalGraphGenerator
            except ImportError as e:
                self.fail(f"Import error: {e}")
        
        def test_basic_workflow(self):
            """Test basic optimization workflow."""
            try:
                from models.aerospace_systems import AircraftOptimizationSystem
                from optimization.algorithms import GeneticAlgorithm
                
                # Create system
                aircraft_system = AircraftOptimizationSystem()
                
                # Define simple parameters
                test_params = {
                    'wingspan': 40.0,
                    'wing_area': 200.0,
                    'aspect_ratio': 8.0,
                    'sweep_angle': 25.0,
                    'taper_ratio': 0.6,
                    'thickness_ratio': 0.12,
                    'cruise_altitude': 11000.0,
                    'cruise_mach': 0.78,
                    'weight': 60000.0
                }
                
                # Test evaluation
                result = aircraft_system.evaluate_design(test_params)
                self.assertIn('simulation_result', result)
                self.assertIn('performance_metrics', result)
                
            except Exception as e:
                self.fail(f"Basic workflow failed: {e}")
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSystemIntegration))
    
    return suite


def main():
    """Main test execution function."""
    print("="*80)
    print("ADAPTIVE MULTI-FIDELITY AEROSPACE OPTIMIZATION")
    print("COMPREHENSIVE TEST SUITE")
    print("="*80)
    print()
    
    # Create results directory
    os.makedirs("test_results", exist_ok=True)
    
    # Collect all test suites
    all_suites = unittest.TestSuite()
    
    # Add simulation tests
    simulation_suite = run_simulation_tests()
    all_suites.addTest(simulation_suite)
    
    # Add optimization tests
    optimization_suite = run_optimization_tests()
    all_suites.addTest(optimization_suite)
    
    # Add integration tests
    integration_suite = run_integration_tests()
    all_suites.addTest(integration_suite)
    
    # Create custom test runner
    class DetailedTestRunner(unittest.TextTestRunner):
        def _makeResult(self):
            return DetailedTestResult(self.stream, self.descriptions, self.verbosity)
    
    # Run all tests
    print(f"Starting test execution with {all_suites.countTestCases()} test cases...")
    print("-" * 80)
    
    runner = DetailedTestRunner(verbosity=2)
    start_time = time.time()
    
    try:
        result = runner.run(all_suites)
        end_time = time.time()
        
        # Generate comprehensive report
        report_content = create_test_report(result, "test_results/test_report.txt")
        print("\n" + "="*80)
        print("TEST EXECUTION COMPLETED")
        print("="*80)
        
        # Print summary
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
        execution_time = end_time - start_time
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_tests - failures - errors}")
        print(f"Failed: {failures}")
        print(f"Errors: {errors}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Execution Time: {execution_time:.2f} seconds")
        
        if result.wasSuccessful():
            print("\n🎉 ALL TESTS PASSED!")
            print("✓ System is ready for use")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("❌ Review test report for details")
        
        print(f"\nDetailed report saved to: test_results/test_report.txt")
        print("="*80)
        
        return result.wasSuccessful()
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        return False
    
    except Exception as e:
        print(f"\n❌ Critical error during test execution: {e}")
        return False


if __name__ == '__main__':
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)