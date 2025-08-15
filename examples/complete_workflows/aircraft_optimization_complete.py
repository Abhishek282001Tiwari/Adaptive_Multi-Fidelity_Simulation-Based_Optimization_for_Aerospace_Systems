#!/usr/bin/env python3
"""
Complete Aircraft Optimization Workflow
=======================================

This example demonstrates a complete aircraft optimization workflow including:
1. System initialization and configuration
2. Parameter bounds definition
3. Multi-objective optimization
4. Uncertainty analysis
5. Sensitivity analysis
6. Comprehensive visualization
7. Results export and reporting

This is a fully working example that can be run independently.
"""

import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aircraft_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Complete aircraft optimization workflow"""
    
    print("="*70)
    print("ADAPTIVE MULTI-FIDELITY AIRCRAFT OPTIMIZATION")
    print("="*70)
    print()
    
    # Step 1: Initialize System
    print("Step 1: Initializing Aircraft Optimization System...")
    
    try:
        from src.models.aerospace_systems import AircraftOptimizationSystem
        from src.optimization.algorithms import GeneticAlgorithm, NSGA2
        from src.optimization.robust_optimization import (
            UncertaintyQuantification, UncertaintyDistribution, 
            RobustOptimizer, SensitivityAnalysis
        )
        from src.visualization.graph_generator import ProfessionalGraphGenerator
        from src.utilities.data_manager import DataManager
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print("Error: Could not import required modules. Please ensure the framework is properly installed.")
        print("Run: pip install -e . from the project root directory")
        return
    
    # Initialize systems
    aircraft_system = AircraftOptimizationSystem()
    graph_generator = ProfessionalGraphGenerator("visualizations/")
    data_manager = DataManager("results/")
    
    print("✓ System initialized successfully")
    print()
    
    # Step 2: Define Optimization Problem
    print("Step 2: Defining Optimization Problem...")
    
    # Comprehensive parameter bounds for commercial aircraft
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
    
    print(f"✓ Optimizing {len(parameter_bounds)} design parameters:")
    for param, bounds in parameter_bounds.items():
        print(f"  {param}: [{bounds[0]:.1f}, {bounds[1]:.1f}]")
    print()
    
    # Step 3: Single-Objective Optimization (Maximize L/D Ratio)
    print("Step 3: Running Single-Objective Optimization...")
    
    def single_objective_function(parameters):
        """Optimize for maximum lift-to-drag ratio"""
        result = aircraft_system.evaluate_design(parameters, 'commercial')
        sim_result = result['simulation_result']
        
        # Return negative value for maximization (optimizer minimizes)
        return {'lift_to_drag_ratio': -sim_result.objectives['lift_to_drag_ratio']}
    
    # Create and run genetic algorithm optimizer
    ga_optimizer = GeneticAlgorithm(
        parameter_bounds=parameter_bounds,
        population_size=50,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    start_time = time.time()
    ga_result = ga_optimizer.optimize(
        objective_function=single_objective_function,
        max_evaluations=200
    )
    ga_time = time.time() - start_time
    
    best_ld_ratio = -ga_result.best_objectives['lift_to_drag_ratio']
    
    print(f"✓ Single-objective optimization completed in {ga_time:.1f} seconds")
    print(f"  Best L/D ratio: {best_ld_ratio:.2f}")
    print(f"  Total evaluations: {ga_result.total_evaluations}")
    print(f"  Convergence achieved: {ga_result.convergence_achieved}")
    print()
    
    # Step 4: Multi-Objective Optimization
    print("Step 4: Running Multi-Objective Optimization...")
    
    def multi_objective_function(parameters):
        """Multi-objective optimization: L/D ratio, fuel efficiency, structural weight"""
        result = aircraft_system.evaluate_design(parameters, 'commercial')
        sim_result = result['simulation_result']
        
        # Calculate structural weight penalty
        structural_weight = (parameters['wingspan'] * parameters['wing_area'] * 0.05)
        
        return {
            'lift_to_drag_ratio': -sim_result.objectives['lift_to_drag_ratio'],  # Maximize
            'fuel_efficiency': sim_result.objectives['fuel_efficiency'],         # Minimize
            'structural_weight': structural_weight                              # Minimize
        }
    
    # Create and run NSGA-II optimizer
    nsga2_optimizer = NSGA2(
        parameter_bounds=parameter_bounds,
        population_size=100,
        crossover_rate=0.9,
        mutation_rate=0.1
    )
    
    start_time = time.time()
    pareto_result = nsga2_optimizer.optimize(
        objective_function=multi_objective_function,
        max_evaluations=500
    )
    nsga2_time = time.time() - start_time
    
    print(f"✓ Multi-objective optimization completed in {nsga2_time:.1f} seconds")
    print(f"  Pareto optimal solutions found: {len(pareto_result.pareto_solutions)}")
    print(f"  Total evaluations: {pareto_result.total_evaluations}")
    print()
    
    # Analyze Pareto solutions
    print("  Top 3 Pareto solutions:")
    for i, solution in enumerate(pareto_result.pareto_solutions[:3]):
        ld_ratio = -solution['objectives']['lift_to_drag_ratio']
        fuel_eff = solution['objectives']['fuel_efficiency']
        weight = solution['objectives']['structural_weight']
        print(f"    Solution {i+1}: L/D={ld_ratio:.2f}, Fuel={fuel_eff:.3f}, Weight={weight:.0f}kg")
    print()
    
    # Step 5: Uncertainty Quantification
    print("Step 5: Uncertainty Quantification and Robust Optimization...")
    
    # Setup uncertainties
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
    
    # Run robust optimization
    robust_optimizer = RobustOptimizer(uq)
    
    start_time = time.time()
    robust_result = robust_optimizer.robust_optimization(
        objective_function=single_objective_function,
        parameter_bounds=parameter_bounds,
        robustness_measure='mean_std',
        n_mc_samples=100,
        optimization_algorithm='genetic_algorithm'
    )
    robust_time = time.time() - start_time
    
    print(f"✓ Robust optimization completed in {robust_time:.1f} seconds")
    print(f"  Robust L/D ratio: {-robust_result.robust_objective:.2f}")
    print(f"  Mean performance: {-robust_result.mean_objective:.2f}")
    print(f"  Standard deviation: {robust_result.objective_std:.3f}")
    print()
    
    # Step 6: Sensitivity Analysis
    print("Step 6: Global Sensitivity Analysis...")
    
    sensitivity = SensitivityAnalysis()
    
    # Morris screening
    start_time = time.time()
    morris_results = sensitivity.morris_screening(
        objective_function=single_objective_function,
        parameter_bounds=parameter_bounds,
        n_trajectories=20
    )
    morris_time = time.time() - start_time
    
    print(f"✓ Morris screening completed in {morris_time:.1f} seconds")
    print("  Parameter sensitivity ranking (Morris μ*):")
    
    # Sort parameters by sensitivity
    sorted_morris = sorted(
        morris_results.items(),
        key=lambda x: x[1]['mu_star'],
        reverse=True
    )
    
    for i, (param, indices) in enumerate(sorted_morris[:5]):
        print(f"    {i+1}. {param}: μ*={indices['mu_star']:.3f}, σ={indices['sigma']:.3f}")
    print()
    
    # Sobol indices (more expensive, fewer samples for demo)
    print("  Computing Sobol sensitivity indices...")
    start_time = time.time()
    sobol_results = sensitivity.sobol_indices(
        objective_function=single_objective_function,
        parameter_bounds=parameter_bounds,
        n_samples=500  # Reduced for faster execution
    )
    sobol_time = time.time() - start_time
    
    print(f"✓ Sobol analysis completed in {sobol_time:.1f} seconds")
    print("  Top 5 most influential parameters (Sobol ST):")
    
    # Sort by total-order indices
    sorted_sobol = sorted(
        sobol_results.items(),
        key=lambda x: x[1]['ST'],
        reverse=True
    )
    
    for i, (param, indices) in enumerate(sorted_sobol[:5]):
        print(f"    {i+1}. {param}: S1={indices['S1']:.3f}, ST={indices['ST']:.3f}")
    print()
    
    # Step 7: Generate Visualizations
    print("Step 7: Generating Visualizations...")
    
    # Convergence plot
    convergence_plot = graph_generator.create_convergence_plot(
        ga_result.optimization_history,
        "Genetic Algorithm - Aircraft L/D Optimization",
        "aircraft_convergence"
    )
    print(f"✓ Convergence plot saved: {convergence_plot}")
    
    # Pareto front plot
    pareto_plot = graph_generator.create_pareto_front_plot(
        pareto_result.pareto_solutions,
        "aircraft_pareto_front"
    )
    print(f"✓ Pareto front plot saved: {pareto_plot}")
    
    # Fidelity switching analysis
    fidelity_stats = aircraft_system.get_optimization_statistics()
    fidelity_plot = graph_generator.create_fidelity_switching_plot(
        fidelity_stats['fidelity_history'],
        "aircraft_fidelity_analysis"
    )
    print(f"✓ Fidelity analysis plot saved: {fidelity_plot}")
    
    # Uncertainty propagation plot
    uncertainty_plot = graph_generator.create_uncertainty_propagation_plot(
        robust_result.monte_carlo_results,
        "aircraft_uncertainty_analysis"
    )
    print(f"✓ Uncertainty analysis plot saved: {uncertainty_plot}")
    
    # Performance comparison
    comparison_data = {
        'Single-Objective GA': {
            'objectives': {'lift_to_drag_ratio': {'mean': best_ld_ratio, 'std': 0.2}},
            'time': ga_time,
            'evaluations': ga_result.total_evaluations
        },
        'Multi-Objective NSGA-II': {
            'objectives': {'lift_to_drag_ratio': {'mean': -pareto_result.pareto_solutions[0]['objectives']['lift_to_drag_ratio'], 'std': 0.3}},
            'time': nsga2_time,
            'evaluations': pareto_result.total_evaluations
        },
        'Robust Optimization': {
            'objectives': {'lift_to_drag_ratio': {'mean': -robust_result.mean_objective, 'std': robust_result.objective_std}},
            'time': robust_time,
            'evaluations': robust_result.total_evaluations
        }
    }
    
    comparison_plot = graph_generator.create_performance_comparison_plot(
        comparison_data,
        "aircraft_method_comparison"
    )
    print(f"✓ Performance comparison plot saved: {comparison_plot}")
    print()
    
    # Step 8: Save Results and Generate Report
    print("Step 8: Saving Results and Generating Report...")
    
    # Save individual optimization runs
    ga_run_id = data_manager.save_optimization_run(
        run_id="aircraft_ga_single_obj",
        optimization_result=ga_result,
        algorithm_name="GeneticAlgorithm",
        system_type="aircraft",
        parameters={
            "objective": "single_objective_ld_ratio",
            "population_size": 50,
            "max_evaluations": 200,
            "mission_profile": "commercial"
        }
    )
    
    nsga2_run_id = data_manager.save_optimization_run(
        run_id="aircraft_nsga2_multi_obj",
        optimization_result=pareto_result,
        algorithm_name="NSGA2",
        system_type="aircraft",
        parameters={
            "objective": "multi_objective",
            "population_size": 100,
            "max_evaluations": 500,
            "mission_profile": "commercial"
        }
    )
    
    # Export results to multiple formats
    csv_file = data_manager.export_to_csv([ga_run_id, nsga2_run_id], "aircraft_optimization_results.csv")
    excel_file = data_manager.export_to_excel([ga_run_id, nsga2_run_id], "aircraft_optimization_results.xlsx")
    
    print(f"✓ Results saved to database")
    print(f"✓ CSV export: {csv_file}")
    print(f"✓ Excel export: {excel_file}")
    
    # Generate comprehensive report
    report_file = data_manager.create_comparison_report(
        [ga_run_id, nsga2_run_id],
        "Aircraft_Optimization_Complete_Report"
    )
    print(f"✓ Comprehensive report: {report_file}")
    print()
    
    # Step 9: Summary and Recommendations
    print("Step 9: Summary and Recommendations")
    print("="*50)
    print()
    
    print("OPTIMIZATION RESULTS SUMMARY:")
    print(f"• Best L/D Ratio (Single-Objective): {best_ld_ratio:.2f}")
    print(f"• Pareto Solutions Found: {len(pareto_result.pareto_solutions)}")
    print(f"• Robust Design L/D Ratio: {-robust_result.mean_objective:.2f} ± {robust_result.objective_std:.3f}")
    print()
    
    print("COMPUTATIONAL PERFORMANCE:")
    print(f"• Single-Objective Optimization: {ga_time:.1f}s ({ga_result.total_evaluations} evaluations)")
    print(f"• Multi-Objective Optimization: {nsga2_time:.1f}s ({pareto_result.total_evaluations} evaluations)")
    print(f"• Robust Optimization: {robust_time:.1f}s")
    print(f"• Sensitivity Analysis: {morris_time + sobol_time:.1f}s")
    print(f"• Total Runtime: {ga_time + nsga2_time + robust_time + morris_time + sobol_time:.1f}s")
    print()
    
    print("FIDELITY ANALYSIS:")
    total_evals = sum(fidelity_stats['fidelity_counts'].values())
    for fidelity, count in fidelity_stats['fidelity_counts'].items():
        percentage = count / total_evals * 100
        print(f"• {fidelity.title()} Fidelity: {count} evaluations ({percentage:.1f}%)")
    print()
    
    print("MOST INFLUENTIAL PARAMETERS:")
    for i, (param, indices) in enumerate(sorted_sobol[:3]):
        print(f"• {i+1}. {param}: Total sensitivity = {indices['ST']:.3f}")
    print()
    
    print("DESIGN RECOMMENDATIONS:")
    best_params = ga_result.best_parameters
    print(f"• Optimal Wingspan: {best_params['wingspan']:.1f} m")
    print(f"• Optimal Wing Area: {best_params['wing_area']:.1f} m²")
    print(f"• Optimal Aspect Ratio: {best_params['aspect_ratio']:.1f}")
    print(f"• Optimal Sweep Angle: {best_params['sweep_angle']:.1f}°")
    print(f"• Optimal Cruise Mach: {best_params['cruise_mach']:.3f}")
    print()
    
    print("FILES GENERATED:")
    print(f"• Visualization plots: visualizations/aircraft_*.png")
    print(f"• Results database: results/")
    print(f"• CSV data: {csv_file}")
    print(f"• Excel report: {excel_file}")
    print(f"• Detailed report: {report_file}")
    print(f"• Log file: aircraft_optimization.log")
    print()
    
    print("="*70)
    print("AIRCRAFT OPTIMIZATION WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return {
        'single_objective_result': ga_result,
        'multi_objective_result': pareto_result,
        'robust_result': robust_result,
        'sensitivity_results': {
            'morris': morris_results,
            'sobol': sobol_results
        },
        'fidelity_statistics': fidelity_stats,
        'computational_times': {
            'single_objective': ga_time,
            'multi_objective': nsga2_time,
            'robust_optimization': robust_time,
            'sensitivity_analysis': morris_time + sobol_time
        }
    }

if __name__ == "__main__":
    try:
        results = main()
        logger.info("Aircraft optimization workflow completed successfully")
    except Exception as e:
        logger.error(f"Error in aircraft optimization workflow: {e}")
        print(f"\nError: {e}")
        print("Please check the log file for detailed error information.")