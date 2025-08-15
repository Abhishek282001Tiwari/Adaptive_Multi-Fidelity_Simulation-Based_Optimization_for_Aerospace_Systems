#!/usr/bin/env python3
"""
Complete Spacecraft Optimization Workflow
==========================================

This example demonstrates a complete spacecraft optimization workflow including:
1. Earth observation satellite optimization
2. Communication satellite constellation design
3. Interplanetary mission planning
4. Robust design under uncertainty
5. Mission success probability analysis
6. Comprehensive visualization and reporting

This is a fully working example for spacecraft mission design and optimization.
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
        logging.FileHandler('spacecraft_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Complete spacecraft optimization workflow"""
    
    print("="*70)
    print("ADAPTIVE MULTI-FIDELITY SPACECRAFT OPTIMIZATION")
    print("="*70)
    print()
    
    # Step 1: Initialize System
    print("Step 1: Initializing Spacecraft Optimization System...")
    
    try:
        from src.models.aerospace_systems import SpacecraftOptimizationSystem
        from src.optimization.algorithms import BayesianOptimization, NSGA2
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
    spacecraft_system = SpacecraftOptimizationSystem()
    graph_generator = ProfessionalGraphGenerator("visualizations/")
    data_manager = DataManager("results/")
    
    print("✓ System initialized successfully")
    print()
    
    # Mission scenarios to optimize
    missions = {
        'earth_observation': {
            'name': 'Earth Observation Satellite',
            'bounds': {
                'dry_mass': (1500, 6000),           # kg
                'fuel_mass': (8000, 25000),         # kg
                'specific_impulse': (280, 350),     # seconds
                'thrust': (2000, 8000),             # Newtons
                'solar_panel_area': (40.0, 120.0),  # m²
                'thermal_mass': (800, 2500),        # kg
                'target_orbit_altitude': (400, 800), # km
                'mission_duration': (1095, 2190)    # days (3-6 years)
            }
        },
        'communication': {
            'name': 'Communication Satellite',
            'bounds': {
                'dry_mass': (2000, 8000),           # kg
                'fuel_mass': (12000, 35000),        # kg
                'specific_impulse': (300, 400),     # seconds
                'thrust': (1500, 6000),             # Newtons
                'solar_panel_area': (60.0, 180.0),  # m²
                'thermal_mass': (1200, 3500),       # kg
                'target_orbit_altitude': (35786, 35786), # km (GEO)
                'mission_duration': (4380, 7300)    # days (12-20 years)
            }
        },
        'deep_space': {
            'name': 'Deep Space Probe',
            'bounds': {
                'dry_mass': (1000, 4000),           # kg
                'fuel_mass': (15000, 60000),        # kg
                'specific_impulse': (350, 450),     # seconds (ion propulsion)
                'thrust': (100, 1000),              # Newtons (low thrust)
                'solar_panel_area': (80.0, 250.0),  # m² (large for deep space)
                'thermal_mass': (600, 2000),        # kg
                'target_orbit_altitude': (0, 0),    # N/A for interplanetary
                'mission_duration': (3650, 10950)   # days (10-30 years)
            }
        }
    }
    
    results = {}
    
    # Step 2: Earth Observation Satellite Optimization
    print("Step 2: Earth Observation Satellite Optimization...")
    
    mission_type = 'earth_observation'
    mission_config = missions[mission_type]
    parameter_bounds = mission_config['bounds']
    
    print(f"Optimizing {mission_config['name']} with {len(parameter_bounds)} parameters:")
    for param, bounds in parameter_bounds.items():
        if param != 'target_orbit_altitude' or bounds[0] != bounds[1]:
            print(f"  {param}: [{bounds[0]:.1f}, {bounds[1]:.1f}]")
        else:
            print(f"  {param}: {bounds[0]:.0f} km (fixed)")
    print()
    
    def earth_obs_objective(parameters):
        """Optimize Earth observation satellite for mission success and mass efficiency"""
        result = spacecraft_system.evaluate_design(parameters, mission_type)
        sim_result = result['simulation_result']
        
        return {
            'mission_success': -sim_result.objectives['mission_success_probability'],  # Maximize
            'total_mass': parameters['dry_mass'] + parameters['fuel_mass'],           # Minimize
            'power_efficiency': -sim_result.objectives['power_efficiency']            # Maximize
        }
    
    # Use Bayesian Optimization for expensive spacecraft evaluations
    bayesian_optimizer = BayesianOptimization(
        parameter_bounds=parameter_bounds,
        acquisition_function='ei',
        xi=0.01
    )
    
    start_time = time.time()
    earth_obs_result = bayesian_optimizer.optimize(
        objective_function=earth_obs_objective,
        max_evaluations=100
    )
    earth_obs_time = time.time() - start_time
    
    results[mission_type] = {
        'result': earth_obs_result,
        'time': earth_obs_time,
        'mission_config': mission_config
    }
    
    print(f"✓ Earth observation optimization completed in {earth_obs_time:.1f} seconds")
    print(f"  Mission success probability: {-earth_obs_result.best_objectives['mission_success']:.3f}")
    print(f"  Total mass: {earth_obs_result.best_objectives['total_mass']:.0f} kg")
    print(f"  Power efficiency: {-earth_obs_result.best_objectives['power_efficiency']:.3f}")
    print()
    
    # Step 3: Communication Satellite Multi-Objective Optimization
    print("Step 3: Communication Satellite Multi-Objective Optimization...")
    
    mission_type = 'communication'
    mission_config = missions[mission_type]
    parameter_bounds = mission_config['bounds']
    
    def comm_sat_objective(parameters):
        """Multi-objective optimization for communication satellite"""
        result = spacecraft_system.evaluate_design(parameters, mission_type)
        sim_result = result['simulation_result']
        
        # Calculate coverage-related metrics
        power_margin = sim_result.objectives['power_efficiency'] - 1.0
        
        return {
            'mission_success': -sim_result.objectives['mission_success_probability'],  # Maximize
            'total_cost': (parameters['dry_mass'] + parameters['fuel_mass']) * 10000,  # Minimize (cost proxy)
            'power_margin': -power_margin,                                             # Maximize margin
            'mission_lifetime': -parameters['mission_duration']                       # Maximize lifetime
        }
    
    # Use NSGA-II for multi-objective optimization
    nsga2_optimizer = NSGA2(
        parameter_bounds=parameter_bounds,
        population_size=80,
        crossover_rate=0.9,
        mutation_rate=0.1
    )
    
    start_time = time.time()
    comm_pareto_result = nsga2_optimizer.optimize(
        objective_function=comm_sat_objective,
        max_evaluations=400
    )
    comm_time = time.time() - start_time
    
    results[mission_type] = {
        'result': comm_pareto_result,
        'time': comm_time,
        'mission_config': mission_config
    }
    
    print(f"✓ Communication satellite optimization completed in {comm_time:.1f} seconds")
    print(f"  Pareto optimal solutions: {len(comm_pareto_result.pareto_solutions)}")
    print(f"  Best mission success: {-min([s['objectives']['mission_success'] for s in comm_pareto_result.pareto_solutions]):.3f}")
    print()
    
    # Step 4: Deep Space Probe Robust Optimization
    print("Step 4: Deep Space Probe Robust Optimization...")
    
    mission_type = 'deep_space'
    mission_config = missions[mission_type]
    parameter_bounds = mission_config['bounds']
    
    # Define uncertainties for deep space mission
    uq = UncertaintyQuantification()
    
    # Harsh space environment uncertainties
    uq.add_parameter_uncertainty('dry_mass', UncertaintyDistribution(
        'normal', {'mean': 0.0, 'std': 150.0}, bounds=(-500.0, 500.0)
    ))
    uq.add_parameter_uncertainty('specific_impulse', UncertaintyDistribution(
        'normal', {'mean': 0.0, 'std': 10.0}, bounds=(-30.0, 30.0)
    ))
    
    # Deep space environmental uncertainties
    uq.add_environmental_uncertainty('solar_flux_variation', UncertaintyDistribution(
        'uniform', {'low': -100.0, 'high': 50.0}  # Reduced solar flux at distance
    ))
    uq.add_environmental_uncertainty('radiation_degradation', UncertaintyDistribution(
        'lognormal', {'mean': 0.0, 'sigma': 0.15}
    ))
    
    def deep_space_objective(parameters):
        """Deep space mission optimization"""
        result = spacecraft_system.evaluate_design(parameters, mission_type)
        sim_result = result['simulation_result']
        
        return {
            'delta_v_capability': -sim_result.objectives['delta_v_capability'],  # Maximize
            'mission_success': -sim_result.objectives['mission_success_probability'],  # Maximize
            'total_mass': parameters['dry_mass'] + parameters['fuel_mass']       # Minimize
        }
    
    # Run robust optimization
    robust_optimizer = RobustOptimizer(uq)
    
    start_time = time.time()
    deep_space_robust_result = robust_optimizer.robust_optimization(
        objective_function=deep_space_objective,
        parameter_bounds=parameter_bounds,
        robustness_measure='mean_std',
        n_mc_samples=150,
        optimization_algorithm='bayesian_optimization'
    )
    deep_space_time = time.time() - start_time
    
    results[mission_type] = {
        'result': deep_space_robust_result,
        'time': deep_space_time,
        'mission_config': mission_config,
        'robust': True
    }
    
    print(f"✓ Deep space probe robust optimization completed in {deep_space_time:.1f} seconds")
    print(f"  Robust delta-V capability: {-deep_space_robust_result.robust_objective:.0f} m/s")
    print(f"  Mean mission success: {-deep_space_robust_result.mean_objective:.3f}")
    print(f"  Objective std deviation: {deep_space_robust_result.objective_std:.4f}")
    print()
    
    # Step 5: Sensitivity Analysis for Earth Observation Mission
    print("Step 5: Global Sensitivity Analysis...")
    
    sensitivity = SensitivityAnalysis()
    
    # Use the earth observation mission for sensitivity analysis
    parameter_bounds = missions['earth_observation']['bounds']
    
    # Morris screening
    start_time = time.time()
    morris_results = sensitivity.morris_screening(
        objective_function=earth_obs_objective,
        parameter_bounds=parameter_bounds,
        n_trajectories=15
    )
    morris_time = time.time() - start_time
    
    print(f"✓ Morris screening completed in {morris_time:.1f} seconds")
    print("  Parameter sensitivity ranking:")
    
    # Sort by mu_star for mission success objective
    sorted_morris = sorted(
        [(param, indices) for param, indices in morris_results.items()],
        key=lambda x: x[1]['mu_star'],
        reverse=True
    )
    
    for i, (param, indices) in enumerate(sorted_morris[:5]):
        print(f"    {i+1}. {param}: μ*={indices['mu_star']:.3f}")
    print()
    
    # Step 6: Mission Performance Comparison
    print("Step 6: Mission Performance Comparison...")
    
    print("MISSION COMPARISON SUMMARY:")
    print("-" * 50)
    
    for mission_name, mission_data in results.items():
        result = mission_data['result']
        time_taken = mission_data['time']
        config = mission_data['mission_config']
        
        print(f"\n{config['name']}:")
        print(f"  Optimization time: {time_taken:.1f} seconds")
        
        if hasattr(result, 'best_objectives'):  # Single/Robust optimization
            if 'mission_success' in result.best_objectives:
                print(f"  Mission success probability: {-result.best_objectives['mission_success']:.3f}")
            if 'total_mass' in result.best_objectives:
                print(f"  Total mass: {result.best_objectives['total_mass']:.0f} kg")
            if 'delta_v_capability' in result.best_objectives:
                print(f"  Delta-V capability: {-result.best_objectives['delta_v_capability']:.0f} m/s")
        else:  # Multi-objective (Pareto solutions)
            best_success = min([s['objectives']['mission_success'] for s in result.pareto_solutions])
            print(f"  Best mission success: {-best_success:.3f}")
            print(f"  Pareto solutions: {len(result.pareto_solutions)}")
    
    print()
    
    # Step 7: Generate Visualizations
    print("Step 7: Generating Comprehensive Visualizations...")
    
    visualization_files = []
    
    # Earth observation convergence
    if hasattr(results['earth_observation']['result'], 'optimization_history'):
        conv_plot = graph_generator.create_convergence_plot(
            results['earth_observation']['result'].optimization_history,
            "Bayesian Optimization - Earth Observation Satellite",
            "earth_obs_convergence"
        )
        visualization_files.append(conv_plot)
        print(f"✓ Earth obs convergence plot: {conv_plot}")
    
    # Communication satellite Pareto front
    pareto_plot = graph_generator.create_pareto_front_plot(
        results['communication']['result'].pareto_solutions,
        "communication_pareto_front"
    )
    visualization_files.append(pareto_plot)
    print(f"✓ Communication sat Pareto plot: {pareto_plot}")
    
    # Deep space uncertainty analysis
    if hasattr(results['deep_space']['result'], 'monte_carlo_results'):
        uncertainty_plot = graph_generator.create_uncertainty_propagation_plot(
            results['deep_space']['result'].monte_carlo_results,
            "deep_space_uncertainty_analysis"
        )
        visualization_files.append(uncertainty_plot)
        print(f"✓ Deep space uncertainty plot: {uncertainty_plot}")
    
    # Mission comparison
    comparison_data = {}
    for mission_name, mission_data in results.items():
        result = mission_data['result']
        config = mission_data['mission_config']
        
        if hasattr(result, 'best_objectives'):
            success_prob = -result.best_objectives.get('mission_success', -0.9)
            comparison_data[config['name']] = {
                'objectives': {
                    'mission_success_probability': {'mean': success_prob, 'std': 0.02}
                },
                'time': mission_data['time']
            }
    
    comparison_plot = graph_generator.create_performance_comparison_plot(
        comparison_data,
        "spacecraft_mission_comparison"
    )
    visualization_files.append(comparison_plot)
    print(f"✓ Mission comparison plot: {comparison_plot}")
    
    # 3D design space plot for Earth observation
    if hasattr(results['earth_observation']['result'], 'optimization_history'):
        design_3d_plot = graph_generator.create_3d_design_space_plot(
            results['earth_observation']['result'].optimization_history,
            ['dry_mass', 'fuel_mass', 'solar_panel_area'],
            "earth_obs_design_space_3d"
        )
        visualization_files.append(design_3d_plot)
        print(f"✓ 3D design space plot: {design_3d_plot}")
    
    print()
    
    # Step 8: Save Results and Generate Report
    print("Step 8: Saving Results and Generating Comprehensive Report...")
    
    run_ids = []
    
    # Save each mission optimization
    for mission_name, mission_data in results.items():
        result = mission_data['result']
        config = mission_data['mission_config']
        
        run_id = data_manager.save_optimization_run(
            run_id=f"spacecraft_{mission_name}",
            optimization_result=result,
            algorithm_name=result.algorithm_name if hasattr(result, 'algorithm_name') else 'Unknown',
            system_type="spacecraft",
            parameters={
                "mission_type": mission_name,
                "mission_name": config['name'],
                "parameter_count": len(config['bounds']),
                "optimization_time": mission_data['time']
            }
        )
        run_ids.append(run_id)
    
    # Export consolidated results
    csv_file = data_manager.export_to_csv(run_ids, "spacecraft_mission_results.csv")
    excel_file = data_manager.export_to_excel(run_ids, "spacecraft_mission_results.xlsx")
    hdf5_file = data_manager.export_to_hdf5(run_ids, "spacecraft_mission_results.h5")
    
    print(f"✓ Results saved for {len(run_ids)} missions")
    print(f"✓ CSV export: {csv_file}")
    print(f"✓ Excel export: {excel_file}")
    print(f"✓ HDF5 export: {hdf5_file}")
    
    # Generate comprehensive mission report
    report_file = data_manager.create_comparison_report(
        run_ids,
        "Spacecraft_Mission_Optimization_Report"
    )
    print(f"✓ Comprehensive report: {report_file}")
    print()
    
    # Step 9: Final Summary and Design Recommendations
    print("Step 9: Final Summary and Design Recommendations")
    print("="*60)
    print()
    
    print("MISSION OPTIMIZATION RESULTS:")
    total_time = sum([mission_data['time'] for mission_data in results.values()])
    print(f"• Total optimization time: {total_time:.1f} seconds")
    print(f"• Missions optimized: {len(results)}")
    print(f"• Visualizations generated: {len(visualization_files)}")
    print()
    
    print("MISSION-SPECIFIC RESULTS:")
    for mission_name, mission_data in results.items():
        result = mission_data['result']
        config = mission_data['mission_config']
        print(f"\n{config['name']}:")
        
        if hasattr(result, 'best_parameters'):
            params = result.best_parameters
            print(f"  • Optimal dry mass: {params['dry_mass']:.0f} kg")
            print(f"  • Optimal fuel mass: {params['fuel_mass']:.0f} kg")
            print(f"  • Optimal solar panel area: {params['solar_panel_area']:.1f} m²")
            print(f"  • Mission duration: {params['mission_duration']:.0f} days")
            
            if hasattr(result, 'best_objectives'):
                objs = result.best_objectives
                if 'mission_success' in objs:
                    print(f"  • Mission success probability: {-objs['mission_success']:.1%}")
                if 'total_mass' in objs:
                    print(f"  • Total spacecraft mass: {objs['total_mass']:.0f} kg")
    
    print(f"\nMOST INFLUENTIAL DESIGN PARAMETERS:")
    for i, (param, indices) in enumerate(sorted_morris[:3]):
        print(f"• {i+1}. {param}: Sensitivity = {indices['mu_star']:.3f}")
    print()
    
    print("DESIGN RECOMMENDATIONS BY MISSION TYPE:")
    print("• Earth Observation: Prioritize power efficiency and mass optimization")
    print("• Communication: Balance mission lifetime with cost constraints") 
    print("• Deep Space: Maximize delta-V capability with robust design approach")
    print()
    
    print("COMPUTATIONAL PERFORMANCE:")
    for mission_name, mission_data in results.items():
        result = mission_data['result']
        config = mission_data['mission_config']
        evaluations = getattr(result, 'total_evaluations', 'N/A')
        print(f"• {config['name']}: {mission_data['time']:.1f}s ({evaluations} evaluations)")
    print()
    
    print("FILES GENERATED:")
    print("• Visualization plots: visualizations/spacecraft_*.png, earth_obs_*.png, etc.")
    print("• Results database: results/spacecraft_*.json")
    print(f"• CSV data: {csv_file}")
    print(f"• Excel report: {excel_file}")
    print(f"• HDF5 archive: {hdf5_file}")
    print(f"• Comprehensive report: {report_file}")
    print("• Log file: spacecraft_optimization.log")
    print()
    
    print("="*70)
    print("SPACECRAFT OPTIMIZATION WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return {
        'mission_results': results,
        'sensitivity_analysis': morris_results,
        'total_time': total_time,
        'visualization_files': visualization_files,
        'export_files': {
            'csv': csv_file,
            'excel': excel_file,
            'hdf5': hdf5_file,
            'report': report_file
        }
    }

if __name__ == "__main__":
    try:
        results = main()
        logger.info("Spacecraft optimization workflow completed successfully")
    except Exception as e:
        logger.error(f"Error in spacecraft optimization workflow: {e}")
        print(f"\nError: {e}")
        print("Please check the log file for detailed error information.")