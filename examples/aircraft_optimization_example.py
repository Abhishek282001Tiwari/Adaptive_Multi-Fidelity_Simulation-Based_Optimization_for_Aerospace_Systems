#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import logging
from typing import Dict, Any

# Import our modules
from models.aerospace_systems import AircraftOptimizationSystem
from optimization.algorithms import GeneticAlgorithm, ParticleSwarmOptimization, BayesianOptimization
from optimization.robust_optimization import UncertaintyQuantification, UncertaintyDistribution, RobustOptimizer
from utilities.data_manager import DataManager
from visualization.graph_generator import ProfessionalGraphGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_aircraft_parameter_bounds():
    """Define parameter bounds for aircraft optimization."""
    return {
        'wingspan': (30.0, 60.0),
        'wing_area': (150.0, 350.0),
        'aspect_ratio': (7.0, 12.0),
        'sweep_angle': (15.0, 35.0),
        'taper_ratio': (0.4, 0.8),
        'thickness_ratio': (0.10, 0.16),
        'cruise_altitude': (9000.0, 13000.0),
        'cruise_mach': (0.65, 0.85),
        'weight': (40000.0, 80000.0)
    }


def setup_uncertainty_quantification():
    """Setup uncertainty quantification for robust optimization."""
    uq = UncertaintyQuantification()
    
    # Parameter uncertainties (manufacturing tolerances)
    uq.add_parameter_uncertainty('wingspan', UncertaintyDistribution(
        'normal', {'mean': 0.0, 'std': 0.5}, bounds=(-2.0, 2.0)
    ))
    uq.add_parameter_uncertainty('wing_area', UncertaintyDistribution(
        'normal', {'mean': 0.0, 'std': 5.0}, bounds=(-15.0, 15.0)
    ))
    uq.add_parameter_uncertainty('thickness_ratio', UncertaintyDistribution(
        'normal', {'mean': 0.0, 'std': 0.005}, bounds=(-0.02, 0.02)
    ))
    
    # Environmental uncertainties
    uq.add_environmental_uncertainty('temperature_variation', UncertaintyDistribution(
        'uniform', {'low': -10.0, 'high': 10.0}
    ))
    uq.add_environmental_uncertainty('atmospheric_density', UncertaintyDistribution(
        'normal', {'mean': 1.0, 'std': 0.05}, bounds=(0.9, 1.1)
    ))
    
    return uq


def run_aircraft_optimization_comparison():
    """Run comprehensive aircraft optimization comparison."""
    logger.info("Starting aircraft optimization comparison study")
    
    # Initialize systems
    aircraft_system = AircraftOptimizationSystem()
    data_manager = DataManager("results/aircraft_optimization")
    graph_generator = ProfessionalGraphGenerator("visualizations/aircraft")
    
    # Parameter bounds
    param_bounds = create_aircraft_parameter_bounds()
    
    # Mission profiles to test
    mission_profiles = ['commercial', 'regional', 'business_jet']
    
    # Optimization algorithms
    algorithms = {
        'genetic_algorithm': GeneticAlgorithm(param_bounds, population_size=40),
        'particle_swarm': ParticleSwarmOptimization(param_bounds, swarm_size=30),
        'bayesian_optimization': BayesianOptimization(param_bounds)
    }
    
    results = {}
    
    # Run optimization for each algorithm and mission profile
    for mission_profile in mission_profiles:
        logger.info(f"Optimizing for {mission_profile} mission profile")
        results[mission_profile] = {}
        
        for alg_name, optimizer in algorithms.items():
            logger.info(f"Running {alg_name}")
            
            def objective_function(parameters):
                return aircraft_system.evaluate_design(parameters, mission_profile)
            
            # Run optimization
            result = optimizer.optimize(
                objective_function=lambda params: aircraft_system.evaluate_design(params, mission_profile)['simulation_result'],
                max_evaluations=100
            )
            
            # Store results
            results[mission_profile][alg_name] = result
            
            # Save to data manager
            run_id = f"aircraft_{mission_profile}_{alg_name}"
            data_manager.save_optimization_run(
                run_id=run_id,
                optimization_result=result,
                algorithm_name=alg_name,
                system_type="aircraft",
                parameters={'mission_profile': mission_profile, 'param_bounds': param_bounds}
            )
            
            # Generate convergence plot
            if result.optimization_history:
                graph_generator.create_convergence_plot(
                    result.optimization_history,
                    f"{alg_name.title()} - {mission_profile.title()}",
                    f"aircraft_{mission_profile}_{alg_name}"
                )
    
    # Generate comparison plots
    for mission_profile in mission_profiles:
        mission_results = results[mission_profile]
        
        # Create performance comparison
        comparison_data = {}
        for alg_name, result in mission_results.items():
            comparison_data[alg_name] = {
                'objectives': {
                    obj_name: {'mean': obj_value, 'std': 0.0}
                    for obj_name, obj_value in result.best_objectives.items()
                }
            }
        
        graph_generator.create_performance_comparison_plot(
            comparison_data,
            f"aircraft_{mission_profile}_comparison"
        )
    
    # Create 3D design space plot for best algorithm
    best_results = []
    for mission_profile in mission_profiles:
        for alg_name, result in results[mission_profile].items():
            best_results.extend(result.optimization_history)
    
    if best_results:
        graph_generator.create_3d_design_space_plot(
            best_results,
            ['wingspan', 'wing_area', 'aspect_ratio'],
            "aircraft_design_space_exploration"
        )
    
    return results


def run_robust_aircraft_optimization():
    """Run robust optimization with uncertainty quantification."""
    logger.info("Starting robust aircraft optimization")
    
    # Setup systems
    aircraft_system = AircraftOptimizationSystem()
    uq = setup_uncertainty_quantification()
    robust_optimizer = RobustOptimizer(uq)
    data_manager = DataManager("results/robust_aircraft")
    graph_generator = ProfessionalGraphGenerator("visualizations/robust_aircraft")
    
    param_bounds = create_aircraft_parameter_bounds()
    
    def aircraft_objective(parameters):
        result = aircraft_system.evaluate_design(parameters, 'commercial')
        return result['simulation_result']
    
    # Run robust optimization
    robust_result = robust_optimizer.robust_optimization(
        objective_function=aircraft_objective,
        parameter_bounds=param_bounds,
        robustness_measure='mean_std',
        n_mc_samples=50
    )
    
    # Save results
    data_manager.save_optimization_run(
        run_id="robust_aircraft_commercial",
        optimization_result=robust_result,
        algorithm_name="robust_optimization",
        system_type="aircraft",
        parameters={'robustness_measure': 'mean_std', 'n_mc_samples': 50}
    )
    
    # Generate uncertainty propagation plot
    if robust_result.monte_carlo_results:
        graph_generator.create_uncertainty_propagation_plot(
            robust_result.monte_carlo_results,
            "robust_aircraft_uncertainty"
        )
    
    # Generate statistical distributions
    if robust_result.monte_carlo_results:
        objectives_data = {}
        for mc_result in robust_result.monte_carlo_results:
            for obj_name, obj_value in mc_result['objectives'].items():
                if obj_name not in objectives_data:
                    objectives_data[obj_name] = []
                objectives_data[obj_name].append(obj_value)
        
        graph_generator.create_statistical_distribution_plot(
            objectives_data,
            "robust_aircraft_distributions"
        )
    
    return robust_result


def analyze_fidelity_performance():
    """Analyze multi-fidelity performance."""
    logger.info("Analyzing multi-fidelity performance")
    
    aircraft_system = AircraftOptimizationSystem()
    data_manager = DataManager("results/fidelity_analysis")
    graph_generator = ProfessionalGraphGenerator("visualizations/fidelity_analysis")
    
    # Test parameters
    test_parameters = {
        'wingspan': 45.0,
        'wing_area': 250.0,
        'aspect_ratio': 9.0,
        'sweep_angle': 25.0,
        'taper_ratio': 0.6,
        'thickness_ratio': 0.12,
        'cruise_altitude': 11000.0,
        'cruise_mach': 0.78,
        'weight': 60000.0
    }
    
    # Evaluate with different fidelity levels
    fidelity_results = []
    
    for fidelity in ['low', 'medium', 'high']:
        from simulation.base import FidelityLevel
        fidelity_level = FidelityLevel(fidelity)
        
        # Multiple evaluations to get timing statistics
        times = []
        results = []
        
        for i in range(10):
            result = aircraft_system.multi_fidelity_sim.evaluate(
                test_parameters, force_fidelity=fidelity_level
            )
            times.append(result.computation_time)
            results.append(result)
        
        fidelity_results.append({
            'fidelity': fidelity,
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'avg_objectives': {
                obj_name: np.mean([r.objectives[obj_name] for r in results])
                for obj_name in results[0].objectives.keys()
            }
        })
    
    # Save fidelity analysis
    data_manager.save_optimization_run(
        run_id="fidelity_performance_analysis",
        optimization_result=fidelity_results,
        algorithm_name="fidelity_analysis",
        system_type="aircraft",
        parameters=test_parameters
    )
    
    return fidelity_results


def generate_comprehensive_report():
    """Generate comprehensive optimization report."""
    logger.info("Generating comprehensive optimization report")
    
    data_manager = DataManager("results/aircraft_optimization")
    graph_generator = ProfessionalGraphGenerator("visualizations/comprehensive_report")
    
    # Get all optimization runs
    data_manager._scan_existing_results()
    run_ids = list(data_manager.optimization_runs.keys())
    
    if not run_ids:
        logger.warning("No optimization runs found for report generation")
        return
    
    # Export to different formats
    csv_file = data_manager.export_to_csv(run_ids, "aircraft_optimization_results")
    excel_file = data_manager.export_to_excel(run_ids, "aircraft_optimization_comprehensive")
    
    # Create comparison report
    comparison_report = data_manager.create_comparison_report(
        run_ids, "aircraft_optimization_comparison"
    )
    
    # Generate comprehensive visualization report
    optimization_results = []
    for run_id in run_ids:
        run_data = data_manager.load_optimization_run(run_id)
        if run_data and 'optimization_result' in run_data:
            optimization_results.append(run_data['optimization_result'])
    
    if optimization_results:
        generated_plots = graph_generator.generate_comprehensive_report(
            optimization_results, "aircraft_comprehensive"
        )
    
    logger.info(f"Generated comprehensive report with {len(generated_plots)} plots")
    logger.info(f"CSV export: {csv_file}")
    logger.info(f"Excel export: {excel_file}")
    logger.info(f"Comparison report: {comparison_report}")
    
    return {
        'csv_file': csv_file,
        'excel_file': excel_file,
        'comparison_report': comparison_report,
        'generated_plots': generated_plots
    }


def main():
    """Main execution function."""
    logger.info("Starting aircraft optimization example")
    
    try:
        # Create necessary directories
        os.makedirs("results", exist_ok=True)
        os.makedirs("visualizations", exist_ok=True)
        
        # Run optimization comparison
        optimization_results = run_aircraft_optimization_comparison()
        logger.info("Completed optimization comparison")
        
        # Run robust optimization
        robust_results = run_robust_aircraft_optimization()
        logger.info("Completed robust optimization")
        
        # Analyze fidelity performance
        fidelity_results = analyze_fidelity_performance()
        logger.info("Completed fidelity analysis")
        
        # Generate comprehensive report
        report_results = generate_comprehensive_report()
        logger.info("Generated comprehensive report")
        
        # Print summary
        print("\n" + "="*80)
        print("AIRCRAFT OPTIMIZATION EXAMPLE - SUMMARY")
        print("="*80)
        
        print(f"\nOptimization Algorithms Tested: {len(optimization_results['commercial'])}")
        print(f"Mission Profiles Analyzed: {len(optimization_results)}")
        print(f"Fidelity Levels Evaluated: {len(fidelity_results)}")
        
        print("\nBest Results by Mission Profile:")
        for mission_profile, mission_results in optimization_results.items():
            best_alg = max(mission_results.keys(), 
                          key=lambda alg: sum(mission_results[alg].best_objectives.values()))
            best_fitness = sum(mission_results[best_alg].best_objectives.values())
            print(f"  {mission_profile.title()}: {best_alg.title()} (fitness: {best_fitness:.3f})")
        
        print(f"\nRobust Optimization Results:")
        print(f"  Best Parameters: {robust_results.robust_parameters}")
        print(f"  Robust Objectives: {robust_results.robust_objectives}")
        print(f"  Reliability Metrics: {robust_results.reliability_metrics}")
        
        print(f"\nFidelity Performance Analysis:")
        for fidelity_result in fidelity_results:
            print(f"  {fidelity_result['fidelity'].title()} Fidelity: "
                  f"{fidelity_result['avg_time']:.3f}±{fidelity_result['std_time']:.3f}s")
        
        print(f"\nGenerated Files:")
        if 'csv_file' in report_results:
            print(f"  CSV Export: {report_results['csv_file']}")
        if 'excel_file' in report_results:
            print(f"  Excel Report: {report_results['excel_file']}")
        if 'comparison_report' in report_results:
            print(f"  Comparison Report: {report_results['comparison_report']}")
        
        print("\n" + "="*80)
        print("Example completed successfully!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error in aircraft optimization example: {e}")
        raise


if __name__ == "__main__":
    main()