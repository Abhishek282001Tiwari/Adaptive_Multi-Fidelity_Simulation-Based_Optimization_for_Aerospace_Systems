#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import logging
from typing import Dict, Any

# Import our modules
from models.aerospace_systems import SpacecraftOptimizationSystem
from optimization.algorithms import GeneticAlgorithm, ParticleSwarmOptimization, BayesianOptimization
from optimization.robust_optimization import UncertaintyQuantification, UncertaintyDistribution, RobustOptimizer
from utilities.data_manager import DataManager
from visualization.graph_generator import ProfessionalGraphGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_spacecraft_parameter_bounds():
    """Define parameter bounds for spacecraft optimization."""
    return {
        'dry_mass': (3000.0, 15000.0),
        'fuel_mass': (8000.0, 50000.0),
        'specific_impulse': (250.0, 400.0),
        'thrust': (2000.0, 25000.0),
        'solar_panel_area': (20.0, 120.0),
        'thermal_mass': (800.0, 5000.0),
        'target_orbit_altitude': (300.0, 1000.0),
        'mission_duration': (365.0, 2555.0),
        'propellant_efficiency': (0.85, 0.96),
        'attitude_control_mass': (100.0, 600.0),
        'payload_mass': (500.0, 5000.0)
    }


def setup_spacecraft_uncertainty_quantification():
    """Setup uncertainty quantification for spacecraft robust optimization."""
    uq = UncertaintyQuantification()
    
    # Parameter uncertainties
    uq.add_parameter_uncertainty('dry_mass', UncertaintyDistribution(
        'normal', {'mean': 0.0, 'std': 200.0}, bounds=(-800.0, 800.0)
    ))
    uq.add_parameter_uncertainty('fuel_mass', UncertaintyDistribution(
        'normal', {'mean': 0.0, 'std': 500.0}, bounds=(-2000.0, 2000.0)
    ))
    uq.add_parameter_uncertainty('specific_impulse', UncertaintyDistribution(
        'normal', {'mean': 0.0, 'std': 10.0}, bounds=(-30.0, 30.0)
    ))
    uq.add_parameter_uncertainty('solar_panel_area', UncertaintyDistribution(
        'normal', {'mean': 0.0, 'std': 2.0}, bounds=(-8.0, 8.0)
    ))
    
    # Environmental uncertainties
    uq.add_environmental_uncertainty('solar_flux_variation', UncertaintyDistribution(
        'uniform', {'low': 0.95, 'high': 1.05}
    ))
    uq.add_environmental_uncertainty('atmospheric_density_uncertainty', UncertaintyDistribution(
        'lognormal', {'mean': 0.0, 'sigma': 0.3}, bounds=(0.5, 2.0)
    ))
    uq.add_environmental_uncertainty('temperature_variation', UncertaintyDistribution(
        'uniform', {'low': -25.0, 'high': 25.0}
    ))
    
    # Model uncertainties
    uq.add_model_uncertainty('thrust_efficiency', UncertaintyDistribution(
        'beta', {'alpha': 9, 'beta': 1}, bounds=(0.85, 1.0)
    ))
    uq.add_model_uncertainty('solar_degradation', UncertaintyDistribution(
        'triangular', {'left': 0.98, 'mode': 0.99, 'right': 1.0}
    ))
    
    return uq


def run_spacecraft_optimization_comparison():
    """Run comprehensive spacecraft optimization comparison."""
    logger.info("Starting spacecraft optimization comparison study")
    
    # Initialize systems
    spacecraft_system = SpacecraftOptimizationSystem()
    data_manager = DataManager("results/spacecraft_optimization")
    graph_generator = ProfessionalGraphGenerator("visualizations/spacecraft")
    
    # Parameter bounds
    param_bounds = create_spacecraft_parameter_bounds()
    
    # Mission types to test
    mission_types = ['earth_observation', 'communication', 'deep_space']
    
    # Optimization algorithms
    algorithms = {
        'genetic_algorithm': GeneticAlgorithm(param_bounds, population_size=50),
        'particle_swarm': ParticleSwarmOptimization(param_bounds, swarm_size=35),
        'bayesian_optimization': BayesianOptimization(param_bounds)
    }
    
    results = {}
    
    # Run optimization for each algorithm and mission type
    for mission_type in mission_types:
        logger.info(f"Optimizing for {mission_type} mission")
        results[mission_type] = {}
        
        for alg_name, optimizer in algorithms.items():
            logger.info(f"Running {alg_name}")
            
            def objective_function(parameters):
                return spacecraft_system.evaluate_design(parameters, mission_type)
            
            # Run optimization
            result = optimizer.optimize(
                objective_function=lambda params: spacecraft_system.evaluate_design(params, mission_type)['simulation_result'],
                max_evaluations=80
            )
            
            # Store results
            results[mission_type][alg_name] = result
            
            # Save to data manager
            run_id = f"spacecraft_{mission_type}_{alg_name}"
            data_manager.save_optimization_run(
                run_id=run_id,
                optimization_result=result,
                algorithm_name=alg_name,
                system_type="spacecraft",
                parameters={'mission_type': mission_type, 'param_bounds': param_bounds}
            )
            
            # Generate convergence plot
            if result.optimization_history:
                graph_generator.create_convergence_plot(
                    result.optimization_history,
                    f"{alg_name.title()} - {mission_type.title()}",
                    f"spacecraft_{mission_type}_{alg_name}"
                )
    
    # Generate comparison plots for each mission type
    for mission_type in mission_types:
        mission_results = results[mission_type]
        
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
            f"spacecraft_{mission_type}_comparison"
        )
    
    # Create multi-objective Pareto front analysis
    for mission_type in mission_types:
        pareto_results = []
        for alg_name, result in results[mission_type].items():
            for entry in result.optimization_history:
                if 'objectives' in entry:
                    pareto_results.append({'objectives': entry['objectives']})
        
        if pareto_results:
            graph_generator.create_pareto_front_plot(
                pareto_results,
                f"spacecraft_{mission_type}_pareto"
            )
    
    # Create 3D design space exploration
    all_results = []
    for mission_type in mission_types:
        for alg_name, result in results[mission_type].items():
            all_results.extend(result.optimization_history)
    
    if all_results:
        graph_generator.create_3d_design_space_plot(
            all_results,
            ['dry_mass', 'fuel_mass', 'solar_panel_area'],
            "spacecraft_design_space_exploration"
        )
    
    return results


def run_robust_spacecraft_optimization():
    """Run robust optimization with uncertainty quantification for spacecraft."""
    logger.info("Starting robust spacecraft optimization")
    
    # Setup systems
    spacecraft_system = SpacecraftOptimizationSystem()
    uq = setup_spacecraft_uncertainty_quantification()
    robust_optimizer = RobustOptimizer(uq)
    data_manager = DataManager("results/robust_spacecraft")
    graph_generator = ProfessionalGraphGenerator("visualizations/robust_spacecraft")
    
    param_bounds = create_spacecraft_parameter_bounds()
    
    def spacecraft_objective(parameters):
        result = spacecraft_system.evaluate_design(parameters, 'earth_observation')
        return result['simulation_result']
    
    # Run robust optimization with different robustness measures
    robustness_measures = ['mean_std', 'worst_case', 'cvar']
    robust_results = {}
    
    for measure in robustness_measures:
        logger.info(f"Running robust optimization with {measure} measure")
        
        robust_result = robust_optimizer.robust_optimization(
            objective_function=spacecraft_objective,
            parameter_bounds=param_bounds,
            robustness_measure=measure,
            n_mc_samples=60
        )
        
        robust_results[measure] = robust_result
        
        # Save results
        data_manager.save_optimization_run(
            run_id=f"robust_spacecraft_{measure}",
            optimization_result=robust_result,
            algorithm_name="robust_optimization",
            system_type="spacecraft",
            parameters={'robustness_measure': measure, 'n_mc_samples': 60}
        )
        
        # Generate uncertainty propagation plot
        if robust_result.monte_carlo_results:
            graph_generator.create_uncertainty_propagation_plot(
                robust_result.monte_carlo_results,
                f"robust_spacecraft_{measure}_uncertainty"
            )
    
    # Compare robustness measures
    comparison_data = {}
    for measure, result in robust_results.items():
        comparison_data[measure] = {
            'objectives': {
                obj_name: {'mean': obj_value, 'std': 0.0}
                for obj_name, obj_value in result.robust_objectives.items()
            }
        }
    
    graph_generator.create_performance_comparison_plot(
        comparison_data,
        "robust_spacecraft_measures_comparison"
    )
    
    return robust_results


def analyze_spacecraft_mission_sensitivity():
    """Analyze sensitivity to mission parameters."""
    logger.info("Analyzing spacecraft mission sensitivity")
    
    spacecraft_system = SpacecraftOptimizationSystem()
    data_manager = DataManager("results/spacecraft_sensitivity")
    graph_generator = ProfessionalGraphGenerator("visualizations/spacecraft_sensitivity")
    
    # Base spacecraft configuration
    base_parameters = {
        'dry_mass': 8000.0,
        'fuel_mass': 25000.0,
        'specific_impulse': 320.0,
        'thrust': 10000.0,
        'solar_panel_area': 60.0,
        'thermal_mass': 2000.0,
        'target_orbit_altitude': 600.0,
        'mission_duration': 1095.0,  # 3 years
        'propellant_efficiency': 0.90,
        'attitude_control_mass': 300.0,
        'payload_mass': 2000.0
    }
    
    # Parameters to vary for sensitivity analysis
    sensitivity_params = {
        'mission_duration': np.linspace(365, 2555, 10),
        'target_orbit_altitude': np.linspace(300, 1000, 10),
        'solar_panel_area': np.linspace(30, 100, 10),
        'fuel_mass': np.linspace(15000, 40000, 10)
    }
    
    sensitivity_results = {}
    
    for param_name, param_values in sensitivity_params.items():
        logger.info(f"Analyzing sensitivity to {param_name}")
        
        param_results = []
        
        for value in param_values:
            test_params = base_parameters.copy()
            test_params[param_name] = value
            
            result = spacecraft_system.evaluate_design(test_params, 'earth_observation')
            
            param_results.append({
                'parameter_value': value,
                'objectives': result['simulation_result'].objectives,
                'performance_metrics': result['performance_metrics'],
                'mission_success': result['mission_success_probability']['overall_success']
            })
        
        sensitivity_results[param_name] = param_results
    
    # Save sensitivity analysis results
    data_manager.save_optimization_run(
        run_id="spacecraft_sensitivity_analysis",
        optimization_result=sensitivity_results,
        algorithm_name="sensitivity_analysis",
        system_type="spacecraft",
        parameters=base_parameters
    )
    
    # Generate sensitivity plots
    for param_name, param_results in sensitivity_results.items():
        param_values = [r['parameter_value'] for r in param_results]
        
        # Plot objective sensitivity
        objectives_data = {}
        for obj_name in param_results[0]['objectives'].keys():
            objectives_data[obj_name] = [r['objectives'][obj_name] for r in param_results]
        
        # Create custom sensitivity plot
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = ['#1f4e79', '#4472c4', '#70ad47', '#ff7c00']
        
        for i, (obj_name, obj_values) in enumerate(objectives_data.items()):
            if i < len(axes):
                axes[i].plot(param_values, obj_values, 'o-', color=colors[i % len(colors)], linewidth=2, markersize=6)
                axes[i].set_xlabel(param_name.replace('_', ' ').title())
                axes[i].set_ylabel(obj_name.replace('_', ' ').title())
                axes[i].set_title(f'Sensitivity of {obj_name.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(len(objectives_data), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        sensitivity_plot_path = f"visualizations/spacecraft_sensitivity/sensitivity_{param_name}.png"
        os.makedirs(os.path.dirname(sensitivity_plot_path), exist_ok=True)
        plt.savefig(sensitivity_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved sensitivity plot: {sensitivity_plot_path}")
    
    return sensitivity_results


def analyze_fidelity_switching_performance():
    """Analyze multi-fidelity switching performance for spacecraft."""
    logger.info("Analyzing multi-fidelity switching performance")
    
    spacecraft_system = SpacecraftOptimizationSystem()
    data_manager = DataManager("results/spacecraft_fidelity")
    graph_generator = ProfessionalGraphGenerator("visualizations/spacecraft_fidelity")
    
    # Test different fidelity strategies
    from simulation.adaptive_fidelity import FidelitySwitchingStrategy
    
    strategies = [
        FidelitySwitchingStrategy.CONSERVATIVE,
        FidelitySwitchingStrategy.AGGRESSIVE,
        FidelitySwitchingStrategy.BALANCED,
        FidelitySwitchingStrategy.ADAPTIVE
    ]
    
    strategy_results = {}
    
    for strategy in strategies:
        logger.info(f"Testing {strategy.value} fidelity strategy")
        
        # Create new spacecraft system with specified strategy
        test_spacecraft = SpacecraftOptimizationSystem(strategy)
        
        # Run short optimization
        param_bounds = create_spacecraft_parameter_bounds()
        optimizer = GeneticAlgorithm(param_bounds, population_size=20)
        
        result = optimizer.optimize(
            objective_function=lambda params: test_spacecraft.evaluate_design(params, 'earth_observation')['simulation_result'],
            max_evaluations=40
        )
        
        # Get fidelity statistics
        fidelity_stats = test_spacecraft.get_optimization_statistics()
        
        strategy_results[strategy.value] = {
            'optimization_result': result,
            'fidelity_statistics': fidelity_stats,
            'fidelity_history': test_spacecraft.fidelity_manager.evaluation_history
        }
        
        # Save results
        data_manager.save_optimization_run(
            run_id=f"spacecraft_fidelity_{strategy.value}",
            optimization_result=result,
            algorithm_name=f"GA_with_{strategy.value}_fidelity",
            system_type="spacecraft",
            parameters={'fidelity_strategy': strategy.value}
        )
        
        # Generate fidelity switching plot
        if strategy_results[strategy.value]['fidelity_history']:
            graph_generator.create_fidelity_switching_plot(
                strategy_results[strategy.value]['fidelity_history'],
                f"spacecraft_fidelity_{strategy.value}"
            )
    
    # Compare fidelity strategies
    comparison_data = {}
    for strategy_name, strategy_data in strategy_results.items():
        result = strategy_data['optimization_result']
        comparison_data[strategy_name] = {
            'objectives': {
                obj_name: {'mean': obj_value, 'std': 0.0}
                for obj_name, obj_value in result.best_objectives.items()
            }
        }
    
    graph_generator.create_performance_comparison_plot(
        comparison_data,
        "spacecraft_fidelity_strategies_comparison"
    )
    
    return strategy_results


def generate_spacecraft_comprehensive_report():
    """Generate comprehensive spacecraft optimization report."""
    logger.info("Generating comprehensive spacecraft report")
    
    data_manager = DataManager("results/spacecraft_optimization")
    graph_generator = ProfessionalGraphGenerator("visualizations/spacecraft_comprehensive")
    
    # Get all spacecraft optimization runs
    data_manager._scan_existing_results()
    spacecraft_runs = [run_id for run_id, run_info in data_manager.optimization_runs.items()
                      if run_info['metadata']['system_type'] == 'spacecraft']
    
    if not spacecraft_runs:
        logger.warning("No spacecraft optimization runs found for report generation")
        return
    
    # Export results
    csv_file = data_manager.export_to_csv(spacecraft_runs, "spacecraft_optimization_results")
    excel_file = data_manager.export_to_excel(spacecraft_runs, "spacecraft_optimization_comprehensive")
    
    # Create comparison report
    comparison_report = data_manager.create_comparison_report(
        spacecraft_runs, "spacecraft_optimization_comparison"
    )
    
    # Generate comprehensive plots
    optimization_results = []
    for run_id in spacecraft_runs:
        run_data = data_manager.load_optimization_run(run_id)
        if run_data and 'optimization_result' in run_data:
            optimization_results.append(run_data['optimization_result'])
    
    generated_plots = {}
    if optimization_results:
        generated_plots = graph_generator.generate_comprehensive_report(
            optimization_results, "spacecraft_comprehensive"
        )
    
    logger.info(f"Generated comprehensive spacecraft report with {len(generated_plots)} plots")
    
    return {
        'csv_file': csv_file,
        'excel_file': excel_file,
        'comparison_report': comparison_report,
        'generated_plots': generated_plots,
        'spacecraft_runs_analyzed': len(spacecraft_runs)
    }


def main():
    """Main execution function for spacecraft optimization example."""
    logger.info("Starting spacecraft optimization example")
    
    try:
        # Create necessary directories
        os.makedirs("results", exist_ok=True)
        os.makedirs("visualizations", exist_ok=True)
        
        # Run optimization comparison
        optimization_results = run_spacecraft_optimization_comparison()
        logger.info("Completed spacecraft optimization comparison")
        
        # Run robust optimization
        robust_results = run_robust_spacecraft_optimization()
        logger.info("Completed robust spacecraft optimization")
        
        # Analyze mission sensitivity
        sensitivity_results = analyze_spacecraft_mission_sensitivity()
        logger.info("Completed mission sensitivity analysis")
        
        # Analyze fidelity switching
        fidelity_results = analyze_fidelity_switching_performance()
        logger.info("Completed fidelity switching analysis")
        
        # Generate comprehensive report
        report_results = generate_spacecraft_comprehensive_report()
        logger.info("Generated comprehensive spacecraft report")
        
        # Print summary
        print("\n" + "="*80)
        print("SPACECRAFT OPTIMIZATION EXAMPLE - SUMMARY")
        print("="*80)
        
        print(f"\nOptimization Algorithms Tested: {len(optimization_results['earth_observation'])}")
        print(f"Mission Types Analyzed: {len(optimization_results)}")
        print(f"Robust Optimization Measures: {len(robust_results)}")
        print(f"Fidelity Strategies Tested: {len(fidelity_results)}")
        
        print("\nBest Results by Mission Type:")
        for mission_type, mission_results in optimization_results.items():
            best_alg = max(mission_results.keys(), 
                          key=lambda alg: sum(mission_results[alg].best_objectives.values()))
            best_fitness = sum(mission_results[best_alg].best_objectives.values())
            print(f"  {mission_type.title()}: {best_alg.title()} (fitness: {best_fitness:.3f})")
        
        print(f"\nRobust Optimization Results:")
        for measure, result in robust_results.items():
            reliability = result.reliability_metrics.get('overall_reliability', 0.0)
            print(f"  {measure.title()}: Reliability = {reliability:.3f}")
        
        print(f"\nFidelity Strategy Performance:")
        for strategy_name, strategy_data in fidelity_results.items():
            total_time = strategy_data['optimization_result'].total_time
            best_fitness = sum(strategy_data['optimization_result'].best_objectives.values())
            print(f"  {strategy_name.title()}: Time = {total_time:.1f}s, Fitness = {best_fitness:.3f}")
        
        print(f"\nSensitivity Analysis:")
        for param_name in sensitivity_results.keys():
            print(f"  Analyzed sensitivity to {param_name.replace('_', ' ')}")
        
        print(f"\nGenerated Files:")
        if 'csv_file' in report_results:
            print(f"  CSV Export: {report_results['csv_file']}")
        if 'excel_file' in report_results:
            print(f"  Excel Report: {report_results['excel_file']}")
        if 'comparison_report' in report_results:
            print(f"  Comparison Report: {report_results['comparison_report']}")
        
        print(f"\nAnalyzed {report_results.get('spacecraft_runs_analyzed', 0)} spacecraft optimization runs")
        print(f"Generated {len(report_results.get('generated_plots', {}))} visualization plots")
        
        print("\n" + "="*80)
        print("Spacecraft optimization example completed successfully!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error in spacecraft optimization example: {e}")
        raise


if __name__ == "__main__":
    main()