#!/usr/bin/env python3
"""
Local Data Generator
Adaptive Multi-Fidelity Simulation-Based Optimization for Aerospace Systems

Replaces all external API calls with local data generation.
No network dependencies - completely offline operation.
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

class LocalDataGenerator:
    """Generate all data locally without external API dependencies."""
    
    def __init__(self, seed=42):
        """Initialize with reproducible random seed."""
        np.random.seed(seed)
        self.seed = seed
        
    def generate_optimization_results(self, algorithm='genetic_algorithm', problem_size=50, fidelity='medium'):
        """Generate realistic optimization results locally."""
        
        # Base performance parameters based on algorithm
        base_params = {
            'genetic_algorithm': {
                'convergence_time': 120.0,
                'solution_quality': 0.92,
                'success_rate': 0.94,
                'cost_reduction': 87.5
            },
            'particle_swarm_optimization': {
                'convergence_time': 85.0,
                'solution_quality': 0.89,
                'success_rate': 0.96,
                'cost_reduction': 89.2
            },
            'bayesian_optimization': {
                'convergence_time': 45.0,
                'solution_quality': 0.94,
                'success_rate': 0.98,
                'cost_reduction': 86.1
            }
        }
        
        # Fidelity modifiers
        fidelity_modifiers = {
            'low': {'time': 0.1, 'quality': 0.85, 'cost_reduction': 1.15},
            'medium': {'time': 1.0, 'quality': 0.95, 'cost_reduction': 1.05},
            'high': {'time': 8.5, 'quality': 1.0, 'cost_reduction': 1.0}
        }
        
        base = base_params.get(algorithm, base_params['genetic_algorithm'])
        modifier = fidelity_modifiers.get(fidelity, fidelity_modifiers['medium'])
        
        # Generate with realistic noise
        result = {
            'algorithm': algorithm,
            'problem_size': problem_size,
            'fidelity_level': fidelity,
            'convergence_time_seconds': base['convergence_time'] * modifier['time'] * (1 + np.random.normal(0, 0.1)),
            'solution_quality': min(1.0, base['solution_quality'] * modifier['quality'] * (1 + np.random.normal(0, 0.05))),
            'success_rate': min(1.0, base['success_rate'] * (1 + np.random.normal(0, 0.02))),
            'cost_reduction_percent': base['cost_reduction'] * modifier['cost_reduction'] * (1 + np.random.normal(0, 0.03)),
            'memory_usage_mb': problem_size * 4.2 * modifier['time'] * (1 + np.random.normal(0, 0.1)),
            'iterations_to_convergence': int(50 + problem_size * 2 + np.random.normal(0, 10)),
            'timestamp': datetime.now().isoformat()
        }
        
        # Ensure realistic bounds
        result['convergence_time_seconds'] = max(1.0, result['convergence_time_seconds'])
        result['solution_quality'] = max(0.5, min(1.0, result['solution_quality']))
        result['cost_reduction_percent'] = max(75.0, min(95.0, result['cost_reduction_percent']))
        result['memory_usage_mb'] = max(10.0, result['memory_usage_mb'])
        result['iterations_to_convergence'] = max(10, result['iterations_to_convergence'])
        
        return result
    
    def generate_aircraft_design_data(self, design_type='wing_optimization'):
        """Generate realistic aircraft design optimization data."""
        
        design_templates = {
            'wing_optimization': {
                'variables': ['wingspan_m', 'chord_length_m', 'sweep_angle_deg', 'thickness_ratio'],
                'objectives': ['lift_to_drag_ratio', 'structural_weight_kg'],
                'constraints': ['max_stress_mpa', 'flutter_speed_ms']
            },
            'fuselage_design': {
                'variables': ['length_m', 'diameter_m', 'fineness_ratio', 'cabin_volume_m3'],
                'objectives': ['drag_coefficient', 'structural_efficiency'],
                'constraints': ['pressure_differential_pa', 'fatigue_life_cycles']
            }
        }
        
        template = design_templates.get(design_type, design_templates['wing_optimization'])
        
        # Generate realistic design data
        design_data = {
            'design_type': design_type,
            'design_variables': {},
            'objective_values': {},
            'constraint_values': {},
            'optimization_history': [],
            'performance_metrics': {}
        }
        
        # Generate design variables
        for var in template['variables']:
            if 'wingspan' in var:
                design_data['design_variables'][var] = 35.0 + np.random.uniform(-5, 10)
            elif 'chord' in var:
                design_data['design_variables'][var] = 2.5 + np.random.uniform(-0.5, 1.0)
            elif 'angle' in var:
                design_data['design_variables'][var] = 25.0 + np.random.uniform(-10, 15)
            elif 'ratio' in var:
                design_data['design_variables'][var] = 0.12 + np.random.uniform(-0.05, 0.08)
            else:
                design_data['design_variables'][var] = 1.0 + np.random.uniform(-0.5, 2.0)
        
        # Generate objectives
        for obj in template['objectives']:
            if 'drag' in obj:
                design_data['objective_values'][obj] = 18.5 + np.random.uniform(-2, 8)
            elif 'weight' in obj:
                design_data['objective_values'][obj] = 2500 + np.random.uniform(-500, 1000)
            else:
                design_data['objective_values'][obj] = 1.0 + np.random.uniform(-0.3, 0.8)
        
        # Generate optimization history
        for i in range(50):
            iteration_data = {
                'iteration': i + 1,
                'best_objective': 20.0 - i * 0.2 + np.random.normal(0, 0.5),
                'computational_time': (i + 1) * 1.2 + np.random.uniform(0, 0.5),
                'fidelity_used': np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
            }
            design_data['optimization_history'].append(iteration_data)
        
        return design_data
    
    def generate_spacecraft_trajectory_data(self, mission_type='mars_transfer'):
        """Generate realistic spacecraft trajectory optimization data."""
        
        mission_templates = {
            'mars_transfer': {
                'departure_date': '2025-07-15',
                'flight_time_days': 260,
                'delta_v_budget_ms': 3800,
                'c3_energy_km2s2': 12.5
            },
            'lunar_mission': {
                'departure_date': '2025-03-10',
                'flight_time_days': 8,
                'delta_v_budget_ms': 3200,
                'c3_energy_km2s2': 8.2
            }
        }
        
        template = mission_templates.get(mission_type, mission_templates['mars_transfer'])
        
        trajectory_data = {
            'mission_type': mission_type,
            'trajectory_parameters': {},
            'performance_metrics': {},
            'optimization_results': {},
            'validation_data': {}
        }
        
        # Generate trajectory parameters with noise
        trajectory_data['trajectory_parameters'] = {
            'departure_date': template['departure_date'],
            'flight_time_days': template['flight_time_days'] + np.random.uniform(-10, 15),
            'departure_c3_km2s2': template['c3_energy_km2s2'] + np.random.uniform(-1, 2),
            'arrival_c3_km2s2': 2.5 + np.random.uniform(-0.5, 1.0),
            'total_delta_v_ms': template['delta_v_budget_ms'] + np.random.uniform(-200, 300)
        }
        
        # Generate performance metrics
        trajectory_data['performance_metrics'] = {
            'fuel_mass_fraction': 0.65 + np.random.uniform(-0.1, 0.15),
            'mission_success_probability': 0.94 + np.random.uniform(-0.05, 0.05),
            'trajectory_accuracy_km': 50 + np.random.uniform(-20, 30),
            'computational_efficiency': 0.87 + np.random.uniform(-0.1, 0.1)
        }
        
        return trajectory_data
    
    def generate_benchmark_suite(self, num_test_cases=100):
        """Generate comprehensive benchmark test suite."""
        
        benchmark_suite = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'total_test_cases': num_test_cases,
                'generator_version': '1.0.0',
                'seed_used': self.seed
            },
            'aircraft_benchmarks': [],
            'spacecraft_benchmarks': [],
            'mathematical_benchmarks': [],
            'performance_summary': {}
        }
        
        # Generate aircraft benchmarks
        for i in range(num_test_cases // 3):
            aircraft_data = self.generate_aircraft_design_data()
            aircraft_data['test_id'] = f"aircraft_{i+1:03d}"
            benchmark_suite['aircraft_benchmarks'].append(aircraft_data)
        
        # Generate spacecraft benchmarks
        for i in range(num_test_cases // 3):
            spacecraft_data = self.generate_spacecraft_trajectory_data()
            spacecraft_data['test_id'] = f"spacecraft_{i+1:03d}"
            benchmark_suite['spacecraft_benchmarks'].append(spacecraft_data)
        
        # Generate mathematical function benchmarks
        for i in range(num_test_cases - 2 * (num_test_cases // 3)):
            math_data = {
                'test_id': f"math_{i+1:03d}",
                'function_type': np.random.choice(['sphere', 'rosenbrock', 'ackley', 'rastrigin']),
                'dimension': np.random.choice([10, 20, 50, 100]),
                'global_optimum': 0.0,
                'search_bounds': [-10, 10],
                'optimization_result': self.generate_optimization_results()
            }
            benchmark_suite['mathematical_benchmarks'].append(math_data)
        
        return benchmark_suite
    
    def save_offline_dataset(self, output_dir='../data/generated'):
        """Save complete offline dataset for framework operation."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating comprehensive offline dataset...")
        
        # Generate benchmark suite
        benchmark_suite = self.generate_benchmark_suite(150)
        
        with open(output_path / 'offline_benchmark_suite.json', 'w') as f:
            json.dump(benchmark_suite, f, indent=2, default=str)
        
        # Generate optimization results for all algorithms
        algorithms = ['genetic_algorithm', 'particle_swarm_optimization', 'bayesian_optimization']
        fidelities = ['low', 'medium', 'high']
        problem_sizes = [25, 50, 100, 200]
        
        all_results = []
        for algorithm in algorithms:
            for fidelity in fidelities:
                for size in problem_sizes:
                    for _ in range(5):  # 5 runs per combination
                        result = self.generate_optimization_results(algorithm, size, fidelity)
                        all_results.append(result)
        
        # Save as CSV for easy analysis
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path / 'optimization_results_dataset.csv', index=False)
        
        # Generate aircraft design dataset
        aircraft_designs = []
        for i in range(50):
            design = self.generate_aircraft_design_data()
            aircraft_designs.append(design)
        
        with open(output_path / 'aircraft_design_dataset.json', 'w') as f:
            json.dump(aircraft_designs, f, indent=2, default=str)
        
        # Generate spacecraft mission dataset
        spacecraft_missions = []
        for i in range(30):
            mission = self.generate_spacecraft_trajectory_data()
            spacecraft_missions.append(mission)
        
        with open(output_path / 'spacecraft_mission_dataset.json', 'w') as f:
            json.dump(spacecraft_missions, f, indent=2, default=str)
        
        print(f"✓ Offline dataset generated: {output_path}")
        return output_path

def generate_offline_data():
    """Main function to generate all offline data."""
    generator = LocalDataGenerator()
    output_path = generator.save_offline_dataset()
    return output_path

if __name__ == "__main__":
    generate_offline_data()