#!/usr/bin/env python3
"""
Optimization Convergence Visualization
Adaptive Multi-Fidelity Simulation-Based Optimization for Aerospace Systems

Creates publication-ready convergence plots for all optimization algorithms
with professional aerospace color schemes and detailed metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

# Professional aerospace color scheme
AEROSPACE_COLORS = {
    'primary_blue': '#003366',
    'secondary_blue': '#0066CC', 
    'accent_orange': '#FF6600',
    'success_green': '#006633',
    'warning_amber': '#FF9900',
    'error_red': '#CC0000',
    'light_gray': '#E6E6E6',
    'dark_gray': '#666666',
    'background': '#F8F9FA'
}

# Algorithm-specific colors
ALGORITHM_COLORS = {
    'GeneticAlgorithm': AEROSPACE_COLORS['primary_blue'],
    'ParticleSwarmOptimization': AEROSPACE_COLORS['secondary_blue'],
    'BayesianOptimization': AEROSPACE_COLORS['accent_orange'],
    'NSGA2': AEROSPACE_COLORS['success_green']
}

class ConvergencePlotter:
    """Professional convergence visualization system for aerospace optimization."""
    
    def __init__(self, results_dir='../results', output_dir='./plots'):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set professional matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.facecolor': AEROSPACE_COLORS['background'],
            'figure.facecolor': 'white'
        })
    
    def load_optimization_data(self):
        """Load optimization results from JSON files."""
        aircraft_file = self.results_dir / 'aircraft_optimization_results.json'
        spacecraft_file = self.results_dir / 'spacecraft_optimization_results.json'
        
        with open(aircraft_file, 'r') as f:
            aircraft_data = json.load(f)
        
        with open(spacecraft_file, 'r') as f:
            spacecraft_data = json.load(f)
            
        return aircraft_data, spacecraft_data
    
    def plot_aircraft_convergence(self, aircraft_data):
        """Create convergence plots for all aircraft optimization runs."""
        designs = aircraft_data['optimization_results']['aircraft_designs']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Aircraft Optimization Convergence Analysis\nAdaptive Multi-Fidelity Framework', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        for idx, (design_name, design_data) in enumerate(designs.items()):
            if idx >= 5:  # Limit to 5 designs
                break
                
            row = idx // 3
            col = idx % 3
            ax = axes[row, col] if idx < 3 else axes[1, idx-3]
            
            convergence = design_data['optimization_results']['convergence_data']
            algorithm = design_data['optimization_results']['algorithm']
            
            # Plot convergence curve
            if 'generations' in convergence:
                x_data = convergence['generations']
                y_data = convergence['best_fitness']
                xlabel = 'Generation'
            elif 'iterations' in convergence:
                x_data = convergence['iterations']
                y_data = convergence['best_fitness']
                xlabel = 'Iteration'
            elif 'evaluations' in convergence:
                x_data = convergence['evaluations']
                y_data = convergence['best_fitness']
                xlabel = 'Evaluation'
            
            # Main convergence line
            ax.plot(x_data, y_data, 
                   color=ALGORITHM_COLORS.get(algorithm, AEROSPACE_COLORS['primary_blue']),
                   linewidth=2.5, marker='o', markersize=4, alpha=0.8,
                   label='Best L/D Ratio')
            
            # Add average fitness if available
            if 'average_fitness' in convergence:
                ax.plot(x_data, convergence['average_fitness'],
                       color=AEROSPACE_COLORS['dark_gray'], 
                       linewidth=1.5, linestyle='--', alpha=0.7,
                       label='Average L/D Ratio')
            
            # Highlight final performance
            final_ld = design_data['optimization_results']['final_performance']['lift_to_drag_ratio']
            ax.axhline(y=final_ld, color=AEROSPACE_COLORS['error_red'], 
                      linestyle=':', alpha=0.7, linewidth=1)
            
            # Formatting
            ax.set_title(f'{design_name.replace("_", " ").title()}\n{algorithm}', 
                        fontweight='bold', fontsize=11)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Lift-to-Drag Ratio')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Add performance metrics as text
            cost_savings = design_data['optimization_results']['computational_metrics']['cost_savings_vs_high_fidelity']
            ax.text(0.02, 0.98, f'Cost Savings: {cost_savings:.1%}', 
                   transform=ax.transAxes, fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=AEROSPACE_COLORS['light_gray'], alpha=0.8),
                   verticalalignment='top')
        
        # Remove empty subplot
        if len(designs) < 6:
            axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'aircraft_convergence_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'aircraft_convergence_analysis.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_spacecraft_convergence(self, spacecraft_data):
        """Create convergence plots for spacecraft optimization runs."""
        missions = spacecraft_data['spacecraft_optimization_results']['spacecraft_missions']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Spacecraft Mission Optimization Convergence\nMulti-Fidelity Adaptive Framework', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        for idx, (mission_name, mission_data) in enumerate(missions.items()):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            convergence = mission_data['optimization_results']['convergence_data']
            algorithm = mission_data['optimization_results']['algorithm']
            
            # Handle different data structures
            if 'best_mission_success' in convergence:
                if 'evaluations' in convergence:
                    x_data = convergence['evaluations']
                    xlabel = 'Evaluation'
                elif 'iterations' in convergence:
                    x_data = convergence['iterations']
                    xlabel = 'Iteration'
                y_data = convergence['best_mission_success']
                ylabel = 'Mission Success Probability'
            elif 'hypervolume' in convergence:
                x_data = convergence['generations']
                y_data = convergence['hypervolume']
                xlabel = 'Generation'
                ylabel = 'Hypervolume'
            elif 'best_delta_v' in convergence:
                x_data = convergence['generations']
                y_data = convergence['best_delta_v']
                xlabel = 'Generation'
                ylabel = 'Delta-V Capability (m/s)'
            
            # Main convergence plot
            ax.plot(x_data, y_data,
                   color=ALGORITHM_COLORS.get(algorithm, AEROSPACE_COLORS['secondary_blue']),
                   linewidth=2.5, marker='s', markersize=4, alpha=0.8,
                   label='Objective Value')
            
            # Add secondary metrics if available
            if 'delta_v_capability' in convergence:
                ax2 = ax.twinx()
                ax2.plot(x_data, convergence['delta_v_capability'],
                        color=AEROSPACE_COLORS['accent_orange'],
                        linewidth=2, linestyle='--', alpha=0.7)
                ax2.set_ylabel('Delta-V (m/s)', color=AEROSPACE_COLORS['accent_orange'])
                ax2.tick_params(axis='y', labelcolor=AEROSPACE_COLORS['accent_orange'])
            
            # Formatting
            mission_title = mission_name.replace('_', ' ').title()
            ax.set_title(f'{mission_title}\n{algorithm}', fontweight='bold', fontsize=11)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            # Add cost savings info
            cost_savings = mission_data['optimization_results']['computational_metrics']['cost_savings_vs_high_fidelity']
            ax.text(0.02, 0.98, f'Cost Savings: {cost_savings:.1%}', 
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=AEROSPACE_COLORS['light_gray'], alpha=0.8),
                   verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'spacecraft_convergence_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'spacecraft_convergence_analysis.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_algorithm_comparison(self, aircraft_data, spacecraft_data):
        """Create comprehensive algorithm comparison visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Optimization Algorithm Performance Comparison\nAerospace Multi-Fidelity Framework', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # Extract convergence data for each algorithm
        algorithm_performance = {}
        
        # Aircraft data
        for design_name, design_data in aircraft_data['optimization_results']['aircraft_designs'].items():
            algorithm = design_data['optimization_results']['algorithm']
            convergence = design_data['optimization_results']['convergence_data']
            
            if algorithm not in algorithm_performance:
                algorithm_performance[algorithm] = {'aircraft': [], 'spacecraft': []}
            
            if 'generations' in convergence:
                algorithm_performance[algorithm]['aircraft'].append({
                    'x': convergence['generations'],
                    'y': convergence['best_fitness'],
                    'name': design_name
                })
        
        # Spacecraft data
        for mission_name, mission_data in spacecraft_data['spacecraft_optimization_results']['spacecraft_missions'].items():
            algorithm = mission_data['optimization_results']['algorithm']
            convergence = mission_data['optimization_results']['convergence_data']
            
            if algorithm not in algorithm_performance:
                algorithm_performance[algorithm] = {'aircraft': [], 'spacecraft': []}
            
            if 'best_mission_success' in convergence:
                if 'evaluations' in convergence:
                    x_data = convergence['evaluations']
                else:
                    x_data = convergence['iterations']
                algorithm_performance[algorithm]['spacecraft'].append({
                    'x': x_data,
                    'y': convergence['best_mission_success'],
                    'name': mission_name
                })
        
        # Plot aircraft algorithm comparison
        ax = axes[0, 0]
        for algorithm, data in algorithm_performance.items():
            for run in data['aircraft']:
                ax.plot(run['x'], run['y'], 
                       color=ALGORITHM_COLORS.get(algorithm, AEROSPACE_COLORS['primary_blue']),
                       alpha=0.7, linewidth=1.5, label=algorithm if run == data['aircraft'][0] else "")
        
        ax.set_title('Aircraft Optimization Convergence', fontweight='bold')
        ax.set_xlabel('Generation/Iteration')
        ax.set_ylabel('Lift-to-Drag Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot spacecraft algorithm comparison
        ax = axes[0, 1]
        for algorithm, data in algorithm_performance.items():
            for run in data['spacecraft']:
                ax.plot(run['x'], run['y'],
                       color=ALGORITHM_COLORS.get(algorithm, AEROSPACE_COLORS['secondary_blue']),
                       alpha=0.7, linewidth=1.5, label=algorithm if run == data['spacecraft'][0] else "")
        
        ax.set_title('Spacecraft Mission Optimization', fontweight='bold')
        ax.set_xlabel('Evaluation/Iteration')
        ax.set_ylabel('Mission Success Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Performance summary statistics
        ax = axes[1, 0]
        algorithms = list(algorithm_performance.keys())
        aircraft_final_values = []
        spacecraft_final_values = []
        
        for alg in algorithms:
            # Aircraft final values
            if algorithm_performance[alg]['aircraft']:
                final_vals = [run['y'][-1] for run in algorithm_performance[alg]['aircraft']]
                aircraft_final_values.append(np.mean(final_vals))
            else:
                aircraft_final_values.append(0)
            
            # Spacecraft final values
            if algorithm_performance[alg]['spacecraft']:
                final_vals = [run['y'][-1] for run in algorithm_performance[alg]['spacecraft']]
                spacecraft_final_values.append(np.mean(final_vals))
            else:
                spacecraft_final_values.append(0)
        
        x_pos = np.arange(len(algorithms))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, aircraft_final_values, width, 
                      label='Aircraft L/D Ratio', color=AEROSPACE_COLORS['primary_blue'], alpha=0.8)
        
        ax2 = ax.twinx()
        bars2 = ax2.bar(x_pos + width/2, spacecraft_final_values, width,
                       label='Mission Success Rate', color=AEROSPACE_COLORS['accent_orange'], alpha=0.8)
        
        ax.set_xlabel('Optimization Algorithm')
        ax.set_ylabel('Average L/D Ratio', color=AEROSPACE_COLORS['primary_blue'])
        ax2.set_ylabel('Average Success Rate', color=AEROSPACE_COLORS['accent_orange'])
        ax.set_title('Algorithm Performance Summary', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([alg.replace('Optimization', '').replace('Algorithm', '') for alg in algorithms], rotation=45)
        
        # Cost savings comparison
        ax = axes[1, 1]
        cost_savings_data = []
        labels = []
        
        for design_name, design_data in aircraft_data['optimization_results']['aircraft_designs'].items():
            cost_savings = design_data['optimization_results']['computational_metrics']['cost_savings_vs_high_fidelity']
            cost_savings_data.append(cost_savings * 100)
            labels.append(design_name.split('_')[0].title())
        
        bars = ax.bar(labels, cost_savings_data, 
                     color=[AEROSPACE_COLORS['success_green'] if x > 85 else AEROSPACE_COLORS['warning_amber'] for x in cost_savings_data],
                     alpha=0.8)
        
        ax.set_title('Computational Cost Savings by Design', fontweight='bold')
        ax.set_ylabel('Cost Savings (%)')
        ax.set_ylim(0, 100)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, cost_savings_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'algorithm_comparison_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'algorithm_comparison_analysis.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_convergence_summary(self, aircraft_data, spacecraft_data):
        """Generate a comprehensive convergence summary report."""
        summary_data = {
            'Aircraft Optimization': {
                'Total Designs': len(aircraft_data['optimization_results']['aircraft_designs']),
                'Average L/D Ratio': aircraft_data['optimization_results']['summary_statistics']['average_ld_ratio'],
                'Overall Cost Savings': f"{aircraft_data['optimization_results']['summary_statistics']['overall_cost_savings']:.1%}",
                'Success Rate': f"{aircraft_data['optimization_results']['summary_statistics']['success_rate']:.1%}"
            },
            'Spacecraft Optimization': {
                'Total Missions': len(spacecraft_data['spacecraft_optimization_results']['spacecraft_missions']),
                'Average Success Rate': f"{spacecraft_data['spacecraft_optimization_results']['summary_statistics']['average_mission_success']:.3f}",
                'Overall Cost Savings': f"{spacecraft_data['spacecraft_optimization_results']['summary_statistics']['overall_cost_savings']:.1%}",
                'Total Evaluations': spacecraft_data['spacecraft_optimization_results']['summary_statistics']['total_function_evaluations']
            }
        }
        
        # Save summary as JSON
        with open(self.output_dir / 'convergence_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Convergence analysis complete!")
        print(f"Generated plots saved to: {self.output_dir}")
        print(f"Summary statistics:")
        for category, metrics in summary_data.items():
            print(f"\n{category}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

def main():
    """Main function to generate all convergence visualizations."""
    plotter = ConvergencePlotter()
    
    # Load optimization data
    aircraft_data, spacecraft_data = plotter.load_optimization_data()
    
    print("Generating optimization convergence visualizations...")
    
    # Generate all plots
    plotter.plot_aircraft_convergence(aircraft_data)
    print("✓ Aircraft convergence plots generated")
    
    plotter.plot_spacecraft_convergence(spacecraft_data)
    print("✓ Spacecraft convergence plots generated")
    
    plotter.plot_algorithm_comparison(aircraft_data, spacecraft_data)
    print("✓ Algorithm comparison plots generated")
    
    plotter.generate_convergence_summary(aircraft_data, spacecraft_data)
    print("✓ Convergence summary generated")

if __name__ == "__main__":
    main()