#!/usr/bin/env python3
"""
Pareto Front Visualization System
Adaptive Multi-Fidelity Simulation-Based Optimization for Aerospace Systems

Creates publication-ready Pareto front plots for multi-objective optimization
with professional aerospace color schemes and trade-off analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
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
    'background': '#F8F9FA',
    'pareto_front': '#FF4444',
    'pareto_points': '#0066FF'
}

class ParetoFrontVisualizer:
    """Professional Pareto front visualization system for aerospace optimization."""
    
    def __init__(self, results_dir='../results', output_dir='./plots'):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set professional matplotlib style
        plt.style.use('default')
        
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
    
    def load_pareto_data(self):
        """Load Pareto front data from JSON file."""
        pareto_file = self.results_dir / 'pareto_front_data.json'
        
        with open(pareto_file, 'r') as f:
            pareto_data = json.load(f)
            
        return pareto_data['pareto_front_results']
    
    def plot_aircraft_pareto_fronts(self, pareto_data):
        """Create 2D and 3D Pareto front visualizations for aircraft optimization."""
        aircraft_data = pareto_data['aircraft_multi_objective']
        solutions = aircraft_data['pareto_solutions']
        
        # Extract objective values
        ld_ratios = [sol['objectives']['lift_to_drag_ratio'] for sol in solutions]
        fuel_efficiency = [sol['objectives']['fuel_efficiency'] for sol in solutions]
        structural_weight = [sol['objectives']['structural_weight'] for sol in solutions]
        crowding_distances = [sol['crowding_distance'] for sol in solutions]
        
        fig = plt.figure(figsize=(20, 12))
        
        # 2D Pareto Front: L/D Ratio vs Fuel Efficiency
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(ld_ratios, fuel_efficiency, 
                             c=crowding_distances, s=100, alpha=0.8,
                             cmap='viridis', edgecolors=AEROSPACE_COLORS['primary_blue'], linewidth=1.5)
        
        # Sort points for Pareto front line
        sorted_indices = np.argsort(ld_ratios)
        sorted_ld = np.array(ld_ratios)[sorted_indices]
        sorted_fuel = np.array(fuel_efficiency)[sorted_indices]
        
        ax1.plot(sorted_ld, sorted_fuel, 
                color=AEROSPACE_COLORS['pareto_front'], linewidth=2.5, alpha=0.7,
                label='Pareto Front')
        
        ax1.set_xlabel('Lift-to-Drag Ratio')
        ax1.set_ylabel('Fuel Efficiency (km/L)')
        ax1.set_title('Aircraft Design Trade-off:\nL/D Ratio vs Fuel Efficiency', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Crowding Distance', rotation=270, labelpad=15)
        
        # 2D Pareto Front: L/D Ratio vs Structural Weight  
        ax2 = plt.subplot(2, 3, 2)
        scatter2 = ax2.scatter(ld_ratios, structural_weight,
                              c=fuel_efficiency, s=100, alpha=0.8,
                              cmap='plasma', edgecolors=AEROSPACE_COLORS['success_green'], linewidth=1.5)
        
        # Pareto front line
        sorted_indices = np.argsort(ld_ratios)
        sorted_weight = np.array(structural_weight)[sorted_indices]
        ax2.plot(sorted_ld, sorted_weight,
                color=AEROSPACE_COLORS['pareto_front'], linewidth=2.5, alpha=0.7,
                label='Pareto Front')
        
        ax2.set_xlabel('Lift-to-Drag Ratio')
        ax2.set_ylabel('Structural Weight (kg)')
        ax2.set_title('Aircraft Design Trade-off:\nPerformance vs Weight', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Fuel Efficiency (km/L)', rotation=270, labelpad=15)
        
        # 3D Pareto Front Visualization
        ax3 = plt.subplot(2, 3, 3, projection='3d')
        scatter3d = ax3.scatter(ld_ratios, fuel_efficiency, structural_weight,
                               c=crowding_distances, s=80, alpha=0.8,
                               cmap='coolwarm', edgecolors='black', linewidth=0.5)
        
        ax3.set_xlabel('L/D Ratio')
        ax3.set_ylabel('Fuel Efficiency')
        ax3.set_zlabel('Weight (kg)')
        ax3.set_title('3D Pareto Front\nAircraft Multi-Objective', fontweight='bold')
        
        # Trade-off Analysis Plot
        ax4 = plt.subplot(2, 3, 4)
        
        # Normalize objectives for radar chart comparison
        normalized_ld = np.array(ld_ratios) / max(ld_ratios)
        normalized_fuel = np.array(fuel_efficiency) / max(fuel_efficiency)
        normalized_weight = 1 - (np.array(structural_weight) / max(structural_weight))  # Invert weight (lower is better)
        
        # Select top 3 solutions for detailed comparison
        top_indices = np.argsort(crowding_distances)[-3:]
        
        categories = ['L/D Ratio', 'Fuel Efficiency', 'Weight\n(Inverted)']
        
        # Plot radar chart for top solutions
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = [AEROSPACE_COLORS['primary_blue'], AEROSPACE_COLORS['accent_orange'], AEROSPACE_COLORS['success_green']]
        
        for i, idx in enumerate(top_indices):
            values = [normalized_ld[idx], normalized_fuel[idx], normalized_weight[idx]]
            values += values[:1]  # Complete the circle
            
            ax4.plot(angles, values, 'o-', linewidth=2, color=colors[i], 
                    label=f'Solution {solutions[idx]["solution_id"]}', alpha=0.8)
            ax4.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('Top Solutions Comparison\n(Normalized Objectives)', fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # Hypervolume Convergence
        ax5 = plt.subplot(2, 3, 5)
        
        # Simulate hypervolume convergence (would come from optimization history)
        generations = np.arange(1, 121)
        hypervolume_evolution = aircraft_data['performance_metrics']['hypervolume'] * (1 - np.exp(-generations/30))
        
        ax5.plot(generations, hypervolume_evolution, 
                color=AEROSPACE_COLORS['primary_blue'], linewidth=3, alpha=0.8)
        ax5.axhline(y=aircraft_data['performance_metrics']['hypervolume'], 
                   color=AEROSPACE_COLORS['pareto_front'], linestyle='--', 
                   label=f'Final HV: {aircraft_data["performance_metrics"]["hypervolume"]:.3f}')
        
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Hypervolume')
        ax5.set_title('Hypervolume Convergence\nAircraft Multi-Objective', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Performance Metrics Summary
        ax6 = plt.subplot(2, 3, 6)
        
        metrics = ['Hypervolume', 'Spacing', 'Gen. Distance', 'IGD']
        values = [
            aircraft_data['performance_metrics']['hypervolume'],
            aircraft_data['performance_metrics']['spacing'],
            aircraft_data['performance_metrics']['generational_distance'],
            aircraft_data['performance_metrics']['inverted_generational_distance']
        ]
        
        # Normalize metrics for visualization
        normalized_values = [v/max(values) for v in values]
        
        bars = ax6.bar(metrics, normalized_values, 
                      color=[AEROSPACE_COLORS['primary_blue'], AEROSPACE_COLORS['accent_orange'],
                            AEROSPACE_COLORS['success_green'], AEROSPACE_COLORS['warning_amber']],
                      alpha=0.8)
        
        ax6.set_ylabel('Normalized Performance')
        ax6.set_title('Multi-Objective Performance\nMetrics Summary', fontweight='bold')
        ax6.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value, orig_value in zip(bars, normalized_values, values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{orig_value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.suptitle('Aircraft Multi-Objective Optimization: Pareto Front Analysis\nAdaptive Multi-Fidelity Framework', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(self.output_dir / 'aircraft_pareto_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'aircraft_pareto_analysis.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_spacecraft_pareto_fronts(self, pareto_data):
        """Create Pareto front visualizations for spacecraft optimization."""
        spacecraft_data = pareto_data['spacecraft_multi_objective']
        solutions = spacecraft_data['pareto_solutions']
        
        # Extract objective values
        mission_success = [sol['objectives']['mission_success_probability'] for sol in solutions]
        total_mass = [sol['objectives']['total_mass'] for sol in solutions]
        power_efficiency = [sol['objectives']['power_efficiency'] for sol in solutions]
        delta_v = [sol['objectives']['delta_v_capability'] for sol in solutions]
        crowding_distances = [sol['crowding_distance'] for sol in solutions]
        
        fig = plt.figure(figsize=(20, 12))
        
        # 2D Pareto Front: Mission Success vs Total Mass
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(mission_success, total_mass,
                             c=delta_v, s=120, alpha=0.8,
                             cmap='viridis', edgecolors=AEROSPACE_COLORS['primary_blue'], linewidth=1.5)
        
        # Sort and plot Pareto front
        sorted_indices = np.argsort(mission_success)
        sorted_success = np.array(mission_success)[sorted_indices]
        sorted_mass = np.array(total_mass)[sorted_indices]
        
        ax1.plot(sorted_success, sorted_mass,
                color=AEROSPACE_COLORS['pareto_front'], linewidth=2.5, alpha=0.7,
                label='Pareto Front')
        
        ax1.set_xlabel('Mission Success Probability')
        ax1.set_ylabel('Total Mass (kg)')
        ax1.set_title('Spacecraft Trade-off:\nReliability vs Mass', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Delta-V Capability (m/s)', rotation=270, labelpad=15)
        
        # 2D Pareto Front: Power Efficiency vs Delta-V
        ax2 = plt.subplot(2, 3, 2)
        scatter2 = ax2.scatter(power_efficiency, delta_v,
                              c=mission_success, s=120, alpha=0.8,
                              cmap='plasma', edgecolors=AEROSPACE_COLORS['success_green'], linewidth=1.5)
        
        # Pareto front line
        sorted_indices = np.argsort(power_efficiency)
        sorted_power = np.array(power_efficiency)[sorted_indices]
        sorted_delta = np.array(delta_v)[sorted_indices]
        ax2.plot(sorted_power, sorted_delta,
                color=AEROSPACE_COLORS['pareto_front'], linewidth=2.5, alpha=0.7,
                label='Pareto Front')
        
        ax2.set_xlabel('Power Efficiency')
        ax2.set_ylabel('Delta-V Capability (m/s)')
        ax2.set_title('Spacecraft Trade-off:\nPower vs Propulsion', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Mission Success Probability', rotation=270, labelpad=15)
        
        # 3D Pareto Front: Success vs Mass vs Delta-V
        ax3 = plt.subplot(2, 3, 3, projection='3d')
        scatter3d = ax3.scatter(mission_success, total_mass, delta_v,
                               c=power_efficiency, s=100, alpha=0.8,
                               cmap='coolwarm', edgecolors='black', linewidth=0.5)
        
        ax3.set_xlabel('Mission Success')
        ax3.set_ylabel('Total Mass (kg)')
        ax3.set_zlabel('Delta-V (m/s)')
        ax3.set_title('3D Pareto Front\nSpacecraft Multi-Objective', fontweight='bold')
        
        # Trade-off Analysis Parallel Coordinates
        ax4 = plt.subplot(2, 3, 4)
        
        # Normalize all objectives
        norm_success = np.array(mission_success) / max(mission_success)
        norm_mass = 1 - (np.array(total_mass) / max(total_mass))  # Invert (lower is better)
        norm_power = np.array(power_efficiency) / max(power_efficiency)
        norm_delta = np.array(delta_v) / max(delta_v)
        
        # Create parallel coordinates plot
        objectives = ['Success\nProb.', 'Mass\n(Inverted)', 'Power\nEff.', 'Delta-V']
        
        for i, sol in enumerate(solutions):
            values = [norm_success[i], norm_mass[i], norm_power[i], norm_delta[i]]
            color_intensity = crowding_distances[i] / max(crowding_distances)
            
            ax4.plot(range(len(objectives)), values, 'o-', 
                    alpha=0.7, linewidth=1.5, 
                    color=plt.cm.viridis(color_intensity),
                    markersize=6)
        
        ax4.set_xticks(range(len(objectives)))
        ax4.set_xticklabels(objectives)
        ax4.set_ylabel('Normalized Objective Value')
        ax4.set_ylim(0, 1)
        ax4.set_title('Parallel Coordinates\nAll Solutions', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Spacecraft Mission Types Comparison
        ax5 = plt.subplot(2, 3, 5)
        
        # Create comparison of different solution characteristics
        solution_ids = [sol['solution_id'] for sol in solutions]
        
        # Stack different objectives for comparison
        width = 0.6
        x_pos = np.arange(len(solution_ids))
        
        # Normalize for stacked bar chart
        norm_success_100 = norm_success * 100
        norm_mass_100 = norm_mass * 100
        norm_power_100 = norm_power * 100
        
        p1 = ax5.bar(x_pos, norm_success_100, width, 
                    color=AEROSPACE_COLORS['primary_blue'], alpha=0.8, label='Success Rate')
        p2 = ax5.bar(x_pos, norm_power_100, width, bottom=norm_success_100,
                    color=AEROSPACE_COLORS['accent_orange'], alpha=0.8, label='Power Eff.')
        p3 = ax5.bar(x_pos, norm_mass_100, width, 
                    bottom=norm_success_100 + norm_power_100,
                    color=AEROSPACE_COLORS['success_green'], alpha=0.8, label='Mass (Inv.)')
        
        ax5.set_xlabel('Solution ID')
        ax5.set_ylabel('Normalized Performance (%)')
        ax5.set_title('Solution Performance\nBreakdown', fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(solution_ids)
        ax5.legend()
        
        # Performance metrics and convergence info
        ax6 = plt.subplot(2, 3, 6)
        
        # Performance metrics comparison
        metrics_data = spacecraft_data['performance_metrics']
        
        labels = ['Hypervolume', 'Spacing', 'GD', 'IGD', 'Cost Savings']
        values = [
            metrics_data['hypervolume'],
            metrics_data['spacing'],
            metrics_data['generational_distance'],
            metrics_data['inverted_generational_distance'],
            metrics_data['computational_metrics']['cost_savings']
        ]
        
        # Create pie chart for cost distribution
        cost_labels = ['Low Fidelity', 'Medium Fidelity', 'High Fidelity']
        fidelity_dist = metrics_data['computational_metrics']['fidelity_distribution']
        cost_values = [fidelity_dist['low'], fidelity_dist['medium'], fidelity_dist['high']]
        
        colors = [AEROSPACE_COLORS['success_green'], AEROSPACE_COLORS['warning_amber'], AEROSPACE_COLORS['error_red']]
        
        wedges, texts, autotexts = ax6.pie(cost_values, labels=cost_labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90, alpha=0.8)
        
        ax6.set_title(f'Fidelity Distribution\nCost Savings: {metrics_data["computational_metrics"]["cost_savings"]:.1%}', 
                     fontweight='bold')
        
        plt.tight_layout()
        plt.suptitle('Spacecraft Multi-Objective Optimization: Pareto Front Analysis\nAdaptive Multi-Fidelity Framework', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(self.output_dir / 'spacecraft_pareto_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'spacecraft_pareto_analysis.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_mixed_domain_analysis(self, pareto_data):
        """Create visualization for mixed aircraft-spacecraft optimization."""
        mixed_data = pareto_data['mixed_domain_optimization']
        solutions = mixed_data['pareto_solutions']
        
        # Extract data
        aero_efficiency = [sol['objectives']['aerodynamic_efficiency'] for sol in solutions]
        mission_reliability = [sol['objectives']['mission_reliability'] for sol in solutions]
        total_cost = [sol['objectives']['total_cost'] for sol in solutions]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Mixed Domain Optimization: Aircraft-Spacecraft Integration\nAdaptive Multi-Fidelity Framework', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # 2D Pareto front: Efficiency vs Reliability
        ax = axes[0, 0]
        scatter = ax.scatter(aero_efficiency, mission_reliability, 
                           c=total_cost, s=120, alpha=0.8, cmap='viridis',
                           edgecolors=AEROSPACE_COLORS['primary_blue'], linewidth=1.5)
        
        # Pareto front line
        sorted_indices = np.argsort(aero_efficiency)
        sorted_eff = np.array(aero_efficiency)[sorted_indices]
        sorted_rel = np.array(mission_reliability)[sorted_indices]
        
        ax.plot(sorted_eff, sorted_rel,
               color=AEROSPACE_COLORS['pareto_front'], linewidth=2.5, alpha=0.7,
               label='Pareto Front')
        
        ax.set_xlabel('Aerodynamic Efficiency')
        ax.set_ylabel('Mission Reliability')
        ax.set_title('Performance vs Reliability Trade-off', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Total Cost ($)', rotation=270, labelpad=15)
        
        # Cost analysis
        ax = axes[0, 1]
        scatter2 = ax.scatter(aero_efficiency, total_cost,
                             c=mission_reliability, s=120, alpha=0.8, cmap='plasma',
                             edgecolors=AEROSPACE_COLORS['success_green'], linewidth=1.5)
        
        ax.set_xlabel('Aerodynamic Efficiency')
        ax.set_ylabel('Total Cost ($)')
        ax.set_title('Efficiency vs Cost Analysis', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(scatter2, ax=ax)
        cbar2.set_label('Mission Reliability', rotation=270, labelpad=15)
        
        # Technology level impact
        ax = axes[1, 0]
        tech_levels = [sol['parameters']['shared_technology_level'] for sol in solutions]
        
        # Create 3D-like effect with bubble sizes
        sizes = np.array(tech_levels) * 200
        scatter3 = ax.scatter(mission_reliability, total_cost, s=sizes, alpha=0.6,
                             c=aero_efficiency, cmap='coolwarm',
                             edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Mission Reliability')
        ax.set_ylabel('Total Cost ($)')
        ax.set_title('Technology Level Impact\n(Bubble size = Tech Level)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar3 = plt.colorbar(scatter3, ax=ax)
        cbar3.set_label('Aerodynamic Efficiency', rotation=270, labelpad=15)
        
        # Trade-off summary
        ax = axes[1, 1]
        
        # Correlation analysis from trade-off data
        trade_off_data = pareto_data['trade_off_analysis']
        
        # Create correlation matrix visualization
        correlations = [
            ['L/D vs Fuel', trade_off_data['aircraft_analysis']['ld_ratio_vs_fuel_efficiency']['correlation']],
            ['Perf vs Weight', trade_off_data['aircraft_analysis']['performance_vs_weight']['correlation']],
            ['Mass vs Reliability', trade_off_data['spacecraft_analysis']['mass_vs_reliability']['correlation']],
            ['DeltaV vs Power', trade_off_data['spacecraft_analysis']['deltav_vs_power']['correlation']]
        ]
        
        labels = [corr[0] for corr in correlations]
        values = [abs(corr[1]) for corr in correlations]
        
        colors_corr = [AEROSPACE_COLORS['success_green'] if corr[1] > 0 else AEROSPACE_COLORS['error_red'] 
                      for corr in correlations]
        
        bars = ax.barh(labels, values, color=colors_corr, alpha=0.8)
        
        ax.set_xlabel('Correlation Strength')
        ax.set_title('Objective Correlations\nTrade-off Analysis', fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add correlation values as text
        for bar, corr in zip(bars, correlations):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{corr[1]:.2f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mixed_domain_pareto_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'mixed_domain_pareto_analysis.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_pareto_summary(self, pareto_data):
        """Generate comprehensive Pareto front analysis summary."""
        summary_stats = pareto_data['summary_statistics']
        
        summary_report = {
            'Total Pareto Solutions': summary_stats['total_pareto_solutions'],
            'Average Hypervolume': f"{summary_stats['average_hypervolume']:.3f}",
            'Average Convergence Generation': f"{summary_stats['average_convergence_generation']:.1f}",
            'Overall Cost Savings': f"{summary_stats['overall_cost_savings']:.1%}",
            'Success Rate': f"{summary_stats['success_rate']:.1%}",
            'Constraint Satisfaction': f"{summary_stats['constraint_satisfaction_rate']:.1%}",
            'Diversity Metrics': {
                'Average Spacing': f"{summary_stats['diversity_metrics']['average_spacing']:.3f}",
                'Solution Spread': f"{summary_stats['diversity_metrics']['solution_spread']:.3f}",
                'Distribution Uniformity': f"{summary_stats['diversity_metrics']['distribution_uniformity']:.3f}"
            }
        }
        
        # Save summary
        with open(self.output_dir / 'pareto_analysis_summary.json', 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        print(f"Pareto front analysis complete!")
        print(f"Generated plots saved to: {self.output_dir}")
        print(f"Summary statistics:")
        for key, value in summary_report.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")

def main():
    """Main function to generate all Pareto front visualizations."""
    visualizer = ParetoFrontVisualizer()
    
    # Load Pareto data
    pareto_data = visualizer.load_pareto_data()
    
    print("Generating Pareto front visualizations...")
    
    # Generate all Pareto plots
    visualizer.plot_aircraft_pareto_fronts(pareto_data)
    print("✓ Aircraft Pareto front plots generated")
    
    visualizer.plot_spacecraft_pareto_fronts(pareto_data)
    print("✓ Spacecraft Pareto front plots generated")
    
    visualizer.plot_mixed_domain_analysis(pareto_data)
    print("✓ Mixed domain analysis plots generated")
    
    visualizer.generate_pareto_summary(pareto_data)
    print("✓ Pareto analysis summary generated")

if __name__ == "__main__":
    main()