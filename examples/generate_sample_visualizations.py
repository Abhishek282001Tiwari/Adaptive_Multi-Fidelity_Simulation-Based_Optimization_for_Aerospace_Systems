#!/usr/bin/env python3
"""
Sample Visualization Generator
==============================

This script generates sample visualization outputs that would be produced by the 
optimization framework. These serve as examples of the types of plots and analysis
charts that the system generates.

This script can run independently of the main framework to demonstrate
the visualization capabilities.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory
os.makedirs('visualizations', exist_ok=True)

def generate_convergence_plot():
    """Generate sample convergence plot for aircraft optimization"""
    
    # Sample convergence data
    generations = np.arange(0, 50)
    
    # Simulate optimization convergence with noise
    best_fitness = 12.0 + 16.0 * (1 - np.exp(-generations/8)) + np.random.normal(0, 0.5, len(generations))
    best_fitness = np.maximum.accumulate(best_fitness)  # Ensure monotonic improvement
    
    # Add some algorithm comparison
    avg_fitness = best_fitness - np.random.uniform(1, 4, len(generations))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness', marker='o', markersize=3)
    ax.plot(generations, avg_fitness, 'r--', linewidth=1.5, label='Population Average', alpha=0.7)
    
    ax.fill_between(generations, avg_fitness, best_fitness, alpha=0.2, color='blue')
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Lift-to-Drag Ratio', fontsize=12)
    ax.set_title('Aircraft Optimization Convergence\nGenetic Algorithm - Commercial Airliner', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add performance annotations
    ax.annotate(f'Final L/D: {best_fitness[-1]:.2f}', 
                xy=(generations[-1], best_fitness[-1]), 
                xytext=(generations[-5], best_fitness[-1]+1),
                fontsize=10, ha='right',
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.6))
    
    plt.tight_layout()
    plt.savefig('visualizations/aircraft_convergence_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: aircraft_convergence_sample.png")

def generate_pareto_front():
    """Generate sample Pareto front for multi-objective optimization"""
    
    # Generate sample Pareto front data
    n_solutions = 25
    
    # Create realistic Pareto front shape
    f1 = np.linspace(15, 28, n_solutions)  # L/D ratio
    f2 = 5.2 - 0.08 * f1 + 0.001 * f1**2 + np.random.normal(0, 0.05, n_solutions)  # Fuel efficiency
    
    # Add some dominated solutions for comparison
    n_dominated = 50
    f1_dominated = np.random.uniform(12, 30, n_dominated)
    f2_dominated = np.random.uniform(2.8, 5.5, n_dominated)
    
    # Filter out non-dominated from the "dominated" set
    dominated_mask = []
    for i in range(n_dominated):
        is_dominated = False
        for j in range(n_solutions):
            if f1[j] >= f1_dominated[i] and f2[j] <= f2_dominated[i]:
                if f1[j] > f1_dominated[i] or f2[j] < f2_dominated[i]:
                    is_dominated = True
                    break
        dominated_mask.append(is_dominated)
    
    f1_dominated = f1_dominated[dominated_mask]
    f2_dominated = f2_dominated[dominated_mask]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot dominated solutions
    ax.scatter(f1_dominated, f2_dominated, c='lightgray', alpha=0.6, s=30, label='Dominated Solutions')
    
    # Plot Pareto front
    ax.scatter(f1, f2, c='red', s=80, label='Pareto Optimal Solutions', edgecolor='darkred', linewidth=1, zorder=5)
    ax.plot(f1, f2, 'r-', alpha=0.8, linewidth=2, zorder=4)
    
    # Highlight some specific solutions
    knee_idx = len(f1)//2
    ax.scatter(f1[knee_idx], f2[knee_idx], c='gold', s=150, marker='*', 
               label='Knee Point Solution', edgecolor='orange', linewidth=2, zorder=6)
    
    ax.set_xlabel('Lift-to-Drag Ratio', fontsize=12)
    ax.set_ylabel('Fuel Efficiency (L/100km/pax)', fontsize=12)
    ax.set_title('Multi-Objective Aircraft Optimization\nPareto Front: L/D Ratio vs Fuel Efficiency', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key solutions
    ax.annotate(f'Best Efficiency\nL/D: {f1[0]:.1f}\nFuel: {f2[0]:.2f}', 
                xy=(f1[0], f2[0]), xytext=(f1[0]-3, f2[0]+0.3),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax.annotate(f'Best L/D\nL/D: {f1[-1]:.1f}\nFuel: {f2[-1]:.2f}', 
                xy=(f1[-1], f2[-1]), xytext=(f1[-1]+1, f2[-1]-0.4),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('visualizations/aircraft_pareto_front_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: aircraft_pareto_front_sample.png")

def generate_fidelity_analysis():
    """Generate fidelity switching analysis plot"""
    
    evaluations = np.arange(1, 201)
    
    # Simulate adaptive fidelity switching
    fidelity_levels = []
    for i, eval_num in enumerate(evaluations):
        if eval_num < 50:
            # Start with mostly low fidelity
            level = np.random.choice([1, 2], p=[0.8, 0.2])
        elif eval_num < 120:
            # Mixed fidelity in middle phase
            level = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        else:
            # More high fidelity towards end
            level = np.random.choice([1, 2, 3], p=[0.2, 0.3, 0.5])
        fidelity_levels.append(level)
    
    # Create subplot figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Fidelity switching over time
    colors = {1: 'green', 2: 'orange', 3: 'red'}
    for i, level in enumerate(fidelity_levels):
        ax1.scatter(evaluations[i], level, c=colors[level], alpha=0.7, s=30)
    
    ax1.set_ylabel('Fidelity Level', fontsize=11)
    ax1.set_title('Adaptive Fidelity Switching During Optimization', fontsize=13, fontweight='bold')
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(['Low\n(~0.1s)', 'Medium\n(~2s)', 'High\n(~15s)'])
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative computational cost
    costs = {'1': 0.1, '2': 2.0, '3': 15.0}
    cumulative_cost = [costs[str(level)] for level in fidelity_levels]
    cumulative_cost = np.cumsum(cumulative_cost)
    
    # Compare with high-fidelity only
    high_fidelity_only = np.cumsum([15.0] * len(evaluations))
    
    ax2.plot(evaluations, cumulative_cost, 'b-', linewidth=2, label='Adaptive Multi-Fidelity')
    ax2.plot(evaluations, high_fidelity_only, 'r--', linewidth=2, label='High-Fidelity Only')
    
    savings = (1 - cumulative_cost[-1] / high_fidelity_only[-1]) * 100
    ax2.text(0.6, 0.8, f'Cost Savings: {savings:.1f}%', transform=ax2.transAxes,
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    ax2.set_ylabel('Cumulative Time (seconds)', fontsize=11)
    ax2.set_title('Computational Cost Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Fidelity distribution pie chart
    fidelity_counts = {1: fidelity_levels.count(1), 
                      2: fidelity_levels.count(2), 
                      3: fidelity_levels.count(3)}
    
    labels = ['Low Fidelity\n(Fast)', 'Medium Fidelity\n(Balanced)', 'High Fidelity\n(Accurate)']
    sizes = [fidelity_counts[1], fidelity_counts[2], fidelity_counts[3]]
    colors_pie = ['lightgreen', 'orange', 'lightcoral']
    
    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 10})
    ax3.set_title('Fidelity Level Distribution', fontsize=13, fontweight='bold')
    
    ax1.set_xlabel('Evaluation Number', fontsize=11)
    ax2.set_xlabel('Evaluation Number', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('visualizations/fidelity_analysis_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: fidelity_analysis_sample.png")

def generate_uncertainty_analysis():
    """Generate uncertainty propagation analysis plot"""
    
    # Generate sample uncertainty data
    np.random.seed(42)  # For reproducibility
    
    # Monte Carlo results for L/D ratio under uncertainty
    n_samples = 1000
    nominal_ld = 26.5
    
    # Parameter uncertainties
    wingspan_var = np.random.normal(0, 0.5, n_samples)
    area_var = np.random.normal(0, 5.0, n_samples)
    mach_var = np.random.normal(0, 0.01, n_samples)
    
    # Environmental uncertainties
    wind_var = np.random.uniform(-5, 5, n_samples)
    temp_var = np.random.normal(0, 3, n_samples)
    
    # Combined effect on L/D ratio (simplified model)
    ld_samples = nominal_ld + 0.3*wingspan_var + 0.05*area_var - 0.1*abs(wind_var) + 0.02*temp_var + np.random.normal(0, 0.5, n_samples)
    
    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histogram of L/D ratio distribution
    ax1.hist(ld_samples, bins=40, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    ax1.axvline(np.mean(ld_samples), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ld_samples):.2f}')
    ax1.axvline(nominal_ld, color='green', linestyle='-', linewidth=2, label=f'Nominal: {nominal_ld:.2f}')
    
    # Add confidence intervals
    ci_95 = np.percentile(ld_samples, [2.5, 97.5])
    ax1.axvspan(ci_95[0], ci_95[1], alpha=0.2, color='yellow', label=f'95% CI')
    
    ax1.set_xlabel('Lift-to-Drag Ratio', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('L/D Ratio Distribution Under Uncertainty', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sensitivity tornado diagram
    parameters = ['Wingspan', 'Wing Area', 'Cruise Mach', 'Wind Speed', 'Temperature']
    sensitivities = [0.35, 0.28, 0.15, 0.12, 0.08]  # Example sensitivity indices
    
    y_pos = np.arange(len(parameters))
    bars = ax2.barh(y_pos, sensitivities, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(parameters)
    ax2.set_xlabel('Sensitivity Index', fontsize=11)
    ax2.set_title('Parameter Sensitivity Analysis', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', ha='left', va='center', fontsize=10)
    
    # Plot 3: Scatter plot showing parameter correlation
    ax3.scatter(wingspan_var, ld_samples, alpha=0.6, s=20, color='blue')
    
    # Fit trend line
    z = np.polyfit(wingspan_var, ld_samples, 1)
    p = np.poly1d(z)
    ax3.plot(wingspan_var, p(wingspan_var), "r--", alpha=0.8, linewidth=2)
    
    correlation = np.corrcoef(wingspan_var, ld_samples)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes,
             fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    ax3.set_xlabel('Wingspan Uncertainty (m)', fontsize=11)
    ax3.set_ylabel('L/D Ratio', fontsize=11)
    ax3.set_title('Wingspan Impact on Performance', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics summary
    ax4.axis('off')
    
    stats_text = f"""
    UNCERTAINTY ANALYSIS SUMMARY
    
    • Mean L/D Ratio: {np.mean(ld_samples):.2f}
    • Standard Deviation: {np.std(ld_samples):.3f}
    • 95% Confidence Interval: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]
    • Coefficient of Variation: {np.std(ld_samples)/np.mean(ld_samples)*100:.1f}%
    
    • Worst Case (1%): {np.percentile(ld_samples, 1):.2f}
    • Best Case (99%): {np.percentile(ld_samples, 99):.2f}
    
    • Probability > 25.0: {np.mean(ld_samples > 25.0)*100:.1f}%
    • Probability > 27.0: {np.mean(ld_samples > 27.0)*100:.1f}%
    
    RISK ASSESSMENT:
    • Low Risk: Performance within ±5% of nominal
    • Risk Level: {"Low" if np.std(ld_samples)/np.mean(ld_samples) < 0.05 else "Medium"}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('visualizations/uncertainty_analysis_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: uncertainty_analysis_sample.png")

def generate_spacecraft_comparison():
    """Generate spacecraft mission comparison visualization"""
    
    missions = ['Earth Observation', 'Communication Satellite', 'Deep Space Probe']
    
    # Sample performance data
    performance_data = {
        'Mission Success Probability': [0.942, 0.896, 0.873],
        'Total Mass (tons)': [2.85, 4.12, 3.67],
        'Delta-V Capability (km/s)': [0.485, 0.312, 12.8],
        'Mission Duration (years)': [5.2, 15.8, 18.5],
        'Power Efficiency': [1.15, 1.32, 0.98]
    }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mission Success Probability Comparison
    bars1 = ax1.bar(missions, performance_data['Mission Success Probability'], 
                    color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_ylabel('Mission Success Probability', fontsize=11)
    ax1.set_title('Mission Success Probability by Mission Type', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.8, 1.0)
    
    # Add value labels
    for bar, value in zip(bars1, performance_data['Mission Success Probability']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Mass vs Delta-V scatter
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    for i, mission in enumerate(missions):
        ax2.scatter(performance_data['Total Mass (tons)'][i], 
                   performance_data['Delta-V Capability (km/s)'][i],
                   s=200, c=colors[i], alpha=0.8, edgecolor='black', linewidth=1,
                   label=mission)
    
    ax2.set_xlabel('Total Mass (tons)', fontsize=11)
    ax2.set_ylabel('Delta-V Capability (km/s)', fontsize=11)
    ax2.set_title('Mass vs Delta-V Trade-off', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Radar chart for multi-dimensional comparison
    categories = ['Success\nProbability', 'Mass\nEfficiency', 'Delta-V\nCapability', 
                 'Mission\nDuration', 'Power\nEfficiency']
    
    # Normalize data to 0-1 scale for radar chart
    normalized_data = {}
    for mission in missions:
        normalized_data[mission] = [
            performance_data['Mission Success Probability'][missions.index(mission)],
            1 - (performance_data['Total Mass (tons)'][missions.index(mission)] - min(performance_data['Total Mass (tons)'])) / 
                (max(performance_data['Total Mass (tons)']) - min(performance_data['Total Mass (tons)'])),
            (performance_data['Delta-V Capability (km/s)'][missions.index(mission)] - min(performance_data['Delta-V Capability (km/s)'])) /
                (max(performance_data['Delta-V Capability (km/s)']) - min(performance_data['Delta-V Capability (km/s)'])),
            (performance_data['Mission Duration (years)'][missions.index(mission)] - min(performance_data['Mission Duration (years)'])) /
                (max(performance_data['Mission Duration (years)']) - min(performance_data['Mission Duration (years)'])),
            performance_data['Power Efficiency'][missions.index(mission)] / max(performance_data['Power Efficiency'])
        ]
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax3.set_theta_offset(np.pi / 2)
    ax3.set_theta_direction(-1)
    ax3.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    for i, (mission, values) in enumerate(normalized_data.items()):
        values += values[:1]  # Complete the circle
        ax3.plot(angles, values, 'o-', linewidth=2, label=mission, color=colors[i])
        ax3.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax3.set_ylim(0, 1)
    ax3.set_title('Multi-Dimensional Mission Comparison', fontsize=12, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax3.grid(True)
    
    # Plot 4: Cost-Benefit Analysis
    # Simulate cost data (normalized)
    costs = [1.0, 2.3, 3.1]  # Relative costs
    benefits = performance_data['Mission Success Probability']
    
    for i, mission in enumerate(missions):
        ax4.scatter(costs[i], benefits[i], s=300, c=colors[i], alpha=0.8, 
                   edgecolor='black', linewidth=2, label=mission)
        ax4.annotate(mission, (costs[i], benefits[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Relative Cost', fontsize=11)
    ax4.set_ylabel('Mission Success Probability', fontsize=11)
    ax4.set_title('Cost-Benefit Analysis', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add efficiency frontier
    frontier_x = np.linspace(min(costs), max(costs), 100)
    frontier_y = 0.85 + 0.1 * np.log(frontier_x + 0.1)
    ax4.plot(frontier_x, frontier_y, 'k--', alpha=0.5, label='Efficiency Frontier')
    ax4.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/spacecraft_comparison_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: spacecraft_comparison_sample.png")

def generate_algorithm_performance():
    """Generate algorithm performance comparison"""
    
    algorithms = ['Genetic\nAlgorithm', 'Particle Swarm\nOptimization', 'Bayesian\nOptimization']
    
    # Performance metrics
    metrics = {
        'Convergence Rate (%)': [95, 92, 98],
        'Average Time (min)': [8.5, 6.2, 12.3],
        'Best Solution Quality': [0.95, 0.89, 0.97],
        'Robustness': [0.88, 0.82, 0.93]
    }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Convergence Rate
    bars = ax1.bar(algorithms, metrics['Convergence Rate (%)'], 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Convergence Rate (%)', fontsize=11)
    ax1.set_title('Algorithm Convergence Success Rate', fontsize=12, fontweight='bold')
    ax1.set_ylim(80, 100)
    
    for bar, value in zip(bars, metrics['Convergence Rate (%)']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value}%', ha='center', va='bottom', fontsize=10)
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Time vs Quality scatter
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, alg in enumerate(algorithms):
        ax2.scatter(metrics['Average Time (min)'][i], metrics['Best Solution Quality'][i],
                   s=300, c=colors[i], alpha=0.8, edgecolor='black', linewidth=2,
                   label=alg.replace('\n', ' '))
    
    ax2.set_xlabel('Average Optimization Time (minutes)', fontsize=11)
    ax2.set_ylabel('Solution Quality (normalized)', fontsize=11)
    ax2.set_title('Time vs Quality Trade-off', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.8, 1.0)
    
    # Plot 3: Robustness comparison
    bars = ax3.bar(algorithms, metrics['Robustness'], 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.set_ylabel('Robustness Score', fontsize=11)
    ax3.set_title('Algorithm Robustness Under Uncertainty', fontsize=12, fontweight='bold')
    ax3.set_ylim(0.7, 1.0)
    
    for bar, value in zip(bars, metrics['Robustness']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Overall performance radar
    categories = ['Convergence\nRate', 'Speed', 'Solution\nQuality', 'Robustness']
    
    # Normalize for radar chart
    normalized_metrics = []
    for i in range(len(algorithms)):
        normalized_metrics.append([
            metrics['Convergence Rate (%)'][i] / 100,
            1 - (metrics['Average Time (min)'][i] - min(metrics['Average Time (min)'])) / 
                (max(metrics['Average Time (min)']) - min(metrics['Average Time (min)'])),
            metrics['Best Solution Quality'][i],
            metrics['Robustness'][i]
        ])
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax4.set_theta_offset(np.pi / 2)
    ax4.set_theta_direction(-1)
    ax4.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    for i, (alg, values) in enumerate(zip(algorithms, normalized_metrics)):
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=alg.replace('\n', ' '), color=colors[i])
        ax4.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax4.set_ylim(0, 1)
    ax4.set_title('Overall Algorithm Performance', fontsize=12, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/algorithm_performance_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: algorithm_performance_sample.png")

def create_visualization_index():
    """Create an index file describing all visualizations"""
    
    index_content = {
        "visualization_samples": {
            "description": "Sample visualizations generated by the Adaptive Multi-Fidelity Optimization Framework",
            "generated_date": datetime.now().isoformat(),
            "files": {
                "aircraft_convergence_sample.png": {
                    "title": "Aircraft Optimization Convergence",
                    "description": "Shows optimization convergence for genetic algorithm optimizing aircraft L/D ratio",
                    "type": "convergence_plot",
                    "parameters": ["L/D ratio", "generation", "population_average"]
                },
                "aircraft_pareto_front_sample.png": {
                    "title": "Multi-Objective Pareto Front",
                    "description": "Pareto front for aircraft optimization balancing L/D ratio and fuel efficiency",
                    "type": "pareto_front",
                    "parameters": ["L/D ratio", "fuel efficiency", "Pareto optimality"]
                },
                "fidelity_analysis_sample.png": {
                    "title": "Fidelity Switching Analysis", 
                    "description": "Analysis of adaptive fidelity switching showing cost savings and distribution",
                    "type": "fidelity_analysis",
                    "parameters": ["fidelity level", "computational cost", "cost savings"]
                },
                "uncertainty_analysis_sample.png": {
                    "title": "Uncertainty Propagation Analysis",
                    "description": "Monte Carlo analysis showing performance distribution under uncertainty",
                    "type": "uncertainty_analysis",
                    "parameters": ["probability distribution", "sensitivity", "confidence intervals"]
                },
                "spacecraft_comparison_sample.png": {
                    "title": "Spacecraft Mission Comparison",
                    "description": "Comparative analysis of different spacecraft mission types and their trade-offs",
                    "type": "mission_comparison",
                    "parameters": ["mission success", "mass", "delta-V", "cost-benefit"]
                },
                "algorithm_performance_sample.png": {
                    "title": "Algorithm Performance Comparison",
                    "description": "Comprehensive comparison of optimization algorithm performance metrics",
                    "type": "algorithm_comparison",
                    "parameters": ["convergence rate", "solution quality", "robustness", "computational time"]
                }
            },
            "usage_notes": [
                "These are sample visualizations showing the types of plots the framework generates",
                "Actual results will vary based on specific optimization problems and parameters",
                "All plots use professional aerospace industry color schemes and formatting",
                "Visualizations are generated in high resolution (300 DPI) for publication quality"
            ]
        }
    }
    
    with open('visualizations/visualization_index.json', 'w') as f:
        json.dump(index_content, f, indent=2)
    
    print("✓ Generated: visualization_index.json")

def main():
    """Generate all sample visualizations"""
    
    print("="*60)
    print("GENERATING SAMPLE VISUALIZATIONS")
    print("="*60)
    print()
    
    print("Creating sample visualization outputs that demonstrate the")
    print("types of plots and analysis charts generated by the framework...")
    print()
    
    # Generate all visualizations
    generate_convergence_plot()
    generate_pareto_front()
    generate_fidelity_analysis()
    generate_uncertainty_analysis()
    generate_spacecraft_comparison()
    generate_algorithm_performance()
    create_visualization_index()
    
    print()
    print("="*60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*60)
    print()
    print("Generated 6 sample visualization files plus index:")
    print("• aircraft_convergence_sample.png")
    print("• aircraft_pareto_front_sample.png") 
    print("• fidelity_analysis_sample.png")
    print("• uncertainty_analysis_sample.png")
    print("• spacecraft_comparison_sample.png")
    print("• algorithm_performance_sample.png")
    print("• visualization_index.json")
    print()
    print("All files saved in: visualizations/")
    print()

if __name__ == "__main__":
    main()