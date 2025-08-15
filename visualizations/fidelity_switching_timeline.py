#!/usr/bin/env python3
"""
Fidelity Switching Timeline Visualization
Adaptive Multi-Fidelity Simulation-Based Optimization for Aerospace Systems

Creates professional timeline visualizations showing adaptive fidelity switching
decisions with cost analysis and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from datetime import datetime, timedelta
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
    'low_fidelity': '#90EE90',
    'medium_fidelity': '#FFD700',
    'high_fidelity': '#FF6B6B'
}

# Fidelity colors
FIDELITY_COLORS = {
    'low': AEROSPACE_COLORS['low_fidelity'],
    'medium': AEROSPACE_COLORS['medium_fidelity'],
    'high': AEROSPACE_COLORS['high_fidelity']
}

class FidelitySwitchingVisualizer:
    """Professional fidelity switching timeline visualization system."""
    
    def __init__(self, results_dir='../results', output_dir='./plots'):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set professional matplotlib style
        plt.style.use('default')
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'figure.figsize': (14, 10),
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
    
    def load_fidelity_data(self):
        """Load fidelity switching logs from CSV file."""
        fidelity_file = self.results_dir / 'fidelity_switching_logs.csv'
        
        df = pd.read_csv(fidelity_file)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def plot_fidelity_timeline(self, df):
        """Create comprehensive fidelity switching timeline visualization."""
        # Get unique optimization runs
        runs = df['optimization_run'].unique()
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle('Adaptive Fidelity Switching Timeline Analysis\nMulti-Fidelity Optimization Framework', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # Select representative runs for detailed analysis
        featured_runs = runs[:6] if len(runs) >= 6 else runs
        
        for idx, run in enumerate(featured_runs):
            if idx >= 6:
                break
                
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Filter data for this run
            run_data = df[df['optimization_run'] == run].copy()
            run_data = run_data.sort_values('evaluation_number')
            
            # Create timeline plot
            eval_numbers = run_data['evaluation_number'].values
            
            # Map fidelity levels to numerical values for plotting
            fidelity_map = {'low': 1, 'medium': 2, 'high': 3}
            fidelity_values = [fidelity_map[f] for f in run_data['current_fidelity']]
            
            # Plot fidelity level over time
            for i in range(len(eval_numbers)):
                color = FIDELITY_COLORS[run_data.iloc[i]['current_fidelity']]
                ax.scatter(eval_numbers[i], fidelity_values[i], 
                          c=color, s=100, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Connect points to show evolution
            ax.plot(eval_numbers, fidelity_values, 
                   color=AEROSPACE_COLORS['dark_gray'], linewidth=1, alpha=0.5)
            
            # Add cost overlay
            ax2 = ax.twinx()
            cost_savings = run_data['cost_savings_vs_high_fidelity'].values * 100
            ax2.plot(eval_numbers, cost_savings, 
                    color=AEROSPACE_COLORS['accent_orange'], linewidth=2.5, alpha=0.8,
                    label='Cost Savings %')
            ax2.set_ylabel('Cost Savings (%)', color=AEROSPACE_COLORS['accent_orange'])
            ax2.tick_params(axis='y', labelcolor=AEROSPACE_COLORS['accent_orange'])
            
            # Formatting
            ax.set_xlabel('Evaluation Number')
            ax.set_ylabel('Fidelity Level')
            ax.set_yticks([1, 2, 3])
            ax.set_yticklabels(['Low', 'Medium', 'High'])
            ax.set_title(f'{run.replace("_", " ").title()}\n{run_data.iloc[0]["algorithm"]}', 
                        fontweight='bold', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add final cost savings as text
            final_savings = cost_savings[-1]
            ax.text(0.02, 0.98, f'Final Savings: {final_savings:.1f}%', 
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=AEROSPACE_COLORS['light_gray'], alpha=0.8),
                   verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fidelity_switching_timeline.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'fidelity_switching_timeline.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_switching_reasons_analysis(self, df):
        """Analyze and visualize reasons for fidelity switching."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fidelity Switching Decision Analysis\nAdaptive Multi-Fidelity Framework', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # Switching reasons frequency
        ax = axes[0, 0]
        reason_counts = df['switch_reason'].value_counts()
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(reason_counts)), reason_counts.values,
                      color=[AEROSPACE_COLORS['primary_blue'], AEROSPACE_COLORS['secondary_blue'],
                            AEROSPACE_COLORS['accent_orange'], AEROSPACE_COLORS['success_green'],
                            AEROSPACE_COLORS['warning_amber']][:len(reason_counts)],
                      alpha=0.8)
        
        ax.set_yticks(range(len(reason_counts)))
        ax.set_yticklabels([reason.replace('_', ' ').title() for reason in reason_counts.index])
        ax.set_xlabel('Frequency')
        ax.set_title('Fidelity Switching Reasons\nFrequency Analysis', fontweight='bold')
        
        # Add frequency labels
        for i, (bar, count) in enumerate(zip(bars, reason_counts.values)):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   str(count), ha='left', va='center', fontweight='bold')
        
        # Cost impact by switching reason
        ax = axes[0, 1]
        reason_costs = df.groupby('switch_reason')['computational_cost_seconds'].mean()
        
        bars = ax.bar(range(len(reason_costs)), reason_costs.values,
                     color=AEROSPACE_COLORS['accent_orange'], alpha=0.8)
        
        ax.set_xticks(range(len(reason_costs)))
        ax.set_xticklabels([reason.replace('_', ' ')[:8] for reason in reason_costs.index], 
                          rotation=45, ha='right')
        ax.set_ylabel('Average Cost (seconds)')
        ax.set_title('Computational Cost by\nSwitching Reason', fontweight='bold')
        
        # Fidelity distribution over time
        ax = axes[1, 0]
        
        # Group by evaluation number ranges to show evolution
        eval_ranges = pd.cut(df['evaluation_number'], bins=10, labels=False)
        df_with_ranges = df.copy()
        df_with_ranges['eval_range'] = eval_ranges
        
        fidelity_evolution = df_with_ranges.groupby(['eval_range', 'current_fidelity']).size().unstack(fill_value=0)
        
        # Calculate percentages
        fidelity_percentages = fidelity_evolution.div(fidelity_evolution.sum(axis=1), axis=0) * 100
        
        # Stacked area plot
        eval_range_labels = range(len(fidelity_percentages))
        
        ax.stackplot(eval_range_labels, 
                    fidelity_percentages['low'] if 'low' in fidelity_percentages.columns else [0]*len(eval_range_labels),
                    fidelity_percentages['medium'] if 'medium' in fidelity_percentages.columns else [0]*len(eval_range_labels),
                    fidelity_percentages['high'] if 'high' in fidelity_percentages.columns else [0]*len(eval_range_labels),
                    labels=['Low Fidelity', 'Medium Fidelity', 'High Fidelity'],
                    colors=[FIDELITY_COLORS['low'], FIDELITY_COLORS['medium'], FIDELITY_COLORS['high']],
                    alpha=0.8)
        
        ax.set_xlabel('Optimization Progress (Evaluation Ranges)')
        ax.set_ylabel('Fidelity Usage (%)')
        ax.set_title('Fidelity Distribution Evolution\nOver Optimization Progress', fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 100)
        
        # Decision confidence analysis
        ax = axes[1, 1]
        
        # Scatter plot: Uncertainty vs Decision Confidence
        scatter = ax.scatter(df['uncertainty_level'], df['decision_confidence'],
                           c=df['computational_cost_seconds'], s=80, alpha=0.7,
                           cmap='viridis', edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Uncertainty Level')
        ax.set_ylabel('Decision Confidence')
        ax.set_title('Decision Quality Analysis\nUncertainty vs Confidence', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Computational Cost (s)', rotation=270, labelpad=15)
        
        # Add trend line
        z = np.polyfit(df['uncertainty_level'], df['decision_confidence'], 1)
        p = np.poly1d(z)
        ax.plot(df['uncertainty_level'], p(df['uncertainty_level']), 
               color=AEROSPACE_COLORS['error_red'], linestyle='--', linewidth=2, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fidelity_switching_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'fidelity_switching_analysis.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_cost_efficiency_analysis(self, df):
        """Create detailed cost efficiency and savings analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Computational Cost Efficiency Analysis\nAdaptive Fidelity Management System', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # Cost savings over time
        ax = axes[0, 0]
        
        for run in df['optimization_run'].unique()[:5]:  # Limit to 5 runs for clarity
            run_data = df[df['optimization_run'] == run].sort_values('evaluation_number')
            cost_savings = run_data['cost_savings_vs_high_fidelity'] * 100
            
            ax.plot(run_data['evaluation_number'], cost_savings,
                   linewidth=2, alpha=0.8, marker='o', markersize=4,
                   label=run.split('_')[0].title())
        
        ax.set_xlabel('Evaluation Number')
        ax.set_ylabel('Cost Savings (%)')
        ax.set_title('Cost Savings Evolution\nAcross Optimization Runs', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Cumulative cost comparison
        ax = axes[0, 1]
        
        # Calculate cumulative costs for different strategies
        df_sorted = df.sort_values(['optimization_run', 'evaluation_number'])
        
        # Group by run and calculate cumulative costs
        cumulative_data = []
        for run in df['optimization_run'].unique()[:3]:  # Sample runs
            run_data = df_sorted[df_sorted['optimization_run'] == run]
            cumulative_cost = run_data['cumulative_cost'].values
            high_fidelity_equivalent = cumulative_cost / (1 - run_data['cost_savings_vs_high_fidelity'])
            
            ax.plot(run_data['evaluation_number'], cumulative_cost,
                   linewidth=2.5, label=f'{run.split("_")[0]} (Adaptive)',
                   color=AEROSPACE_COLORS['primary_blue'], alpha=0.8)
            ax.plot(run_data['evaluation_number'], high_fidelity_equivalent,
                   linewidth=2, linestyle='--', alpha=0.6,
                   color=AEROSPACE_COLORS['error_red'], 
                   label=f'{run.split("_")[0]} (High-Fi Only)' if run == df['optimization_run'].unique()[0] else "")
        
        ax.set_xlabel('Evaluation Number')
        ax.set_ylabel('Cumulative Cost (seconds)')
        ax.set_title('Cumulative Cost Comparison\nAdaptive vs High-Fidelity Only', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Fidelity level cost distribution
        ax = axes[0, 2]
        
        fidelity_costs = df.groupby('current_fidelity')['computational_cost_seconds'].agg(['mean', 'std'])
        
        fidelities = fidelity_costs.index
        means = fidelity_costs['mean']
        stds = fidelity_costs['std']
        
        colors = [FIDELITY_COLORS[f] for f in fidelities]
        
        bars = ax.bar(fidelities, means, yerr=stds, capsize=5,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Computational Cost (seconds)')
        ax.set_title('Cost Distribution by\nFidelity Level', fontweight='bold')
        ax.set_yscale('log')
        
        # Add cost values as text
        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                   f'{mean_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # Algorithm comparison
        ax = axes[1, 0]
        
        algo_efficiency = df.groupby('algorithm')['cost_savings_vs_high_fidelity'].agg(['mean', 'std'])
        
        algorithms = algo_efficiency.index
        efficiency_means = algo_efficiency['mean'] * 100
        efficiency_stds = algo_efficiency['std'] * 100
        
        bars = ax.bar(algorithms, efficiency_means, yerr=efficiency_stds, capsize=5,
                     color=[AEROSPACE_COLORS['primary_blue'], AEROSPACE_COLORS['secondary_blue'],
                           AEROSPACE_COLORS['accent_orange']][:len(algorithms)],
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Average Cost Savings (%)')
        ax.set_title('Cost Efficiency by\nOptimization Algorithm', fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add efficiency values as text
        for bar, mean_val in zip(bars, efficiency_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Convergence rate vs cost savings
        ax = axes[1, 1]
        
        scatter = ax.scatter(df['convergence_rate'], df['cost_savings_vs_high_fidelity'] * 100,
                           c=df['computational_cost_seconds'], s=80, alpha=0.7,
                           cmap='plasma', edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Convergence Rate')
        ax.set_ylabel('Cost Savings (%)')
        ax.set_title('Convergence vs Efficiency\nTrade-off Analysis', fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Computational Cost (s)', rotation=270, labelpad=15)
        
        # Add trend line
        valid_mask = ~(np.isnan(df['convergence_rate']) | np.isnan(df['cost_savings_vs_high_fidelity']))
        if valid_mask.sum() > 1:
            z = np.polyfit(df[valid_mask]['convergence_rate'], 
                          df[valid_mask]['cost_savings_vs_high_fidelity'] * 100, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df['convergence_rate'].min(), df['convergence_rate'].max(), 100)
            ax.plot(x_trend, p(x_trend), 
                   color=AEROSPACE_COLORS['error_red'], linestyle='--', linewidth=2, alpha=0.8)
        
        # Switching strategy analysis
        ax = axes[1, 2]
        
        # Calculate switching frequency
        switching_freq = df.groupby('optimization_run').apply(
            lambda x: len(x[x['current_fidelity'] != x['current_fidelity'].shift()]) / len(x)
        )
        
        final_savings = df.groupby('optimization_run')['cost_savings_vs_high_fidelity'].last() * 100
        
        # Combine data
        strategy_data = pd.DataFrame({
            'switching_frequency': switching_freq,
            'final_savings': final_savings
        }).dropna()
        
        scatter2 = ax.scatter(strategy_data['switching_frequency'], strategy_data['final_savings'],
                             s=120, alpha=0.8, color=AEROSPACE_COLORS['success_green'],
                             edgecolors='black', linewidth=1)
        
        ax.set_xlabel('Switching Frequency')
        ax.set_ylabel('Final Cost Savings (%)')
        ax.set_title('Switching Strategy Impact\nFrequency vs Final Savings', fontweight='bold')
        
        # Add correlation coefficient
        if len(strategy_data) > 1:
            correlation = strategy_data['switching_frequency'].corr(strategy_data['final_savings'])
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=AEROSPACE_COLORS['light_gray'], alpha=0.8),
                   verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cost_efficiency_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'cost_efficiency_analysis.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_switching_summary(self, df):
        """Generate comprehensive switching analysis summary."""
        summary_stats = {
            'Total Evaluations': len(df),
            'Unique Optimization Runs': df['optimization_run'].nunique(),
            'Algorithms Tested': df['algorithm'].unique().tolist(),
            'Fidelity Distribution': {
                'Low Fidelity': f"{(df['current_fidelity'] == 'low').sum() / len(df) * 100:.1f}%",
                'Medium Fidelity': f"{(df['current_fidelity'] == 'medium').sum() / len(df) * 100:.1f}%",
                'High Fidelity': f"{(df['current_fidelity'] == 'high').sum() / len(df) * 100:.1f}%"
            },
            'Average Cost Savings': f"{df['cost_savings_vs_high_fidelity'].mean() * 100:.1f}%",
            'Maximum Cost Savings': f"{df['cost_savings_vs_high_fidelity'].max() * 100:.1f}%",
            'Average Decision Confidence': f"{df['decision_confidence'].mean():.3f}",
            'Most Common Switch Reason': df['switch_reason'].value_counts().index[0],
            'Total Computational Time': f"{df['cumulative_cost'].max():.1f} seconds"
        }
        
        # Save summary as JSON
        import json
        with open(self.output_dir / 'fidelity_switching_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Fidelity switching analysis complete!")
        print(f"Generated plots saved to: {self.output_dir}")
        print(f"Summary statistics:")
        for key, value in summary_stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            elif isinstance(value, list):
                print(f"  {key}: {', '.join(value)}")
            else:
                print(f"  {key}: {value}")

def main():
    """Main function to generate all fidelity switching visualizations."""
    visualizer = FidelitySwitchingVisualizer()
    
    # Load fidelity switching data
    df = visualizer.load_fidelity_data()
    
    print("Generating fidelity switching timeline visualizations...")
    
    # Generate all plots
    visualizer.plot_fidelity_timeline(df)
    print("✓ Fidelity switching timeline generated")
    
    visualizer.plot_switching_reasons_analysis(df)
    print("✓ Switching reasons analysis generated")
    
    visualizer.plot_cost_efficiency_analysis(df)
    print("✓ Cost efficiency analysis generated")
    
    visualizer.generate_switching_summary(df)
    print("✓ Switching analysis summary generated")

if __name__ == "__main__":
    main()