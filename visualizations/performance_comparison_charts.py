#!/usr/bin/env python3
"""
Performance Comparison Charts
Adaptive Multi-Fidelity Simulation-Based Optimization for Aerospace Systems

Creates comprehensive performance comparison visualizations for optimization
algorithms with statistical analysis and aerospace-specific metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

class PerformanceComparisonVisualizer:
    """Professional performance comparison visualization system."""
    
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
    
    def load_performance_data(self):
        """Load performance comparison data from CSV file."""
        performance_file = self.results_dir / 'performance_comparison.csv'
        
        df = pd.read_csv(performance_file)
        
        return df
    
    def plot_algorithm_performance_overview(self, df):
        """Create comprehensive algorithm performance overview."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Optimization Algorithm Performance Comparison\nAdaptive Multi-Fidelity Framework', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # 1. Success Rate Comparison
        ax = axes[0, 0]
        success_rates = df.groupby('algorithm')['success_rate'].agg(['mean', 'std'])
        
        algorithms = success_rates.index
        means = success_rates['mean']
        stds = success_rates['std']
        
        colors = [ALGORITHM_COLORS.get(alg, AEROSPACE_COLORS['primary_blue']) for alg in algorithms]
        
        bars = ax.bar(algorithms, means, yerr=stds, capsize=5,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Success Rate')
        ax.set_title('Algorithm Success Rates\nMean ± Std Dev', fontweight='bold')
        ax.set_ylim(0, 1.1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add percentage labels
        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{mean_val:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Computational Efficiency
        ax = axes[0, 1]
        efficiency_data = df.groupby('algorithm')['computational_efficiency'].agg(['mean', 'std'])
        
        means_eff = efficiency_data['mean']
        stds_eff = efficiency_data['std']
        
        bars = ax.bar(algorithms, means_eff, yerr=stds_eff, capsize=5,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Computational Efficiency')
        ax.set_title('Computational Efficiency\nComparison', fontweight='bold')
        ax.set_ylim(0, 1.1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Solution Quality Distribution
        ax = axes[0, 2]
        
        # Box plot for solution quality
        quality_data = [df[df['algorithm'] == alg]['solution_quality'].values for alg in algorithms]
        
        box_plot = ax.boxplot(quality_data, labels=[alg.replace('Optimization', '').replace('Algorithm', '') 
                                                   for alg in algorithms],
                             patch_artist=True, notch=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax.set_ylabel('Solution Quality')
        ax.set_title('Solution Quality\nDistribution', fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 4. Cost Savings vs Performance
        ax = axes[1, 0]
        
        for alg in algorithms:
            alg_data = df[df['algorithm'] == alg]
            ax.scatter(alg_data['cost_savings_percent'], alg_data['solution_quality'],
                      color=ALGORITHM_COLORS.get(alg, AEROSPACE_COLORS['primary_blue']),
                      alpha=0.7, s=80, label=alg.replace('Optimization', '').replace('Algorithm', ''),
                      edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Cost Savings (%)')
        ax.set_ylabel('Solution Quality')
        ax.set_title('Cost Efficiency vs\nSolution Quality', fontweight='bold')
        ax.legend()
        
        # Add trend line
        x = df['cost_savings_percent']
        y = df['solution_quality']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), color=AEROSPACE_COLORS['error_red'], 
               linestyle='--', linewidth=2, alpha=0.8)
        
        # 5. Convergence Time Analysis
        ax = axes[1, 1]
        
        # Log scale convergence time comparison
        convergence_data = df.groupby('algorithm')['convergence_time_seconds'].agg(['mean', 'std'])
        
        means_conv = convergence_data['mean']
        stds_conv = convergence_data['std']
        
        bars = ax.bar(algorithms, means_conv, yerr=stds_conv, capsize=5,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Convergence Time (seconds)')
        ax.set_title('Average Convergence Time\nComparison', fontweight='bold')
        ax.set_yscale('log')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 6. Robustness Score Radar Chart
        ax = axes[1, 2]
        
        # Calculate average metrics for radar chart
        metrics = ['success_rate', 'robustness_score', 'computational_efficiency', 'solution_quality']
        
        # Normalize metrics to 0-1 scale for radar chart
        normalized_data = {}
        for metric in metrics:
            max_val = df[metric].max()
            min_val = df[metric].min()
            for alg in algorithms[:4]:  # Limit to 4 algorithms for clarity
                if alg not in normalized_data:
                    normalized_data[alg] = []
                alg_mean = df[df['algorithm'] == alg][metric].mean()
                normalized_val = (alg_mean - min_val) / (max_val - min_val)
                normalized_data[alg].append(normalized_val)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        metric_labels = ['Success\nRate', 'Robustness\nScore', 'Comp.\nEfficiency', 'Solution\nQuality']
        
        for i, alg in enumerate(list(normalized_data.keys())[:4]):
            values = normalized_data[alg] + [normalized_data[alg][0]]  # Complete the circle
            color = ALGORITHM_COLORS.get(alg, AEROSPACE_COLORS['primary_blue'])
            
            ax.plot(angles, values, 'o-', linewidth=2, color=color, 
                   label=alg.replace('Optimization', '').replace('Algorithm', ''), alpha=0.8)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Performance\nRadar Chart', fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'algorithm_performance_overview.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'algorithm_performance_overview.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_problem_type_analysis(self, df):
        """Analyze performance across different problem types."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Analysis by Problem Type\nAerospace Optimization Framework', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # 1. Performance by Problem Type
        ax = axes[0, 0]
        
        # Group by problem type and calculate mean performance
        problem_performance = df.groupby('problem_type')['final_objective_value'].mean().sort_values(ascending=False)
        
        # Color code by problem category
        colors_by_type = []
        for prob_type in problem_performance.index:
            if 'aircraft' in prob_type:
                colors_by_type.append(AEROSPACE_COLORS['primary_blue'])
            elif 'spacecraft' in prob_type:
                colors_by_type.append(AEROSPACE_COLORS['accent_orange'])
            elif 'benchmark' in prob_type:
                colors_by_type.append(AEROSPACE_COLORS['success_green'])
            else:
                colors_by_type.append(AEROSPACE_COLORS['dark_gray'])
        
        bars = ax.barh(range(len(problem_performance)), problem_performance.values,
                      color=colors_by_type, alpha=0.8)
        
        ax.set_yticks(range(len(problem_performance)))
        ax.set_yticklabels([ptype.replace('_', ' ').title() for ptype in problem_performance.index])
        ax.set_xlabel('Average Objective Value')
        ax.set_title('Performance by Problem Type', fontweight='bold')
        
        # 2. Algorithm Effectiveness by Problem Category
        ax = axes[0, 1]
        
        # Create effectiveness matrix
        problem_categories = ['aircraft', 'spacecraft', 'benchmark', 'validation', 'uncertainty']
        algorithms = df['algorithm'].unique()
        
        effectiveness_matrix = np.zeros((len(algorithms), len(problem_categories)))
        
        for i, alg in enumerate(algorithms):
            for j, category in enumerate(problem_categories):
                category_data = df[(df['algorithm'] == alg) & (df['problem_type'].str.contains(category))]
                if not category_data.empty:
                    effectiveness_matrix[i, j] = category_data['solution_quality'].mean()
        
        # Create heatmap
        im = ax.imshow(effectiveness_matrix, cmap='RdYlBu_r', aspect='auto', alpha=0.8)
        
        ax.set_xticks(range(len(problem_categories)))
        ax.set_xticklabels([cat.title() for cat in problem_categories])
        ax.set_yticks(range(len(algorithms)))
        ax.set_yticklabels([alg.replace('Optimization', '').replace('Algorithm', '') for alg in algorithms])
        ax.set_title('Algorithm Effectiveness\nby Problem Category', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Solution Quality', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(problem_categories)):
                if effectiveness_matrix[i, j] > 0:
                    text = ax.text(j, i, f'{effectiveness_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
        
        # 3. Cost Savings by Problem Complexity
        ax = axes[1, 0]
        
        # Define complexity levels based on evaluation counts
        df['complexity'] = pd.cut(df['total_evaluations'], 
                                 bins=[0, 1000, 5000, np.inf], 
                                 labels=['Low', 'Medium', 'High'])
        
        complexity_savings = df.groupby(['complexity', 'algorithm'])['cost_savings_percent'].mean().unstack()
        
        complexity_savings.plot(kind='bar', ax=ax, 
                               color=[ALGORITHM_COLORS.get(alg, AEROSPACE_COLORS['primary_blue']) 
                                     for alg in complexity_savings.columns],
                               alpha=0.8)
        
        ax.set_xlabel('Problem Complexity')
        ax.set_ylabel('Average Cost Savings (%)')
        ax.set_title('Cost Savings vs\nProblem Complexity', fontweight='bold')
        ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.get_xticklabels(), rotation=0)
        
        # 4. Fidelity Usage Patterns
        ax = axes[1, 1]
        
        # Calculate average fidelity usage by problem type
        fidelity_cols = ['fidelity_low_percent', 'fidelity_medium_percent', 'fidelity_high_percent']
        
        # Group by aircraft vs spacecraft problems
        aircraft_problems = df[df['problem_type'].str.contains('aircraft')]
        spacecraft_problems = df[df['problem_type'].str.contains('spacecraft')]
        
        aircraft_fidelity = [aircraft_problems[col].mean() for col in fidelity_cols]
        spacecraft_fidelity = [spacecraft_problems[col].mean() for col in fidelity_cols]
        
        x = np.arange(len(fidelity_cols))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, aircraft_fidelity, width, 
                      label='Aircraft Problems', color=AEROSPACE_COLORS['primary_blue'], alpha=0.8)
        bars2 = ax.bar(x + width/2, spacecraft_fidelity, width,
                      label='Spacecraft Problems', color=AEROSPACE_COLORS['accent_orange'], alpha=0.8)
        
        ax.set_xlabel('Fidelity Level')
        ax.set_ylabel('Average Usage (%)')
        ax.set_title('Fidelity Usage Patterns\nby Problem Domain', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Low', 'Medium', 'High'])
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'problem_type_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'problem_type_analysis.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_statistical_analysis(self, df):
        """Create statistical analysis and significance testing."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Statistical Analysis and Algorithm Comparison\nAdaptive Multi-Fidelity Optimization Framework', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # 1. ANOVA Analysis for Solution Quality
        ax = axes[0, 0]
        
        algorithms = df['algorithm'].unique()
        quality_groups = [df[df['algorithm'] == alg]['solution_quality'].values for alg in algorithms]
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*quality_groups)
        
        # Create violin plot
        parts = ax.violinplot(quality_groups, positions=range(len(algorithms)), showmeans=True)
        
        # Color the violins
        colors = [ALGORITHM_COLORS.get(alg, AEROSPACE_COLORS['primary_blue']) for alg in algorithms]
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.8)
        
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels([alg.replace('Optimization', '').replace('Algorithm', '') for alg in algorithms],
                          rotation=45, ha='right')
        ax.set_ylabel('Solution Quality')
        ax.set_title(f'Solution Quality Distribution\nANOVA: F={f_stat:.2f}, p={p_value:.4f}', fontweight='bold')
        
        # 2. Correlation Matrix
        ax = axes[0, 1]
        
        # Select numerical columns for correlation
        numerical_cols = ['total_evaluations', 'convergence_time_seconds', 'final_objective_value',
                         'cost_savings_percent', 'success_rate', 'robustness_score', 
                         'computational_efficiency', 'solution_quality']
        
        corr_matrix = df[numerical_cols].corr()
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        ax.set_xticks(range(len(numerical_cols)))
        ax.set_xticklabels([col.replace('_', ' ').title() for col in numerical_cols], 
                          rotation=45, ha='right')
        ax.set_yticks(range(len(numerical_cols)))
        ax.set_yticklabels([col.replace('_', ' ').title() for col in numerical_cols])
        ax.set_title('Performance Metrics\nCorrelation Matrix', fontweight='bold')
        
        # Add correlation values
        for i in range(len(numerical_cols)):
            for j in range(len(numerical_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", 
                             color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                             fontsize=8, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
        
        # 3. Principal Component Analysis
        ax = axes[0, 2]
        
        # Prepare data for PCA
        feature_cols = ['cost_savings_percent', 'success_rate', 'robustness_score', 
                       'computational_efficiency', 'solution_quality']
        
        X = df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot PCA results
        for alg in algorithms:
            alg_mask = df['algorithm'] == alg
            ax.scatter(X_pca[alg_mask, 0], X_pca[alg_mask, 1], 
                      color=ALGORITHM_COLORS.get(alg, AEROSPACE_COLORS['primary_blue']),
                      alpha=0.7, s=80, label=alg.replace('Optimization', '').replace('Algorithm', ''),
                      edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('Principal Component Analysis\nAlgorithm Clustering', fontweight='bold')
        ax.legend()
        
        # 4. Performance Trend Analysis
        ax = axes[1, 0]
        
        # Group by evaluation count ranges to show scaling
        df['eval_range'] = pd.cut(df['total_evaluations'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        trend_data = df.groupby(['eval_range', 'algorithm'])['solution_quality'].mean().unstack()
        
        for alg in trend_data.columns:
            ax.plot(range(len(trend_data.index)), trend_data[alg], 
                   marker='o', linewidth=2.5, alpha=0.8,
                   color=ALGORITHM_COLORS.get(alg, AEROSPACE_COLORS['primary_blue']),
                   label=alg.replace('Optimization', '').replace('Algorithm', ''))
        
        ax.set_xticks(range(len(trend_data.index)))
        ax.set_xticklabels(trend_data.index, rotation=45, ha='right')
        ax.set_xlabel('Problem Scale')
        ax.set_ylabel('Average Solution Quality')
        ax.set_title('Scalability Analysis\nQuality vs Problem Size', fontweight='bold')
        ax.legend()
        
        # 5. Efficiency Frontier
        ax = axes[1, 1]
        
        # Plot efficiency frontier (cost vs quality)
        for alg in algorithms:
            alg_data = df[df['algorithm'] == alg]
            ax.scatter(alg_data['convergence_time_seconds'], alg_data['solution_quality'],
                      color=ALGORITHM_COLORS.get(alg, AEROSPACE_COLORS['primary_blue']),
                      alpha=0.7, s=80, label=alg.replace('Optimization', '').replace('Algorithm', ''),
                      edgecolors='black', linewidth=0.5)
        
        # Find and plot Pareto frontier
        combined_data = df[['convergence_time_seconds', 'solution_quality']].values
        
        # Simple Pareto frontier calculation
        is_efficient = np.ones(len(combined_data), dtype=bool)
        for i, c in enumerate(combined_data):
            if is_efficient[i]:
                # Remove dominated points
                is_efficient[is_efficient] = np.any(combined_data[is_efficient] <= c, axis=1)
                is_efficient[i] = True
        
        frontier_points = combined_data[is_efficient]
        frontier_points = frontier_points[np.argsort(frontier_points[:, 0])]
        
        ax.plot(frontier_points[:, 0], frontier_points[:, 1], 
               color=AEROSPACE_COLORS['error_red'], linewidth=2.5, alpha=0.8,
               label='Efficiency Frontier')
        
        ax.set_xlabel('Convergence Time (seconds)')
        ax.set_ylabel('Solution Quality')
        ax.set_title('Efficiency Frontier\nTime vs Quality Trade-off', fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        
        # 6. Algorithm Ranking
        ax = axes[1, 2]
        
        # Calculate composite performance score
        metrics_to_rank = ['success_rate', 'robustness_score', 'computational_efficiency', 'solution_quality']
        
        # Normalize metrics and calculate weighted average
        weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights
        
        algorithm_scores = {}
        for alg in algorithms:
            alg_data = df[df['algorithm'] == alg]
            score = 0
            for metric, weight in zip(metrics_to_rank, weights):
                normalized_metric = (alg_data[metric].mean() - df[metric].min()) / (df[metric].max() - df[metric].min())
                score += weight * normalized_metric
            algorithm_scores[alg] = score
        
        # Sort algorithms by score
        sorted_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)
        
        algorithms_sorted = [alg for alg, _ in sorted_algorithms]
        scores_sorted = [score for _, score in sorted_algorithms]
        
        colors_sorted = [ALGORITHM_COLORS.get(alg, AEROSPACE_COLORS['primary_blue']) for alg in algorithms_sorted]
        
        bars = ax.barh(range(len(algorithms_sorted)), scores_sorted,
                      color=colors_sorted, alpha=0.8)
        
        ax.set_yticks(range(len(algorithms_sorted)))
        ax.set_yticklabels([alg.replace('Optimization', '').replace('Algorithm', '') for alg in algorithms_sorted])
        ax.set_xlabel('Composite Performance Score')
        ax.set_title('Overall Algorithm Ranking\nWeighted Performance Score', fontweight='bold')
        
        # Add score labels
        for bar, score in zip(bars, scores_sorted):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'statistical_analysis.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_performance_summary(self, df):
        """Generate comprehensive performance comparison summary."""
        algorithms = df['algorithm'].unique()
        
        summary_stats = {
            'Algorithm Performance Summary': {},
            'Overall Statistics': {
                'Total Test Cases': len(df),
                'Problem Types': df['problem_type'].nunique(),
                'Average Cost Savings': f"{df['cost_savings_percent'].mean():.1f}%",
                'Overall Success Rate': f"{df['success_rate'].mean():.1%}"
            }
        }
        
        # Calculate detailed statistics for each algorithm
        for alg in algorithms:
            alg_data = df[df['algorithm'] == alg]
            
            summary_stats['Algorithm Performance Summary'][alg] = {
                'Test Cases': len(alg_data),
                'Average Success Rate': f"{alg_data['success_rate'].mean():.1%}",
                'Average Solution Quality': f"{alg_data['solution_quality'].mean():.3f}",
                'Average Cost Savings': f"{alg_data['cost_savings_percent'].mean():.1f}%",
                'Average Convergence Time': f"{alg_data['convergence_time_seconds'].mean():.1f}s",
                'Robustness Score': f"{alg_data['robustness_score'].mean():.3f}",
                'Computational Efficiency': f"{alg_data['computational_efficiency'].mean():.3f}"
            }
        
        # Best performing algorithm by different metrics
        summary_stats['Best Performers'] = {
            'Highest Success Rate': df.loc[df['success_rate'].idxmax(), 'algorithm'],
            'Best Solution Quality': df.loc[df['solution_quality'].idxmax(), 'algorithm'],
            'Fastest Convergence': df.loc[df['convergence_time_seconds'].idxmin(), 'algorithm'],
            'Highest Cost Savings': df.loc[df['cost_savings_percent'].idxmax(), 'algorithm'],
            'Most Robust': df.loc[df['robustness_score'].idxmax(), 'algorithm']
        }
        
        # Save summary as JSON
        import json
        with open(self.output_dir / 'performance_comparison_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Performance comparison analysis complete!")
        print(f"Generated plots saved to: {self.output_dir}")
        print(f"Summary statistics:")
        for category, data in summary_stats.items():
            print(f"\n{category}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for subkey, subvalue in value.items():
                            print(f"    {subkey}: {subvalue}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {data}")

def main():
    """Main function to generate all performance comparison visualizations."""
    visualizer = PerformanceComparisonVisualizer()
    
    # Load performance data
    df = visualizer.load_performance_data()
    
    print("Generating performance comparison visualizations...")
    
    # Generate all plots
    visualizer.plot_algorithm_performance_overview(df)
    print("✓ Algorithm performance overview generated")
    
    visualizer.plot_problem_type_analysis(df)
    print("✓ Problem type analysis generated")
    
    visualizer.plot_statistical_analysis(df)
    print("✓ Statistical analysis generated")
    
    visualizer.generate_performance_summary(df)
    print("✓ Performance comparison summary generated")

if __name__ == "__main__":
    main()