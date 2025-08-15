import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')


class ProfessionalGraphGenerator:
    def __init__(self, output_dir: str = "visualizations", style: str = "aerospace"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.ProfessionalGraphGenerator")
        
        self.aerospace_colors = {
            'primary_blue': '#1f4e79',
            'secondary_blue': '#4472c4',
            'accent_blue': '#70ad47',
            'light_blue': '#bdd7ee',
            'dark_gray': '#404040',
            'medium_gray': '#767171',
            'light_gray': '#d9d9d9',
            'white': '#ffffff',
            'accent_orange': '#ff7c00',
            'accent_red': '#c5504b'
        }
        
        self.plotly_template = {
            'layout': {
                'font': {'family': 'Arial, sans-serif', 'size': 12},
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'colorway': [
                    self.aerospace_colors['primary_blue'],
                    self.aerospace_colors['secondary_blue'],
                    self.aerospace_colors['accent_blue'],
                    self.aerospace_colors['accent_orange'],
                    self.aerospace_colors['accent_red']
                ],
                'xaxis': {
                    'gridcolor': self.aerospace_colors['light_gray'],
                    'linecolor': self.aerospace_colors['dark_gray'],
                    'tickcolor': self.aerospace_colors['dark_gray']
                },
                'yaxis': {
                    'gridcolor': self.aerospace_colors['light_gray'],
                    'linecolor': self.aerospace_colors['dark_gray'],
                    'tickcolor': self.aerospace_colors['dark_gray']
                }
            }
        }
        
        self._setup_matplotlib_style()
    
    def _setup_matplotlib_style(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 1.2,
            'grid.alpha': 0.3
        })
    
    def create_convergence_plot(self, optimization_history: List[Dict[str, Any]], 
                              algorithm_name: str, save_filename: str) -> str:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        evaluations = [entry.get('evaluation', i) for i, entry in enumerate(optimization_history)]
        
        if 'fitness' in optimization_history[0]:
            fitness_values = [entry['fitness'] for entry in optimization_history]
            
            ax1.plot(evaluations, fitness_values, 
                    color=self.aerospace_colors['primary_blue'], linewidth=2, alpha=0.7)
            
            running_best = []
            best_so_far = float('-inf')
            for fitness in fitness_values:
                best_so_far = max(best_so_far, fitness)
                running_best.append(best_so_far)
            
            ax1.plot(evaluations, running_best, 
                    color=self.aerospace_colors['accent_orange'], linewidth=3, 
                    label='Best So Far')
            
            ax1.set_ylabel('Fitness Value')
            ax1.set_title(f'{algorithm_name} - Fitness Convergence', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        if 'objectives' in optimization_history[0]:
            objectives_data = {}
            for entry in optimization_history:
                for obj_name, obj_value in entry['objectives'].items():
                    if obj_name not in objectives_data:
                        objectives_data[obj_name] = []
                    objectives_data[obj_name].append(obj_value)
            
            colors = [self.aerospace_colors['primary_blue'], 
                     self.aerospace_colors['secondary_blue'],
                     self.aerospace_colors['accent_blue'],
                     self.aerospace_colors['accent_orange']]
            
            for i, (obj_name, values) in enumerate(objectives_data.items()):
                color = colors[i % len(colors)]
                ax2.plot(evaluations, values, color=color, linewidth=2, 
                        label=obj_name.replace('_', ' ').title(), alpha=0.8)
            
            ax2.set_xlabel('Evaluation Number')
            ax2.set_ylabel('Objective Values')
            ax2.set_title('Individual Objectives Progress', fontweight='bold')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{save_filename}_convergence.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Saved convergence plot: {filepath}")
        return str(filepath)
    
    def create_pareto_front_plot(self, optimization_results: List[Dict[str, Any]], 
                               save_filename: str) -> str:
        if len(optimization_results) < 2:
            self.logger.warning("Need at least 2 optimization results for Pareto front")
            return ""
        
        fig = plt.figure(figsize=(14, 10))
        
        if len(optimization_results[0]['objectives']) >= 2:
            ax = fig.add_subplot(111)
            
            obj_names = list(optimization_results[0]['objectives'].keys())
            obj1_name, obj2_name = obj_names[0], obj_names[1]
            
            obj1_values = [result['objectives'][obj1_name] for result in optimization_results]
            obj2_values = [result['objectives'][obj2_name] for result in optimization_results]
            
            scatter = ax.scatter(obj1_values, obj2_values, 
                               c=range(len(obj1_values)), 
                               cmap='viridis', s=60, alpha=0.7, edgecolors='black')
            
            pareto_front = self._find_pareto_front(obj1_values, obj2_values)
            if pareto_front:
                pareto_x, pareto_y = zip(*pareto_front)
                ax.plot(pareto_x, pareto_y, 'r-', linewidth=3, alpha=0.8, label='Pareto Front')
                ax.scatter(pareto_x, pareto_y, c='red', s=100, marker='s', 
                          edgecolors='black', label='Pareto Optimal', zorder=5)
            
            ax.set_xlabel(obj1_name.replace('_', ' ').title(), fontsize=14)
            ax.set_ylabel(obj2_name.replace('_', ' ').title(), fontsize=14)
            ax.set_title('Multi-Objective Optimization - Pareto Front', fontsize=16, fontweight='bold')
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Evaluation Order', fontsize=12)
            
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{save_filename}_pareto_front.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Saved Pareto front plot: {filepath}")
        return str(filepath)
    
    def create_fidelity_switching_plot(self, fidelity_history: List[Dict[str, Any]], 
                                     save_filename: str) -> str:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        evaluations = list(range(len(fidelity_history)))
        fidelity_levels = [entry['fidelity'].value for entry in fidelity_history]
        
        fidelity_mapping = {'low': 1, 'medium': 2, 'high': 3}
        fidelity_numeric = [fidelity_mapping[f] for f in fidelity_levels]
        
        colors = {'low': self.aerospace_colors['accent_blue'], 
                 'medium': self.aerospace_colors['secondary_blue'], 
                 'high': self.aerospace_colors['primary_blue']}
        
        for i, (eval_num, fidelity) in enumerate(zip(evaluations, fidelity_levels)):
            ax1.scatter(eval_num, fidelity_mapping[fidelity], 
                       c=colors[fidelity], s=50, alpha=0.7)
        
        ax1.set_ylabel('Fidelity Level')
        ax1.set_title('Adaptive Fidelity Switching Timeline', fontsize=16, fontweight='bold')
        ax1.set_yticks([1, 2, 3])
        ax1.set_yticklabels(['Low', 'Medium', 'High'])
        ax1.grid(True, alpha=0.3)
        
        if 'result' in fidelity_history[0] and hasattr(fidelity_history[0]['result'], 'computation_time'):
            computation_times = [entry['result'].computation_time for entry in fidelity_history]
            cumulative_time = np.cumsum(computation_times)
            
            ax2.plot(evaluations, cumulative_time, 
                    color=self.aerospace_colors['accent_orange'], linewidth=2)
            ax2.fill_between(evaluations, cumulative_time, alpha=0.3, 
                           color=self.aerospace_colors['accent_orange'])
            
            ax2.set_xlabel('Evaluation Number')
            ax2.set_ylabel('Cumulative Computation Time (s)')
            ax2.set_title('Computational Cost Over Time', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{save_filename}_fidelity_switching.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Saved fidelity switching plot: {filepath}")
        return str(filepath)
    
    def create_uncertainty_propagation_plot(self, monte_carlo_results: List[Dict[str, Any]], 
                                          save_filename: str) -> str:
        if not monte_carlo_results:
            self.logger.warning("No Monte Carlo results provided")
            return ""
        
        objectives_data = {}
        for result in monte_carlo_results:
            for obj_name, obj_value in result['objectives'].items():
                if obj_name not in objectives_data:
                    objectives_data[obj_name] = []
                objectives_data[obj_name].append(obj_value)
        
        n_objectives = len(objectives_data)
        fig, axes = plt.subplots(2, (n_objectives + 1) // 2, figsize=(16, 10))
        if n_objectives == 1:
            axes = [axes]
        elif n_objectives <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (obj_name, values) in enumerate(objectives_data.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            ax.hist(values, bins=30, density=True, alpha=0.7, 
                   color=self.aerospace_colors['primary_blue'], edgecolor='black')
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            x = np.linspace(min(values), max(values), 100)
            y = (1/(std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_val) / std_val) ** 2)
            ax.plot(x, y, color=self.aerospace_colors['accent_orange'], linewidth=2, 
                   label='Normal Fit')
            
            ax.axvline(mean_val, color=self.aerospace_colors['accent_red'], 
                      linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val + std_val, color=self.aerospace_colors['accent_red'], 
                      linestyle=':', alpha=0.7, label=f'±1σ: {std_val:.3f}')
            ax.axvline(mean_val - std_val, color=self.aerospace_colors['accent_red'], 
                      linestyle=':', alpha=0.7)
            
            ax.set_xlabel(obj_name.replace('_', ' ').title())
            ax.set_ylabel('Probability Density')
            ax.set_title(f'Uncertainty Distribution - {obj_name.replace("_", " ").title()}', 
                        fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{save_filename}_uncertainty_propagation.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Saved uncertainty propagation plot: {filepath}")
        return str(filepath)
    
    def create_performance_comparison_plot(self, comparison_data: Dict[str, Any], 
                                         save_filename: str) -> str:
        algorithms = list(comparison_data.keys())
        
        if not algorithms:
            self.logger.warning("No algorithms to compare")
            return ""
        
        all_objectives = set()
        for alg_data in comparison_data.values():
            if 'objectives' in alg_data:
                all_objectives.update(alg_data['objectives'].keys())
        
        all_objectives = list(all_objectives)
        n_objectives = len(all_objectives)
        
        fig, axes = plt.subplots(2, (n_objectives + 1) // 2, figsize=(16, 12))
        if n_objectives == 1:
            axes = [axes]
        elif n_objectives <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        colors = [self.aerospace_colors['primary_blue'], 
                 self.aerospace_colors['secondary_blue'],
                 self.aerospace_colors['accent_blue'],
                 self.aerospace_colors['accent_orange'],
                 self.aerospace_colors['accent_red']]
        
        for i, obj_name in enumerate(all_objectives):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            means = []
            stds = []
            alg_names = []
            
            for j, algorithm in enumerate(algorithms):
                if (algorithm in comparison_data and 
                    'objectives' in comparison_data[algorithm] and 
                    obj_name in comparison_data[algorithm]['objectives']):
                    
                    obj_data = comparison_data[algorithm]['objectives'][obj_name]
                    means.append(obj_data.get('mean', 0))
                    stds.append(obj_data.get('std', 0))
                    alg_names.append(algorithm)
            
            if means:
                x_pos = np.arange(len(alg_names))
                bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                             color=[colors[j % len(colors)] for j in range(len(alg_names))],
                             alpha=0.8, edgecolor='black')
                
                ax.set_xlabel('Algorithm')
                ax.set_ylabel(obj_name.replace('_', ' ').title())
                ax.set_title(f'Performance Comparison - {obj_name.replace("_", " ").title()}', 
                           fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(alg_names, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                for bar, mean_val in zip(bars, means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(stds) * 0.1,
                           f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{save_filename}_performance_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Saved performance comparison plot: {filepath}")
        return str(filepath)
    
    def create_3d_design_space_plot(self, optimization_history: List[Dict[str, Any]], 
                                  param_names: List[str], save_filename: str) -> str:
        if len(param_names) < 3:
            self.logger.warning("Need at least 3 parameters for 3D plot")
            return ""
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        param1_values = [entry['parameters'][param_names[0]] for entry in optimization_history 
                        if 'parameters' in entry and param_names[0] in entry['parameters']]
        param2_values = [entry['parameters'][param_names[1]] for entry in optimization_history 
                        if 'parameters' in entry and param_names[1] in entry['parameters']]
        param3_values = [entry['parameters'][param_names[2]] for entry in optimization_history 
                        if 'parameters' in entry and param_names[2] in entry['parameters']]
        
        if 'fitness' in optimization_history[0]:
            fitness_values = [entry['fitness'] for entry in optimization_history]
            
            scatter = ax.scatter(param1_values, param2_values, param3_values, 
                               c=fitness_values, cmap='viridis', s=60, alpha=0.7)
            
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
            cbar.set_label('Fitness Value', fontsize=12)
        else:
            ax.scatter(param1_values, param2_values, param3_values, 
                      c=self.aerospace_colors['primary_blue'], s=60, alpha=0.7)
        
        ax.set_xlabel(param_names[0].replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(param_names[1].replace('_', ' ').title(), fontsize=12)
        ax.set_zlabel(param_names[2].replace('_', ' ').title(), fontsize=12)
        ax.set_title('3D Design Space Exploration', fontsize=16, fontweight='bold')
        
        filepath = self.output_dir / f"{save_filename}_3d_design_space.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Saved 3D design space plot: {filepath}")
        return str(filepath)
    
    def create_interactive_dashboard(self, optimization_data: Dict[str, Any], 
                                   save_filename: str) -> str:
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Convergence History', 'Parameter Distribution', 
                          'Objective Correlation', 'Fidelity Usage'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        optimization_history = optimization_data.get('optimization_history', [])
        
        if optimization_history:
            evaluations = list(range(len(optimization_history)))
            
            if 'fitness' in optimization_history[0]:
                fitness_values = [entry['fitness'] for entry in optimization_history]
                fig.add_trace(
                    go.Scatter(x=evaluations, y=fitness_values, mode='lines',
                              name='Fitness', line=dict(color=self.aerospace_colors['primary_blue'])),
                    row=1, col=1
                )
            
            if 'parameters' in optimization_history[0]:
                params = optimization_history[0]['parameters'].keys()
                param_data = {}
                for param in list(params)[:3]:
                    param_data[param] = [entry['parameters'][param] for entry in optimization_history 
                                       if 'parameters' in entry and param in entry['parameters']]
                
                for i, (param, values) in enumerate(param_data.items()):
                    fig.add_trace(
                        go.Histogram(x=values, name=param, opacity=0.7),
                        row=1, col=2
                    )
        
        fig.update_layout(
            template=self.plotly_template,
            title="Optimization Dashboard",
            height=800,
            showlegend=True
        )
        
        filepath = self.output_dir / f"{save_filename}_interactive_dashboard.html"
        fig.write_html(str(filepath))
        
        self.logger.info(f"Saved interactive dashboard: {filepath}")
        return str(filepath)
    
    def create_statistical_distribution_plot(self, data: Dict[str, List[float]], 
                                           save_filename: str) -> str:
        n_vars = len(data)
        fig, axes = plt.subplots(2, (n_vars + 1) // 2, figsize=(16, 10))
        
        if n_vars == 1:
            axes = [axes]
        elif n_vars <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (var_name, values) in enumerate(data.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            ax.hist(values, bins=20, density=True, alpha=0.7, 
                   color=self.aerospace_colors['primary_blue'], 
                   edgecolor='black', label='Data')
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            x = np.linspace(min(values), max(values), 100)
            y = (1/(std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_val) / std_val) ** 2)
            ax.plot(x, y, color=self.aerospace_colors['accent_orange'], 
                   linewidth=2, label='Normal Fit')
            
            percentiles = [5, 25, 50, 75, 95]
            perc_values = np.percentile(values, percentiles)
            
            colors_perc = [self.aerospace_colors['accent_red'], 
                          self.aerospace_colors['medium_gray'],
                          self.aerospace_colors['dark_gray'],
                          self.aerospace_colors['medium_gray'],
                          self.aerospace_colors['accent_red']]
            
            for perc, val, color in zip(percentiles, perc_values, colors_perc):
                ax.axvline(val, color=color, linestyle='--', alpha=0.7, 
                          label=f'{perc}th percentile')
            
            ax.set_xlabel(var_name.replace('_', ' ').title())
            ax.set_ylabel('Probability Density')
            ax.set_title(f'Statistical Distribution - {var_name.replace("_", " ").title()}', 
                        fontweight='bold')
            
            textstr = f'μ = {mean_val:.3f}\nσ = {std_val:.3f}'
            props = dict(boxstyle='round', facecolor=self.aerospace_colors['light_gray'], alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            ax.grid(True, alpha=0.3)
        
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{save_filename}_statistical_distributions.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Saved statistical distribution plot: {filepath}")
        return str(filepath)
    
    def _find_pareto_front(self, obj1_values: List[float], obj2_values: List[float]) -> List[Tuple[float, float]]:
        points = list(zip(obj1_values, obj2_values))
        pareto_front = []
        
        for i, point in enumerate(points):
            is_dominated = False
            for j, other_point in enumerate(points):
                if i != j:
                    if (other_point[0] >= point[0] and other_point[1] >= point[1] and 
                        (other_point[0] > point[0] or other_point[1] > point[1])):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_front.append(point)
        
        pareto_front.sort(key=lambda x: x[0])
        return pareto_front
    
    def generate_comprehensive_report(self, optimization_results: List[Dict[str, Any]], 
                                    report_name: str) -> Dict[str, str]:
        generated_plots = {}
        
        for i, result in enumerate(optimization_results):
            algorithm_name = result.get('algorithm_name', f'Algorithm_{i}')
            
            if 'optimization_history' in result:
                conv_plot = self.create_convergence_plot(
                    result['optimization_history'], 
                    algorithm_name, 
                    f"{report_name}_{algorithm_name}_convergence"
                )
                generated_plots[f'{algorithm_name}_convergence'] = conv_plot
            
            if 'monte_carlo_results' in result:
                unc_plot = self.create_uncertainty_propagation_plot(
                    result['monte_carlo_results'],
                    f"{report_name}_{algorithm_name}_uncertainty"
                )
                generated_plots[f'{algorithm_name}_uncertainty'] = unc_plot
        
        if len(optimization_results) > 1:
            comparison_data = {}
            for result in optimization_results:
                algorithm_name = result.get('algorithm_name', 'Unknown')
                if 'best_objectives' in result:
                    comparison_data[algorithm_name] = {
                        'objectives': {
                            obj_name: {'mean': obj_value, 'std': 0.0}
                            for obj_name, obj_value in result['best_objectives'].items()
                        }
                    }
            
            if comparison_data:
                comp_plot = self.create_performance_comparison_plot(
                    comparison_data, f"{report_name}_comparison"
                )
                generated_plots['performance_comparison'] = comp_plot
        
        self.logger.info(f"Generated comprehensive report with {len(generated_plots)} plots")
        return generated_plots