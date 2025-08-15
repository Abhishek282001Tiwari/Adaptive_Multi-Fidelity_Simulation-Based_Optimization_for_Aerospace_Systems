#!/usr/bin/env python3
"""
Cost Savings Dashboard
Adaptive Multi-Fidelity Simulation-Based Optimization for Aerospace Systems

Creates comprehensive cost savings visualizations and efficiency dashboards
with financial impact analysis and computational resource optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import json
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
    'savings_green': '#00AA44',
    'cost_red': '#CC3333'
}

class CostSavingsDashboard:
    """Professional cost savings dashboard and financial analysis system."""
    
    def __init__(self, results_dir='../results', output_dir='./plots'):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set professional matplotlib style
        plt.style.use('default')
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'figure.figsize': (16, 12),
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
        
        # Cost parameters (representative values)
        self.cost_per_hour = {
            'high_fidelity': 500,  # $/hour for high-fidelity simulation
            'medium_fidelity': 150,  # $/hour for medium-fidelity simulation
            'low_fidelity': 25,    # $/hour for low-fidelity simulation
            'engineer_time': 75    # $/hour for engineer time
        }
    
    def load_all_data(self):
        """Load all relevant data for cost analysis."""
        # Load performance comparison data
        performance_file = self.results_dir / 'performance_comparison.csv'
        performance_df = pd.read_csv(performance_file)
        
        # Load fidelity switching logs
        fidelity_file = self.results_dir / 'fidelity_switching_logs.csv'
        fidelity_df = pd.read_csv(fidelity_file)
        
        # Load optimization results
        aircraft_file = self.results_dir / 'aircraft_optimization_results.json'
        spacecraft_file = self.results_dir / 'spacecraft_optimization_results.json'
        
        with open(aircraft_file, 'r') as f:
            aircraft_data = json.load(f)
        
        with open(spacecraft_file, 'r') as f:
            spacecraft_data = json.load(f)
            
        return performance_df, fidelity_df, aircraft_data, spacecraft_data
    
    def calculate_financial_metrics(self, performance_df, fidelity_df):
        """Calculate detailed financial metrics and cost savings."""
        financial_data = []
        
        for _, row in performance_df.iterrows():
            # Calculate actual costs based on fidelity usage
            low_hours = row['total_evaluations'] * row['fidelity_low_percent'] / 100 * 0.1  # 0.1 hour per low eval
            medium_hours = row['total_evaluations'] * row['fidelity_medium_percent'] / 100 * 0.5  # 0.5 hour per medium eval
            high_hours = row['total_evaluations'] * row['fidelity_high_percent'] / 100 * 2.0  # 2.0 hours per high eval
            
            actual_cost = (low_hours * self.cost_per_hour['low_fidelity'] + 
                          medium_hours * self.cost_per_hour['medium_fidelity'] + 
                          high_hours * self.cost_per_hour['high_fidelity'])
            
            # Calculate high-fidelity only cost
            high_fidelity_only_cost = row['total_evaluations'] * 2.0 * self.cost_per_hour['high_fidelity']
            
            # Calculate savings
            absolute_savings = high_fidelity_only_cost - actual_cost
            relative_savings = row['cost_savings_percent'] / 100
            
            # Add engineer time cost
            engineer_hours = row['convergence_time_seconds'] / 3600  # Convert to hours
            engineer_cost = engineer_hours * self.cost_per_hour['engineer_time']
            
            total_project_cost = actual_cost + engineer_cost
            
            financial_data.append({
                'run_id': row['run_id'],
                'algorithm': row['algorithm'],
                'problem_type': row['problem_type'],
                'actual_cost': actual_cost,
                'high_fidelity_cost': high_fidelity_only_cost,
                'absolute_savings': absolute_savings,
                'relative_savings': relative_savings,
                'engineer_cost': engineer_cost,
                'total_project_cost': total_project_cost,
                'roi_percentage': (absolute_savings / high_fidelity_only_cost) * 100
            })
        
        return pd.DataFrame(financial_data)
    
    def create_executive_dashboard(self, financial_df, performance_df):
        """Create executive summary dashboard with key financial metrics."""
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        fig.suptitle('Executive Cost Savings Dashboard\nAdaptive Multi-Fidelity Optimization Framework', 
                    fontsize=18, fontweight='bold', y=0.96)
        
        # 1. Total Savings Summary (Top Left - Large)
        ax1 = fig.add_subplot(gs[0, :2])
        
        total_savings = financial_df['absolute_savings'].sum()
        total_potential_cost = financial_df['high_fidelity_cost'].sum()
        overall_savings_rate = (total_savings / total_potential_cost) * 100
        
        # Create gauge-style visualization
        theta = np.linspace(0, np.pi, 100)
        radius = 1
        
        # Background arc
        ax1.plot(radius * np.cos(theta), radius * np.sin(theta), 
                color=AEROSPACE_COLORS['light_gray'], linewidth=15, alpha=0.3)
        
        # Savings arc
        savings_theta = np.linspace(0, np.pi * (overall_savings_rate / 100), 100)
        ax1.plot(radius * np.cos(savings_theta), radius * np.sin(savings_theta),
                color=AEROSPACE_COLORS['savings_green'], linewidth=15)
        
        # Center text
        ax1.text(0, 0.3, f'${total_savings/1000:.0f}K', 
                ha='center', va='center', fontsize=24, fontweight='bold',
                color=AEROSPACE_COLORS['savings_green'])
        ax1.text(0, 0.1, 'Total Savings', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax1.text(0, -0.1, f'{overall_savings_rate:.1f}% Reduction', 
                ha='center', va='center', fontsize=12)
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-0.2, 1.2)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title('Overall Cost Reduction', fontsize=16, fontweight='bold', pad=20)
        
        # 2. Savings by Algorithm (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        
        algo_savings = financial_df.groupby('algorithm')['absolute_savings'].sum().sort_values(ascending=True)
        
        colors = [AEROSPACE_COLORS['primary_blue'], AEROSPACE_COLORS['secondary_blue'],
                 AEROSPACE_COLORS['accent_orange'], AEROSPACE_COLORS['success_green']][:len(algo_savings)]
        
        bars = ax2.barh(range(len(algo_savings)), algo_savings.values / 1000,
                       color=colors, alpha=0.8)
        
        ax2.set_yticks(range(len(algo_savings)))
        ax2.set_yticklabels([alg.replace('Optimization', '').replace('Algorithm', '') 
                            for alg in algo_savings.index])
        ax2.set_xlabel('Savings ($K)')
        ax2.set_title('Cost Savings by Algorithm', fontweight='bold', fontsize=14)
        
        # Add value labels
        for bar, value in zip(bars, algo_savings.values):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'${value/1000:.0f}K', ha='left', va='center', fontweight='bold')
        
        # 3. Cost Breakdown Pie Chart
        ax3 = fig.add_subplot(gs[1, 0])
        
        total_low_cost = financial_df['actual_cost'].sum() * 0.15  # Estimated breakdown
        total_medium_cost = financial_df['actual_cost'].sum() * 0.35
        total_high_cost = financial_df['actual_cost'].sum() * 0.50
        
        costs = [total_low_cost, total_medium_cost, total_high_cost]
        labels = ['Low Fidelity', 'Medium Fidelity', 'High Fidelity']
        colors_pie = [AEROSPACE_COLORS['success_green'], AEROSPACE_COLORS['warning_amber'], 
                     AEROSPACE_COLORS['error_red']]
        
        wedges, texts, autotexts = ax3.pie(costs, labels=labels, colors=colors_pie,
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('Cost Distribution\nby Fidelity Level', fontweight='bold', fontsize=12)
        
        # 4. ROI Analysis
        ax4 = fig.add_subplot(gs[1, 1])
        
        roi_data = financial_df.groupby('algorithm')['roi_percentage'].mean().sort_values(ascending=False)
        
        bars = ax4.bar(range(len(roi_data)), roi_data.values,
                      color=[AEROSPACE_COLORS['primary_blue'], AEROSPACE_COLORS['secondary_blue'],
                            AEROSPACE_COLORS['accent_orange'], AEROSPACE_COLORS['success_green']][:len(roi_data)],
                      alpha=0.8)
        
        ax4.set_xticks(range(len(roi_data)))
        ax4.set_xticklabels([alg.replace('Optimization', '').replace('Algorithm', '')[:8] 
                            for alg in roi_data.index], rotation=45, ha='right')
        ax4.set_ylabel('ROI (%)')
        ax4.set_title('Return on Investment\nby Algorithm', fontweight='bold', fontsize=12)
        
        # Add percentage labels
        for bar, value in zip(bars, roi_data.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        # 5. Monthly Projection
        ax5 = fig.add_subplot(gs[1, 2:])
        
        # Project monthly savings assuming similar usage
        monthly_projects = 20  # Assumed number of optimization projects per month
        monthly_savings = total_savings * monthly_projects / len(financial_df)
        annual_savings = monthly_savings * 12
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        cumulative_savings = [monthly_savings * (i + 1) for i in range(12)]
        
        ax5.plot(months, np.array(cumulative_savings) / 1000, 
                marker='o', linewidth=3, markersize=8,
                color=AEROSPACE_COLORS['savings_green'], alpha=0.8)
        ax5.fill_between(months, np.array(cumulative_savings) / 1000, alpha=0.3,
                        color=AEROSPACE_COLORS['savings_green'])
        
        ax5.set_ylabel('Cumulative Savings ($K)')
        ax5.set_title(f'Projected Annual Savings: ${annual_savings/1000:.0f}K', 
                     fontweight='bold', fontsize=12)
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        
        # 6. Problem Type Cost Analysis
        ax6 = fig.add_subplot(gs[2, 0])
        
        problem_savings = financial_df.groupby('problem_type')['relative_savings'].mean() * 100
        problem_categories = []
        category_savings = []
        
        for prob_type, savings in problem_savings.items():
            if 'aircraft' in prob_type:
                category = 'Aircraft'
            elif 'spacecraft' in prob_type:
                category = 'Spacecraft'
            elif 'benchmark' in prob_type:
                category = 'Benchmark'
            else:
                category = 'Other'
            
            if category not in category_savings:
                problem_categories.append(category)
                category_savings.append(savings)
        
        bars = ax6.bar(problem_categories, category_savings,
                      color=[AEROSPACE_COLORS['primary_blue'], AEROSPACE_COLORS['accent_orange'],
                            AEROSPACE_COLORS['success_green'], AEROSPACE_COLORS['dark_gray']][:len(problem_categories)],
                      alpha=0.8)
        
        ax6.set_ylabel('Average Savings (%)')
        ax6.set_title('Savings by Problem\nCategory', fontweight='bold', fontsize=12)
        
        # 7. Cost vs Performance Trade-off
        ax7 = fig.add_subplot(gs[2, 1])
        
        scatter = ax7.scatter(financial_df['total_project_cost'] / 1000, 
                             performance_df['solution_quality'],
                             c=financial_df['relative_savings'], s=80, alpha=0.7,
                             cmap='RdYlGn', edgecolors='black', linewidth=0.5)
        
        ax7.set_xlabel('Total Project Cost ($K)')
        ax7.set_ylabel('Solution Quality')
        ax7.set_title('Cost vs Quality\nTrade-off', fontweight='bold', fontsize=12)
        
        cbar = plt.colorbar(scatter, ax=ax7)
        cbar.set_label('Savings Rate', rotation=270, labelpad=15)
        
        # 8. Key Performance Indicators
        ax8 = fig.add_subplot(gs[2, 2:])
        ax8.axis('off')
        
        # Calculate KPIs
        avg_project_cost = financial_df['total_project_cost'].mean()
        avg_savings_rate = financial_df['relative_savings'].mean() * 100
        total_projects = len(financial_df)
        avg_roi = financial_df['roi_percentage'].mean()
        
        kpis = [
            ('Average Project Cost', f'${avg_project_cost/1000:.1f}K'),
            ('Average Savings Rate', f'{avg_savings_rate:.1f}%'),
            ('Total Projects Analyzed', f'{total_projects}'),
            ('Average ROI', f'{avg_roi:.0f}%'),
            ('Framework Efficiency', f'{performance_df["computational_efficiency"].mean():.2f}'),
            ('Success Rate', f'{performance_df["success_rate"].mean():.1%}')
        ]
        
        # Create KPI boxes
        box_height = 0.8 / len(kpis)
        for i, (kpi_name, kpi_value) in enumerate(kpis):
            y_pos = 0.9 - i * (box_height + 0.02)
            
            # Background rectangle
            rect = Rectangle((0.05, y_pos - box_height/2), 0.9, box_height,
                           facecolor=AEROSPACE_COLORS['light_gray'], alpha=0.3)
            ax8.add_patch(rect)
            
            # KPI text
            ax8.text(0.1, y_pos, kpi_name, fontsize=11, va='center', fontweight='bold')
            ax8.text(0.85, y_pos, kpi_value, fontsize=11, va='center', 
                    ha='right', fontweight='bold', color=AEROSPACE_COLORS['primary_blue'])
        
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.set_title('Key Performance Indicators', fontweight='bold', fontsize=14, y=0.95)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'executive_cost_dashboard.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'executive_cost_dashboard.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_detailed_cost_analysis(self, financial_df, fidelity_df):
        """Create detailed cost analysis and resource utilization charts."""
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        fig.suptitle('Detailed Cost Analysis and Resource Optimization\nAdaptive Multi-Fidelity Framework', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # 1. Cost Savings Evolution
        ax = axes[0, 0]
        
        # Group by algorithm and show cost evolution
        for algorithm in financial_df['algorithm'].unique():
            alg_data = financial_df[financial_df['algorithm'] == algorithm].sort_values('run_id')
            
            ax.plot(range(len(alg_data)), alg_data['relative_savings'] * 100,
                   marker='o', linewidth=2.5, alpha=0.8, markersize=6,
                   label=algorithm.replace('Optimization', '').replace('Algorithm', ''))
        
        ax.set_xlabel('Project Sequence')
        ax.set_ylabel('Cost Savings (%)')
        ax.set_title('Cost Savings Evolution\nby Algorithm', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Resource Utilization Breakdown
        ax = axes[0, 1]
        
        # Calculate resource usage by algorithm
        algo_resource = financial_df.groupby('algorithm')[['actual_cost', 'engineer_cost']].mean()
        
        x = np.arange(len(algo_resource))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, algo_resource['actual_cost'] / 1000, width,
                      label='Computational Cost', color=AEROSPACE_COLORS['primary_blue'], alpha=0.8)
        bars2 = ax.bar(x + width/2, algo_resource['engineer_cost'] / 1000, width,
                      label='Engineering Cost', color=AEROSPACE_COLORS['accent_orange'], alpha=0.8)
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Average Cost ($K)')
        ax.set_title('Resource Utilization\nBreakdown', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([alg.replace('Optimization', '').replace('Algorithm', '')[:8] 
                           for alg in algo_resource.index], rotation=45, ha='right')
        ax.legend()
        
        # 3. Fidelity Cost Distribution
        ax = axes[1, 0]
        
        # Calculate cost distribution across fidelity levels
        fidelity_usage = fidelity_df.groupby('current_fidelity')['computational_cost_seconds'].sum()
        
        # Convert to financial costs
        fidelity_costs = {}
        for fidelity, time_seconds in fidelity_usage.items():
            hours = time_seconds / 3600
            cost = hours * self.cost_per_hour[f'{fidelity}_fidelity']
            fidelity_costs[fidelity] = cost
        
        fidelities = list(fidelity_costs.keys())
        costs = list(fidelity_costs.values())
        
        colors = [AEROSPACE_COLORS['success_green'], AEROSPACE_COLORS['warning_amber'], 
                 AEROSPACE_COLORS['error_red']]
        
        wedges, texts, autotexts = ax.pie(costs, labels=[f.title() for f in fidelities], 
                                         colors=colors, autopct='%1.1f%%', startangle=90)
        
        ax.set_title('Computational Cost\nDistribution by Fidelity', fontweight='bold')
        
        # 4. Cost-Benefit Analysis
        ax = axes[1, 1]
        
        # Scatter plot of cost vs benefit
        benefits = financial_df['absolute_savings']
        investments = financial_df['total_project_cost']
        
        scatter = ax.scatter(investments / 1000, benefits / 1000,
                           c=financial_df['roi_percentage'], s=100, alpha=0.7,
                           cmap='RdYlGn', edgecolors='black', linewidth=0.5)
        
        # Add break-even line
        max_val = max(investments.max(), benefits.max()) / 1000
        ax.plot([0, max_val], [0, max_val], '--', color=AEROSPACE_COLORS['dark_gray'], 
               linewidth=2, alpha=0.8, label='Break-even Line')
        
        ax.set_xlabel('Total Investment ($K)')
        ax.set_ylabel('Absolute Savings ($K)')
        ax.set_title('Cost-Benefit Analysis\nInvestment vs Returns', fontweight='bold')
        ax.legend()
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('ROI (%)', rotation=270, labelpad=15)
        
        # 5. Efficiency Trends
        ax = axes[2, 0]
        
        # Calculate efficiency metrics over time
        efficiency_data = financial_df.copy()
        efficiency_data['efficiency_ratio'] = efficiency_data['absolute_savings'] / efficiency_data['total_project_cost']
        
        problem_efficiency = efficiency_data.groupby('problem_type')['efficiency_ratio'].mean().sort_values(ascending=False)
        
        bars = ax.barh(range(len(problem_efficiency)), problem_efficiency.values,
                      color=AEROSPACE_COLORS['success_green'], alpha=0.8)
        
        ax.set_yticks(range(len(problem_efficiency)))
        ax.set_yticklabels([prob.replace('_', ' ').title()[:15] for prob in problem_efficiency.index])
        ax.set_xlabel('Efficiency Ratio (Savings/Investment)')
        ax.set_title('Problem Type\nEfficiency Ranking', fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, problem_efficiency.values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.2f}', ha='left', va='center', fontweight='bold')
        
        # 6. Financial Projections
        ax = axes[2, 1]
        
        # Create scenario analysis
        scenarios = ['Conservative', 'Realistic', 'Optimistic']
        multipliers = [0.7, 1.0, 1.3]
        
        base_monthly_savings = financial_df['absolute_savings'].mean() * 20  # 20 projects/month
        
        years = np.arange(1, 6)  # 5-year projection
        
        for scenario, multiplier in zip(scenarios, multipliers):
            annual_savings = base_monthly_savings * 12 * multiplier
            cumulative_savings = [annual_savings * year for year in years]
            
            color = AEROSPACE_COLORS['error_red'] if scenario == 'Conservative' else \
                   AEROSPACE_COLORS['primary_blue'] if scenario == 'Realistic' else \
                   AEROSPACE_COLORS['success_green']
            
            ax.plot(years, np.array(cumulative_savings) / 1000000, 
                   marker='o', linewidth=2.5, alpha=0.8, color=color, label=scenario)
        
        ax.set_xlabel('Years')
        ax.set_ylabel('Cumulative Savings ($M)')
        ax.set_title('5-Year Financial\nProjections', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_cost_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'detailed_cost_analysis.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_cost_savings_summary(self, financial_df, performance_df):
        """Generate comprehensive cost savings analysis summary."""
        total_savings = financial_df['absolute_savings'].sum()
        total_investment = financial_df['total_project_cost'].sum()
        avg_roi = financial_df['roi_percentage'].mean()
        
        summary_stats = {
            'Executive Summary': {
                'Total Cost Savings': f'${total_savings/1000:.0f}K',
                'Total Investment': f'${total_investment/1000:.0f}K',
                'Overall ROI': f'{avg_roi:.0f}%',
                'Average Savings Rate': f'{financial_df["relative_savings"].mean() * 100:.1f}%',
                'Projects Analyzed': len(financial_df)
            },
            'Algorithm Performance': {},
            'Resource Utilization': {
                'Average Computational Cost': f'${financial_df["actual_cost"].mean()/1000:.1f}K',
                'Average Engineering Cost': f'${financial_df["engineer_cost"].mean()/1000:.1f}K',
                'Most Cost-Effective Algorithm': financial_df.loc[financial_df['roi_percentage'].idxmax(), 'algorithm'],
                'Highest Absolute Savings': financial_df.loc[financial_df['absolute_savings'].idxmax(), 'algorithm']
            },
            'Financial Projections': {
                'Monthly Savings Potential': f'${total_savings * 20 / len(financial_df) / 1000:.0f}K',
                'Annual Savings Potential': f'${total_savings * 240 / len(financial_df) / 1000:.0f}K',
                'Break-even Point': f'{total_investment / total_savings:.1f} months'
            }
        }
        
        # Algorithm-specific analysis
        for algorithm in financial_df['algorithm'].unique():
            alg_data = financial_df[financial_df['algorithm'] == algorithm]
            
            summary_stats['Algorithm Performance'][algorithm] = {
                'Projects': len(alg_data),
                'Average Savings': f'${alg_data["absolute_savings"].mean()/1000:.1f}K',
                'Average ROI': f'{alg_data["roi_percentage"].mean():.0f}%',
                'Savings Rate': f'{alg_data["relative_savings"].mean() * 100:.1f}%',
                'Total Investment': f'${alg_data["total_project_cost"].sum()/1000:.0f}K'
            }
        
        # Save summary as JSON
        with open(self.output_dir / 'cost_savings_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Cost savings analysis complete!")
        print(f"Generated dashboards saved to: {self.output_dir}")
        print(f"Financial Summary:")
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
    """Main function to generate cost savings dashboard."""
    dashboard = CostSavingsDashboard()
    
    # Load all data
    performance_df, fidelity_df, aircraft_data, spacecraft_data = dashboard.load_all_data()
    
    print("Generating cost savings dashboard...")
    
    # Calculate financial metrics
    financial_df = dashboard.calculate_financial_metrics(performance_df, fidelity_df)
    
    # Generate dashboards
    dashboard.create_executive_dashboard(financial_df, performance_df)
    print("✓ Executive cost dashboard generated")
    
    dashboard.create_detailed_cost_analysis(financial_df, fidelity_df)
    print("✓ Detailed cost analysis generated")
    
    dashboard.generate_cost_savings_summary(financial_df, performance_df)
    print("✓ Cost savings summary generated")

if __name__ == "__main__":
    main()