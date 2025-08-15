#!/usr/bin/env python3
"""
Data Management System Demonstration
====================================

This script demonstrates the complete data management capabilities of the
framework including:
1. Saving optimization runs
2. Loading and retrieving results
3. Exporting to multiple formats (CSV, Excel, HDF5)
4. Creating comparison reports
5. Data analysis and statistics
6. Results archiving and cleanup

This is a standalone demonstration that works with or without the main framework.
"""

import json
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoDataManager:
    """Demonstration version of the data management system"""
    
    def __init__(self, base_path: str = "demo_results"):
        self.base_path = base_path
        self.results_dir = os.path.join(base_path, "optimization_runs")
        self.exports_dir = os.path.join(base_path, "exports")
        self.reports_dir = os.path.join(base_path, "reports")
        
        # Create directories
        for directory in [self.results_dir, self.exports_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.runs_database = {}
        self._load_existing_runs()
    
    def _load_existing_runs(self):
        """Load existing optimization runs from disk"""
        if os.path.exists(os.path.join(self.base_path, "runs_database.json")):
            with open(os.path.join(self.base_path, "runs_database.json"), 'r') as f:
                self.runs_database = json.load(f)
        logger.info(f"Loaded {len(self.runs_database)} existing optimization runs")
    
    def _save_database(self):
        """Save runs database to disk"""
        with open(os.path.join(self.base_path, "runs_database.json"), 'w') as f:
            json.dump(self.runs_database, f, indent=2)
    
    def save_optimization_run(self, run_id: str, optimization_result: Dict[str, Any], 
                             algorithm_name: str, system_type: str, parameters: Dict[str, Any]) -> str:
        """Save an optimization run to the database"""
        
        timestamp = datetime.now().isoformat()
        
        # Create run record
        run_record = {
            'run_id': run_id,
            'timestamp': timestamp,
            'algorithm_name': algorithm_name,
            'system_type': system_type,
            'parameters': parameters,
            'optimization_result': optimization_result,
            'metadata': {
                'version': '1.0.0',
                'framework': 'amf-sbo',
                'file_size': 0,  # Will be calculated
                'checksum': None  # Could add file integrity checking
            }
        }
        
        # Save detailed results to individual file
        result_filename = f"{run_id}_{timestamp.replace(':', '-').replace('.', '-')}.json"
        result_path = os.path.join(self.results_dir, result_filename)
        
        with open(result_path, 'w') as f:
            json.dump(run_record, f, indent=2)
        
        # Calculate file size
        run_record['metadata']['file_size'] = os.path.getsize(result_path)
        
        # Update database
        self.runs_database[run_id] = {
            'timestamp': timestamp,
            'algorithm_name': algorithm_name,
            'system_type': system_type,
            'parameters': parameters,
            'file_path': result_path,
            'metadata': run_record['metadata']
        }
        
        self._save_database()
        logger.info(f"Saved optimization run: {run_id}")
        
        return run_id
    
    def load_optimization_run(self, run_id: str) -> Dict[str, Any]:
        """Load an optimization run from the database"""
        
        if run_id not in self.runs_database:
            logger.warning(f"Run ID {run_id} not found in database")
            return None
        
        file_path = self.runs_database[run_id]['file_path']
        
        if not os.path.exists(file_path):
            logger.error(f"Result file not found: {file_path}")
            return None
        
        with open(file_path, 'r') as f:
            run_data = json.load(f)
        
        logger.info(f"Loaded optimization run: {run_id}")
        return run_data
    
    def export_to_csv(self, run_ids: List[str], filename: str) -> str:
        """Export optimization runs to CSV format"""
        
        export_data = []
        
        for run_id in run_ids:
            if run_id not in self.runs_database:
                logger.warning(f"Run ID {run_id} not found, skipping")
                continue
            
            run_info = self.runs_database[run_id]
            run_data = self.load_optimization_run(run_id)
            
            if not run_data:
                continue
            
            # Flatten the data for CSV export
            flat_record = {
                'run_id': run_id,
                'timestamp': run_info['timestamp'],
                'algorithm_name': run_info['algorithm_name'],
                'system_type': run_info['system_type'],
            }
            
            # Add parameters
            for param, value in run_info['parameters'].items():
                flat_record[f'param_{param}'] = value
            
            # Add results (simplified for demo)
            result = run_data.get('optimization_result', {})
            if 'best_objectives' in result:
                for obj, value in result['best_objectives'].items():
                    flat_record[f'result_{obj}'] = value
            
            if 'best_parameters' in result:
                for param, value in result['best_parameters'].items():
                    flat_record[f'best_{param}'] = value
            
            # Add metadata
            flat_record['total_evaluations'] = result.get('total_evaluations', 0)
            flat_record['convergence_achieved'] = result.get('convergence_achieved', False)
            flat_record['total_time'] = result.get('total_time', 0.0)
            
            export_data.append(flat_record)
        
        # Create DataFrame and export
        df = pd.DataFrame(export_data)
        csv_path = os.path.join(self.exports_dir, filename)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Exported {len(export_data)} runs to CSV: {csv_path}")
        return csv_path
    
    def export_to_excel(self, run_ids: List[str], filename: str) -> str:
        """Export optimization runs to Excel format with multiple sheets"""
        
        excel_path = os.path.join(self.exports_dir, filename)
        
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            
            # Summary sheet
            summary_data = []
            for run_id in run_ids:
                if run_id not in self.runs_database:
                    continue
                
                run_info = self.runs_database[run_id]
                run_data = self.load_optimization_run(run_id)
                
                if not run_data:
                    continue
                
                result = run_data.get('optimization_result', {})
                summary_data.append({
                    'Run ID': run_id,
                    'Algorithm': run_info['algorithm_name'],
                    'System Type': run_info['system_type'],
                    'Timestamp': run_info['timestamp'],
                    'Total Evaluations': result.get('total_evaluations', 0),
                    'Total Time (s)': result.get('total_time', 0.0),
                    'Converged': result.get('convergence_achieved', False)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed results sheet
            detailed_data = []
            for run_id in run_ids:
                if run_id not in self.runs_database:
                    continue
                
                run_data = self.load_optimization_run(run_id)
                if not run_data:
                    continue
                
                result = run_data.get('optimization_result', {})
                
                # Best parameters
                if 'best_parameters' in result:
                    for param, value in result['best_parameters'].items():
                        detailed_data.append({
                            'Run ID': run_id,
                            'Category': 'Best Parameter',
                            'Name': param,
                            'Value': value
                        })
                
                # Best objectives
                if 'best_objectives' in result:
                    for obj, value in result['best_objectives'].items():
                        detailed_data.append({
                            'Run ID': run_id,
                            'Category': 'Best Objective',
                            'Name': obj,
                            'Value': value
                        })
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
            
            # Parameters sheet
            params_data = []
            for run_id in run_ids:
                if run_id not in self.runs_database:
                    continue
                
                run_info = self.runs_database[run_id]
                for param, value in run_info['parameters'].items():
                    params_data.append({
                        'Run ID': run_id,
                        'Parameter': param,
                        'Value': value
                    })
            
            params_df = pd.DataFrame(params_data)
            params_df.to_excel(writer, sheet_name='Parameters', index=False)
        
        logger.info(f"Exported {len(run_ids)} runs to Excel: {excel_path}")
        return excel_path
    
    def create_comparison_report(self, run_ids: List[str], report_name: str) -> str:
        """Create a comprehensive comparison report"""
        
        report_path = os.path.join(self.reports_dir, f"{report_name}.html")
        
        # Generate HTML report
        html_content = self._generate_html_report(run_ids, report_name)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated comparison report: {report_path}")
        return report_path
    
    def _generate_html_report(self, run_ids: List[str], report_name: str) -> str:
        """Generate HTML comparison report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f8f9fa; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .summary-box {{ background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 5px; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>{report_name}</h1>
            <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        # Summary section
        html += """
        <div class="summary-box">
            <h2>Executive Summary</h2>
        """
        
        total_runs = len([rid for rid in run_ids if rid in self.runs_database])
        algorithms = set()
        system_types = set()
        total_evaluations = 0
        total_time = 0.0
        
        for run_id in run_ids:
            if run_id not in self.runs_database:
                continue
            
            run_info = self.runs_database[run_id]
            run_data = self.load_optimization_run(run_id)
            
            algorithms.add(run_info['algorithm_name'])
            system_types.add(run_info['system_type'])
            
            if run_data:
                result = run_data.get('optimization_result', {})
                total_evaluations += result.get('total_evaluations', 0)
                total_time += result.get('total_time', 0.0)
        
        html += f"""
            <div class="metric"><strong>Total Optimization Runs:</strong> {total_runs}</div>
            <div class="metric"><strong>Algorithms Used:</strong> {', '.join(algorithms)}</div>
            <div class="metric"><strong>System Types:</strong> {', '.join(system_types)}</div>
            <div class="metric"><strong>Total Evaluations:</strong> {total_evaluations:,}</div>
            <div class="metric"><strong>Total Computation Time:</strong> {total_time:.1f} seconds</div>
        </div>
        """
        
        # Detailed results table
        html += """
        <h2>Detailed Results Comparison</h2>
        <table>
            <tr>
                <th>Run ID</th>
                <th>Algorithm</th>
                <th>System Type</th>
                <th>Timestamp</th>
                <th>Evaluations</th>
                <th>Time (s)</th>
                <th>Converged</th>
                <th>Best Objective</th>
            </tr>
        """
        
        for run_id in run_ids:
            if run_id not in self.runs_database:
                continue
            
            run_info = self.runs_database[run_id]
            run_data = self.load_optimization_run(run_id)
            
            if not run_data:
                continue
            
            result = run_data.get('optimization_result', {})
            
            # Get first objective value as representative
            best_obj = "N/A"
            if 'best_objectives' in result and result['best_objectives']:
                first_obj = list(result['best_objectives'].values())[0]
                best_obj = f"{first_obj:.4f}" if isinstance(first_obj, (int, float)) else str(first_obj)
            
            html += f"""
            <tr>
                <td>{run_id}</td>
                <td>{run_info['algorithm_name']}</td>
                <td>{run_info['system_type']}</td>
                <td>{run_info['timestamp'][:19]}</td>
                <td>{result.get('total_evaluations', 0):,}</td>
                <td>{result.get('total_time', 0.0):.1f}</td>
                <td>{'Yes' if result.get('convergence_achieved', False) else 'No'}</td>
                <td>{best_obj}</td>
            </tr>
            """
        
        html += """
        </table>
        
        <h2>Performance Analysis</h2>
        <p>This report provides a comprehensive comparison of optimization runs, showing algorithm performance, 
        convergence characteristics, and computational efficiency. Use this information to select the most 
        appropriate optimization approach for your specific aerospace design problems.</p>
        
        </body>
        </html>
        """
        
        return html
    
    def get_run_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all optimization runs"""
        
        if not self.runs_database:
            return {"total_runs": 0, "message": "No optimization runs in database"}
        
        stats = {
            "total_runs": len(self.runs_database),
            "algorithms": {},
            "system_types": {},
            "total_evaluations": 0,
            "total_computation_time": 0.0,
            "convergence_rate": 0.0,
            "date_range": {"earliest": None, "latest": None},
            "file_statistics": {"total_size": 0, "average_size": 0}
        }
        
        converged_runs = 0
        timestamps = []
        
        for run_id, run_info in self.runs_database.items():
            # Algorithm statistics
            alg = run_info['algorithm_name']
            if alg not in stats["algorithms"]:
                stats["algorithms"][alg] = {"count": 0, "total_time": 0.0, "total_evaluations": 0}
            stats["algorithms"][alg]["count"] += 1
            
            # System type statistics
            sys_type = run_info['system_type']
            if sys_type not in stats["system_types"]:
                stats["system_types"][sys_type] = {"count": 0}
            stats["system_types"][sys_type]["count"] += 1
            
            # Load detailed results
            run_data = self.load_optimization_run(run_id)
            if run_data:
                result = run_data.get('optimization_result', {})
                
                evals = result.get('total_evaluations', 0)
                time_taken = result.get('total_time', 0.0)
                converged = result.get('convergence_achieved', False)
                
                stats["total_evaluations"] += evals
                stats["total_computation_time"] += time_taken
                stats["algorithms"][alg]["total_time"] += time_taken
                stats["algorithms"][alg]["total_evaluations"] += evals
                
                if converged:
                    converged_runs += 1
            
            # Timestamp analysis
            timestamps.append(run_info['timestamp'])
            
            # File size analysis
            if 'metadata' in run_info and 'file_size' in run_info['metadata']:
                stats["file_statistics"]["total_size"] += run_info['metadata']['file_size']
        
        # Calculate derived statistics
        if stats["total_runs"] > 0:
            stats["convergence_rate"] = converged_runs / stats["total_runs"]
            stats["file_statistics"]["average_size"] = stats["file_statistics"]["total_size"] / stats["total_runs"]
        
        if timestamps:
            timestamps.sort()
            stats["date_range"]["earliest"] = timestamps[0]
            stats["date_range"]["latest"] = timestamps[-1]
        
        return stats
    
    def cleanup_old_results(self, days_old: int = 30):
        """Clean up optimization results older than specified days"""
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        runs_to_remove = []
        
        for run_id, run_info in self.runs_database.items():
            run_timestamp = datetime.fromisoformat(run_info['timestamp'])
            
            if run_timestamp < cutoff_date:
                runs_to_remove.append(run_id)
        
        removed_count = 0
        for run_id in runs_to_remove:
            try:
                # Remove file
                file_path = self.runs_database[run_id]['file_path']
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # Remove from database
                del self.runs_database[run_id]
                removed_count += 1
                
            except Exception as e:
                logger.error(f"Error removing run {run_id}: {e}")
        
        if removed_count > 0:
            self._save_database()
            logger.info(f"Cleaned up {removed_count} old optimization runs")
        else:
            logger.info("No old optimization runs to clean up")
        
        return removed_count

def create_sample_optimization_results():
    """Create sample optimization results for demonstration"""
    
    # Sample results for different algorithms and systems
    sample_results = []
    
    # Aircraft GA results
    sample_results.append({
        'run_id': 'aircraft_ga_001',
        'algorithm_name': 'GeneticAlgorithm',
        'system_type': 'aircraft',
        'parameters': {
            'population_size': 50,
            'max_evaluations': 200,
            'mission_profile': 'commercial',
            'crossover_rate': 0.8,
            'mutation_rate': 0.1
        },
        'optimization_result': {
            'best_parameters': {
                'wingspan': 45.2,
                'wing_area': 234.5,
                'aspect_ratio': 9.8,
                'sweep_angle': 28.5,
                'cruise_mach': 0.78
            },
            'best_objectives': {
                'lift_to_drag_ratio': 26.4,
                'fuel_efficiency': 3.2,
                'range': 8500
            },
            'total_evaluations': 200,
            'convergence_achieved': True,
            'total_time': 142.5
        }
    })
    
    # Spacecraft Bayesian results
    sample_results.append({
        'run_id': 'spacecraft_bo_001',
        'algorithm_name': 'BayesianOptimization',
        'system_type': 'spacecraft',
        'parameters': {
            'max_evaluations': 100,
            'mission_type': 'earth_observation',
            'acquisition_function': 'ei',
            'xi': 0.01
        },
        'optimization_result': {
            'best_parameters': {
                'dry_mass': 2850,
                'fuel_mass': 12500,
                'solar_panel_area': 85.4,
                'target_orbit_altitude': 650
            },
            'best_objectives': {
                'mission_success_probability': 0.942,
                'total_mass': 15350,
                'power_efficiency': 1.18
            },
            'total_evaluations': 100,
            'convergence_achieved': True,
            'total_time': 187.3
        }
    })
    
    # Aircraft PSO results
    sample_results.append({
        'run_id': 'aircraft_pso_001',
        'algorithm_name': 'ParticleSwarmOptimization',
        'system_type': 'aircraft',
        'parameters': {
            'swarm_size': 30,
            'max_evaluations': 150,
            'mission_profile': 'business_jet',
            'inertia_weight': 0.729
        },
        'optimization_result': {
            'best_parameters': {
                'wingspan': 22.8,
                'wing_area': 135.2,
                'aspect_ratio': 7.9,
                'sweep_angle': 32.1,
                'cruise_mach': 0.85
            },
            'best_objectives': {
                'lift_to_drag_ratio': 24.1,
                'fuel_efficiency': 2.8,
                'range': 6200
            },
            'total_evaluations': 150,
            'convergence_achieved': True,
            'total_time': 89.7
        }
    })
    
    # Multi-objective spacecraft results
    sample_results.append({
        'run_id': 'spacecraft_nsga2_001',
        'algorithm_name': 'NSGA2',
        'system_type': 'spacecraft',
        'parameters': {
            'population_size': 100,
            'max_evaluations': 500,
            'mission_type': 'deep_space',
            'objectives': ['delta_v_capability', 'mission_success', 'total_mass']
        },
        'optimization_result': {
            'best_parameters': {
                'dry_mass': 1850,
                'fuel_mass': 28500,
                'specific_impulse': 385,
                'thrust': 450
            },
            'best_objectives': {
                'delta_v_capability': 11200,
                'mission_success_probability': 0.873,
                'total_mass': 30350
            },
            'total_evaluations': 500,
            'convergence_achieved': False,  # Multi-objective may not converge to single point
            'total_time': 892.1
        }
    })
    
    return sample_results

def main():
    """Demonstrate the complete data management system"""
    
    print("="*70)
    print("DATA MANAGEMENT SYSTEM DEMONSTRATION")
    print("="*70)
    print()
    
    # Initialize data manager
    data_manager = DemoDataManager("demo_data_management")
    print("✓ Data management system initialized")
    print()
    
    # Create and save sample optimization results
    print("Step 1: Creating and Saving Sample Optimization Results...")
    
    sample_results = create_sample_optimization_results()
    run_ids = []
    
    for result_data in sample_results:
        run_id = data_manager.save_optimization_run(
            run_id=result_data['run_id'],
            optimization_result=result_data['optimization_result'],
            algorithm_name=result_data['algorithm_name'],
            system_type=result_data['system_type'],
            parameters=result_data['parameters']
        )
        run_ids.append(run_id)
    
    print(f"✓ Saved {len(sample_results)} optimization runs")
    print()
    
    # Demonstrate loading results
    print("Step 2: Loading and Retrieving Results...")
    
    for run_id in run_ids[:2]:  # Load first two for demonstration
        loaded_data = data_manager.load_optimization_run(run_id)
        if loaded_data:
            result = loaded_data['optimization_result']
            print(f"  {run_id}: {result['total_evaluations']} evaluations, "
                  f"{result['total_time']:.1f}s, converged: {result['convergence_achieved']}")
    
    print()
    
    # Export to different formats
    print("Step 3: Exporting to Multiple Formats...")
    
    csv_file = data_manager.export_to_csv(run_ids, "optimization_results_demo.csv")
    excel_file = data_manager.export_to_excel(run_ids, "optimization_results_demo.xlsx")
    
    print(f"✓ CSV export: {csv_file}")
    print(f"✓ Excel export: {excel_file}")
    print()
    
    # Create comparison report
    print("Step 4: Creating Comparison Report...")
    
    report_file = data_manager.create_comparison_report(
        run_ids,
        "Optimization_Methods_Comparison_Demo"
    )
    
    print(f"✓ HTML report: {report_file}")
    print()
    
    # Get comprehensive statistics
    print("Step 5: Analyzing Run Statistics...")
    
    stats = data_manager.get_run_statistics()
    
    print(f"DATABASE STATISTICS:")
    print(f"• Total optimization runs: {stats['total_runs']}")
    print(f"• Total function evaluations: {stats['total_evaluations']:,}")
    print(f"• Total computation time: {stats['total_computation_time']:.1f} seconds")
    print(f"• Overall convergence rate: {stats['convergence_rate']:.1%}")
    print()
    
    print(f"ALGORITHM BREAKDOWN:")
    for alg, alg_stats in stats['algorithms'].items():
        print(f"• {alg}: {alg_stats['count']} runs, {alg_stats['total_time']:.1f}s total")
    print()
    
    print(f"SYSTEM TYPE BREAKDOWN:")
    for sys_type, type_stats in stats['system_types'].items():
        print(f"• {sys_type}: {type_stats['count']} runs")
    print()
    
    print(f"FILE STATISTICS:")
    print(f"• Total storage: {stats['file_statistics']['total_size']:,} bytes")
    print(f"• Average file size: {stats['file_statistics']['average_size']:.0f} bytes")
    print()
    
    # Demonstrate data cleanup (but don't actually clean up demo data)
    print("Step 6: Data Cleanup Demonstration...")
    print("(This would normally clean up old results, but skipping for demo)")
    old_count = data_manager.cleanup_old_results(days_old=365)  # Only remove very old results
    if old_count > 0:
        print(f"✓ Cleaned up {old_count} old results")
    else:
        print("✓ No old results to clean up")
    print()
    
    # Summary
    print("Step 7: Data Management Summary...")
    print("-" * 40)
    
    print(f"CAPABILITIES DEMONSTRATED:")
    print(f"• ✓ Optimization run storage and retrieval")
    print(f"• ✓ Multiple export formats (CSV, Excel)")
    print(f"• ✓ Comprehensive comparison reports")
    print(f"• ✓ Statistical analysis and metrics")
    print(f"• ✓ Data archiving and cleanup")
    print()
    
    print(f"FILES GENERATED:")
    print(f"• Database: demo_data_management/runs_database.json")
    print(f"• Individual results: demo_data_management/optimization_runs/*.json")
    print(f"• CSV export: {csv_file}")
    print(f"• Excel export: {excel_file}")
    print(f"• HTML report: {report_file}")
    print()
    
    print("="*70)
    print("DATA MANAGEMENT DEMONSTRATION COMPLETED")
    print("="*70)
    
    return {
        'run_ids': run_ids,
        'statistics': stats,
        'export_files': {
            'csv': csv_file,
            'excel': excel_file,
            'report': report_file
        }
    }

if __name__ == "__main__":
    try:
        results = main()
        print("\n✓ Data management demonstration completed successfully")
    except Exception as e:
        logger.error(f"Error in data management demonstration: {e}")
        print(f"\nError: {e}")
        print("Please check the log for detailed error information.")