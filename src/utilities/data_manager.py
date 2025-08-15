import json
import csv
import pickle
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import datetime
import logging
from dataclasses import asdict
import yaml


class DataManager:
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.results_path = self.base_path / "results"
        self.exports_path = self.base_path / "exports"
        self.cache_path = self.base_path / "cache"
        self.metadata_path = self.base_path / "metadata"
        
        for path in [self.results_path, self.exports_path, self.cache_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.DataManager")
        
        self.current_session_id = self._generate_session_id()
        self.optimization_runs = {}
        
    def _generate_session_id(self) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    def save_optimization_run(self, run_id: str, optimization_result: Any, 
                            algorithm_name: str, system_type: str,
                            parameters: Dict[str, Any]) -> str:
        
        run_data = {
            'run_id': run_id,
            'session_id': self.current_session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'algorithm_name': algorithm_name,
            'system_type': system_type,
            'parameters': parameters,
            'optimization_result': self._serialize_optimization_result(optimization_result)
        }
        
        filename = f"{run_id}_{algorithm_name}_{system_type}.json"
        filepath = self.results_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(run_data, f, indent=2, default=self._json_serializer)
        
        self.optimization_runs[run_id] = {
            'filepath': filepath,
            'metadata': {
                'algorithm': algorithm_name,
                'system_type': system_type,
                'timestamp': run_data['timestamp'],
                'parameters': parameters
            }
        }
        
        self.logger.info(f"Saved optimization run {run_id} to {filepath}")
        return str(filepath)
    
    def load_optimization_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        if run_id not in self.optimization_runs:
            self._scan_existing_results()
        
        if run_id in self.optimization_runs:
            filepath = self.optimization_runs[run_id]['filepath']
            with open(filepath, 'r') as f:
                return json.load(f)
        
        self.logger.warning(f"Optimization run {run_id} not found")
        return None
    
    def _scan_existing_results(self):
        for filepath in self.results_path.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    run_id = data.get('run_id')
                    if run_id:
                        self.optimization_runs[run_id] = {
                            'filepath': filepath,
                            'metadata': {
                                'algorithm': data.get('algorithm_name'),
                                'system_type': data.get('system_type'),
                                'timestamp': data.get('timestamp'),
                                'parameters': data.get('parameters', {})
                            }
                        }
            except Exception as e:
                self.logger.warning(f"Could not load {filepath}: {e}")
    
    def _serialize_optimization_result(self, result: Any) -> Dict[str, Any]:
        if hasattr(result, '__dict__'):
            return self._dict_serializer(result.__dict__)
        elif hasattr(result, '_asdict'):
            return self._dict_serializer(result._asdict())
        else:
            return self._dict_serializer(result)
    
    def _dict_serializer(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._dict_serializer(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._dict_serializer(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return self._dict_serializer(obj.__dict__)
        else:
            return obj
    
    def _json_serializer(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def export_to_csv(self, run_ids: List[str], filename: str) -> str:
        export_data = []
        
        for run_id in run_ids:
            run_data = self.load_optimization_run(run_id)
            if run_data:
                flattened_data = self._flatten_optimization_data(run_data)
                export_data.append(flattened_data)
        
        if not export_data:
            self.logger.warning("No data to export")
            return ""
        
        filepath = self.exports_path / f"{filename}.csv"
        
        df = pd.DataFrame(export_data)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Exported {len(export_data)} runs to {filepath}")
        return str(filepath)
    
    def export_to_excel(self, run_ids: List[str], filename: str) -> str:
        filepath = self.exports_path / f"{filename}.xlsx"
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            summary_data = []
            detailed_data = []
            
            for run_id in run_ids:
                run_data = self.load_optimization_run(run_id)
                if run_data:
                    summary_data.append(self._extract_summary_data(run_data))
                    detailed_data.extend(self._extract_detailed_data(run_data))
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            
            self._create_statistics_sheet(writer, summary_data)
        
        self.logger.info(f"Exported {len(run_ids)} runs to {filepath}")
        return str(filepath)
    
    def export_to_hdf5(self, run_ids: List[str], filename: str) -> str:
        filepath = self.exports_path / f"{filename}.h5"
        
        with h5py.File(filepath, 'w') as f:
            for run_id in run_ids:
                run_data = self.load_optimization_run(run_id)
                if run_data:
                    group = f.create_group(run_id)
                    self._write_hdf5_group(group, run_data)
        
        self.logger.info(f"Exported {len(run_ids)} runs to {filepath}")
        return str(filepath)
    
    def _flatten_optimization_data(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        flattened = {
            'run_id': run_data.get('run_id'),
            'session_id': run_data.get('session_id'),
            'timestamp': run_data.get('timestamp'),
            'algorithm_name': run_data.get('algorithm_name'),
            'system_type': run_data.get('system_type')
        }
        
        parameters = run_data.get('parameters', {})
        for param_name, param_value in parameters.items():
            flattened[f'param_{param_name}'] = param_value
        
        opt_result = run_data.get('optimization_result', {})
        
        best_objectives = opt_result.get('best_objectives', {})
        for obj_name, obj_value in best_objectives.items():
            flattened[f'best_{obj_name}'] = obj_value
        
        best_parameters = opt_result.get('best_parameters', {})
        for param_name, param_value in best_parameters.items():
            flattened[f'optimal_{param_name}'] = param_value
        
        flattened['total_evaluations'] = opt_result.get('total_evaluations', 0)
        flattened['convergence_achieved'] = opt_result.get('convergence_achieved', False)
        flattened['total_time'] = opt_result.get('total_time', 0.0)
        
        return flattened
    
    def _extract_summary_data(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        opt_result = run_data.get('optimization_result', {})
        
        summary = {
            'run_id': run_data.get('run_id'),
            'algorithm': run_data.get('algorithm_name'),
            'system_type': run_data.get('system_type'),
            'timestamp': run_data.get('timestamp'),
            'total_evaluations': opt_result.get('total_evaluations', 0),
            'total_time': opt_result.get('total_time', 0.0),
            'convergence_achieved': opt_result.get('convergence_achieved', False)
        }
        
        best_objectives = opt_result.get('best_objectives', {})
        for obj_name, obj_value in best_objectives.items():
            summary[f'best_{obj_name}'] = obj_value
        
        return summary
    
    def _extract_detailed_data(self, run_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        detailed_data = []
        opt_result = run_data.get('optimization_result', {})
        optimization_history = opt_result.get('optimization_history', [])
        
        run_id = run_data.get('run_id')
        algorithm = run_data.get('algorithm_name')
        
        for entry in optimization_history:
            detailed_entry = {
                'run_id': run_id,
                'algorithm': algorithm,
                'evaluation': entry.get('evaluation', 0)
            }
            
            if 'objectives' in entry:
                for obj_name, obj_value in entry['objectives'].items():
                    detailed_entry[f'obj_{obj_name}'] = obj_value
            
            if 'parameters' in entry:
                for param_name, param_value in entry['parameters'].items():
                    detailed_entry[f'param_{param_name}'] = param_value
            
            if 'fitness' in entry:
                detailed_entry['fitness'] = entry['fitness']
            
            detailed_data.append(detailed_entry)
        
        return detailed_data
    
    def _create_statistics_sheet(self, writer, summary_data: List[Dict[str, Any]]):
        if not summary_data:
            return
        
        df = pd.DataFrame(summary_data)
        
        stats_data = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['run_id']:
                stats_data.append({
                    'metric': col,
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    def _write_hdf5_group(self, group, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._write_hdf5_group(subgroup, value)
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    subgroup = group.create_group(key)
                    for i, item in enumerate(value):
                        item_group = subgroup.create_group(f'item_{i}')
                        self._write_hdf5_group(item_group, item)
                else:
                    try:
                        group.create_dataset(key, data=value)
                    except (TypeError, ValueError):
                        group.attrs[key] = str(value)
            else:
                try:
                    if isinstance(value, str):
                        group.attrs[key] = value
                    else:
                        group.create_dataset(key, data=value)
                except (TypeError, ValueError):
                    group.attrs[key] = str(value)
    
    def create_comparison_report(self, run_ids: List[str], report_name: str) -> str:
        report_data = {
            'report_name': report_name,
            'created_at': datetime.datetime.now().isoformat(),
            'runs_analyzed': len(run_ids),
            'comparisons': []
        }
        
        runs_data = []
        for run_id in run_ids:
            run_data = self.load_optimization_run(run_id)
            if run_data:
                runs_data.append(run_data)
        
        if len(runs_data) < 2:
            self.logger.warning("Need at least 2 runs for comparison")
            return ""
        
        algorithms = list(set(run['algorithm_name'] for run in runs_data))
        system_types = list(set(run['system_type'] for run in runs_data))
        
        report_data['algorithms_compared'] = algorithms
        report_data['system_types_compared'] = system_types
        
        performance_comparison = self._compare_performance(runs_data)
        report_data['performance_comparison'] = performance_comparison
        
        efficiency_comparison = self._compare_efficiency(runs_data)
        report_data['efficiency_comparison'] = efficiency_comparison
        
        convergence_comparison = self._compare_convergence(runs_data)
        report_data['convergence_comparison'] = convergence_comparison
        
        filepath = self.exports_path / f"{report_name}_comparison_report.json"
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=self._json_serializer)
        
        self.logger.info(f"Created comparison report: {filepath}")
        return str(filepath)
    
    def _compare_performance(self, runs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        performance_data = {}
        
        for run in runs_data:
            algorithm = run['algorithm_name']
            opt_result = run.get('optimization_result', {})
            best_objectives = opt_result.get('best_objectives', {})
            
            if algorithm not in performance_data:
                performance_data[algorithm] = {'objectives': {}}
            
            for obj_name, obj_value in best_objectives.items():
                if obj_name not in performance_data[algorithm]['objectives']:
                    performance_data[algorithm]['objectives'][obj_name] = []
                performance_data[algorithm]['objectives'][obj_name].append(obj_value)
        
        comparison = {}
        for algorithm, data in performance_data.items():
            comparison[algorithm] = {}
            for obj_name, values in data['objectives'].items():
                if values:
                    comparison[algorithm][obj_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'best': max(values),
                        'worst': min(values)
                    }
        
        return comparison
    
    def _compare_efficiency(self, runs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        efficiency_data = {}
        
        for run in runs_data:
            algorithm = run['algorithm_name']
            opt_result = run.get('optimization_result', {})
            
            total_time = opt_result.get('total_time', 0)
            total_evaluations = opt_result.get('total_evaluations', 1)
            
            if algorithm not in efficiency_data:
                efficiency_data[algorithm] = {
                    'times': [],
                    'evaluations': [],
                    'time_per_evaluation': []
                }
            
            efficiency_data[algorithm]['times'].append(total_time)
            efficiency_data[algorithm]['evaluations'].append(total_evaluations)
            efficiency_data[algorithm]['time_per_evaluation'].append(total_time / total_evaluations)
        
        comparison = {}
        for algorithm, data in efficiency_data.items():
            comparison[algorithm] = {
                'avg_total_time': np.mean(data['times']),
                'avg_evaluations': np.mean(data['evaluations']),
                'avg_time_per_eval': np.mean(data['time_per_evaluation'])
            }
        
        return comparison
    
    def _compare_convergence(self, runs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        convergence_data = {}
        
        for run in runs_data:
            algorithm = run['algorithm_name']
            opt_result = run.get('optimization_result', {})
            
            convergence_achieved = opt_result.get('convergence_achieved', False)
            
            if algorithm not in convergence_data:
                convergence_data[algorithm] = {'convergence_rates': []}
            
            convergence_data[algorithm]['convergence_rates'].append(convergence_achieved)
        
        comparison = {}
        for algorithm, data in convergence_data.items():
            rates = data['convergence_rates']
            comparison[algorithm] = {
                'convergence_rate': sum(rates) / len(rates) if rates else 0.0,
                'total_runs': len(rates)
            }
        
        return comparison
    
    def get_run_statistics(self) -> Dict[str, Any]:
        self._scan_existing_results()
        
        algorithms = {}
        system_types = {}
        total_runs = len(self.optimization_runs)
        
        for run_id, run_info in self.optimization_runs.items():
            metadata = run_info['metadata']
            
            algorithm = metadata.get('algorithm', 'unknown')
            system_type = metadata.get('system_type', 'unknown')
            
            algorithms[algorithm] = algorithms.get(algorithm, 0) + 1
            system_types[system_type] = system_types.get(system_type, 0) + 1
        
        return {
            'total_runs': total_runs,
            'algorithms': algorithms,
            'system_types': system_types,
            'current_session': self.current_session_id
        }
    
    def cleanup_old_results(self, days_old: int = 30):
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
        
        cleaned_count = 0
        for filepath in self.results_path.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    timestamp_str = data.get('timestamp')
                    if timestamp_str:
                        timestamp = datetime.datetime.fromisoformat(timestamp_str)
                        if timestamp < cutoff_date:
                            filepath.unlink()
                            cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"Could not process {filepath}: {e}")
        
        self.logger.info(f"Cleaned up {cleaned_count} old result files")
        self._scan_existing_results()


class ResultsAnalyzer:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.logger = logging.getLogger(f"{__name__}.ResultsAnalyzer")
    
    def analyze_algorithm_performance(self, algorithm_name: str) -> Dict[str, Any]:
        runs = []
        for run_id, run_info in self.data_manager.optimization_runs.items():
            if run_info['metadata']['algorithm'] == algorithm_name:
                run_data = self.data_manager.load_optimization_run(run_id)
                if run_data:
                    runs.append(run_data)
        
        if not runs:
            return {'error': f'No runs found for algorithm {algorithm_name}'}
        
        analysis = {
            'algorithm': algorithm_name,
            'total_runs': len(runs),
            'performance_metrics': self._analyze_performance_metrics(runs),
            'convergence_analysis': self._analyze_convergence(runs),
            'efficiency_analysis': self._analyze_efficiency(runs)
        }
        
        return analysis
    
    def _analyze_performance_metrics(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_objectives = {}
        
        for run in runs:
            opt_result = run.get('optimization_result', {})
            best_objectives = opt_result.get('best_objectives', {})
            
            for obj_name, obj_value in best_objectives.items():
                if obj_name not in all_objectives:
                    all_objectives[obj_name] = []
                all_objectives[obj_name].append(obj_value)
        
        metrics = {}
        for obj_name, values in all_objectives.items():
            if values:
                metrics[obj_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return metrics
    
    def _analyze_convergence(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        convergence_rates = []
        convergence_speeds = []
        
        for run in runs:
            opt_result = run.get('optimization_result', {})
            convergence_achieved = opt_result.get('convergence_achieved', False)
            convergence_rates.append(convergence_achieved)
            
            total_evaluations = opt_result.get('total_evaluations', 0)
            if convergence_achieved and total_evaluations > 0:
                convergence_speeds.append(total_evaluations)
        
        analysis = {
            'convergence_rate': np.mean(convergence_rates),
            'avg_convergence_speed': np.mean(convergence_speeds) if convergence_speeds else None,
            'std_convergence_speed': np.std(convergence_speeds) if convergence_speeds else None
        }
        
        return analysis
    
    def _analyze_efficiency(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        computation_times = []
        evaluations = []
        
        for run in runs:
            opt_result = run.get('optimization_result', {})
            total_time = opt_result.get('total_time', 0)
            total_evaluations = opt_result.get('total_evaluations', 0)
            
            computation_times.append(total_time)
            evaluations.append(total_evaluations)
        
        time_per_eval = [t/e for t, e in zip(computation_times, evaluations) if e > 0]
        
        analysis = {
            'avg_computation_time': np.mean(computation_times),
            'avg_evaluations': np.mean(evaluations),
            'avg_time_per_evaluation': np.mean(time_per_eval) if time_per_eval else None
        }
        
        return analysis