import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
warnings.filterwarnings('ignore')


@dataclass
class UncertaintyDistribution:
    distribution_type: str
    parameters: Dict[str, float]
    bounds: Optional[Tuple[float, float]] = None


@dataclass
class RobustOptimizationResult:
    robust_parameters: Dict[str, float]
    robust_objectives: Dict[str, float]
    uncertainty_analysis: Dict[str, Any]
    monte_carlo_results: List[Dict[str, Any]]
    sensitivity_analysis: Dict[str, Dict[str, float]]
    reliability_metrics: Dict[str, float]
    total_evaluations: int
    computation_time: float


class UncertaintyQuantification:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.UncertaintyQuantification")
        self.parameter_uncertainties = {}
        self.environmental_uncertainties = {}
        self.model_uncertainties = {}
        
    def add_parameter_uncertainty(self, parameter_name: str, uncertainty: UncertaintyDistribution):
        self.parameter_uncertainties[parameter_name] = uncertainty
        
    def add_environmental_uncertainty(self, env_name: str, uncertainty: UncertaintyDistribution):
        self.environmental_uncertainties[env_name] = uncertainty
        
    def add_model_uncertainty(self, model_name: str, uncertainty: UncertaintyDistribution):
        self.model_uncertainties[model_name] = uncertainty
    
    def sample_uncertainties(self, n_samples: int) -> List[Dict[str, Dict[str, float]]]:
        samples = []
        
        for _ in range(n_samples):
            sample = {
                'parameters': {},
                'environment': {},
                'model': {}
            }
            
            for param_name, uncertainty in self.parameter_uncertainties.items():
                sample['parameters'][param_name] = self._sample_distribution(uncertainty)
            
            for env_name, uncertainty in self.environmental_uncertainties.items():
                sample['environment'][env_name] = self._sample_distribution(uncertainty)
            
            for model_name, uncertainty in self.model_uncertainties.items():
                sample['model'][model_name] = self._sample_distribution(uncertainty)
            
            samples.append(sample)
        
        return samples
    
    def _sample_distribution(self, uncertainty: UncertaintyDistribution) -> float:
        if uncertainty.distribution_type == 'normal':
            mean = uncertainty.parameters.get('mean', 0.0)
            std = uncertainty.parameters.get('std', 1.0)
            sample = np.random.normal(mean, std)
        elif uncertainty.distribution_type == 'uniform':
            low = uncertainty.parameters.get('low', 0.0)
            high = uncertainty.parameters.get('high', 1.0)
            sample = np.random.uniform(low, high)
        elif uncertainty.distribution_type == 'lognormal':
            mean = uncertainty.parameters.get('mean', 0.0)
            sigma = uncertainty.parameters.get('sigma', 1.0)
            sample = np.random.lognormal(mean, sigma)
        elif uncertainty.distribution_type == 'beta':
            alpha = uncertainty.parameters.get('alpha', 1.0)
            beta = uncertainty.parameters.get('beta', 1.0)
            sample = np.random.beta(alpha, beta)
        elif uncertainty.distribution_type == 'triangular':
            left = uncertainty.parameters.get('left', 0.0)
            mode = uncertainty.parameters.get('mode', 0.5)
            right = uncertainty.parameters.get('right', 1.0)
            sample = np.random.triangular(left, mode, right)
        else:
            sample = 0.0
        
        if uncertainty.bounds:
            min_val, max_val = uncertainty.bounds
            sample = max(min_val, min(max_val, sample))
        
        return sample
    
    def compute_statistics(self, samples: List[float]) -> Dict[str, float]:
        if not samples:
            return {}
        
        samples_array = np.array(samples)
        
        return {
            'mean': np.mean(samples_array),
            'std': np.std(samples_array),
            'var': np.var(samples_array),
            'min': np.min(samples_array),
            'max': np.max(samples_array),
            'median': np.median(samples_array),
            'q25': np.percentile(samples_array, 25),
            'q75': np.percentile(samples_array, 75),
            'skewness': stats.skew(samples_array),
            'kurtosis': stats.kurtosis(samples_array)
        }
    
    def compute_confidence_intervals(self, samples: List[float], 
                                   confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Tuple[float, float]]:
        if not samples:
            return {}
        
        samples_array = np.array(samples)
        intervals = {}
        
        for level in confidence_levels:
            alpha = 1 - level
            lower = np.percentile(samples_array, 100 * alpha / 2)
            upper = np.percentile(samples_array, 100 * (1 - alpha / 2))
            intervals[f'{level:.2f}'] = (lower, upper)
        
        return intervals


class SensitivityAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SensitivityAnalysis")
    
    def morris_screening(self, objective_function: Callable, 
                        parameter_bounds: Dict[str, Tuple[float, float]],
                        n_trajectories: int = 10, n_levels: int = 4) -> Dict[str, Dict[str, float]]:
        
        parameters = list(parameter_bounds.keys())
        n_params = len(parameters)
        
        delta = n_levels / (2 * (n_levels - 1))
        
        sensitivity_results = {}
        
        for param in parameters:
            elementary_effects = []
            
            for _ in range(n_trajectories):
                base_point = self._generate_random_point(parameter_bounds)
                
                perturbed_point = base_point.copy()
                min_val, max_val = parameter_bounds[param]
                perturbation = delta * (max_val - min_val)
                perturbed_point[param] = min(max_val, base_point[param] + perturbation)
                
                try:
                    base_result = objective_function(base_point)
                    perturbed_result = objective_function(perturbed_point)
                    
                    base_fitness = sum(base_result.objectives.values()) / len(base_result.objectives)
                    perturbed_fitness = sum(perturbed_result.objectives.values()) / len(perturbed_result.objectives)
                    
                    elementary_effect = (perturbed_fitness - base_fitness) / perturbation
                    elementary_effects.append(elementary_effect)
                    
                except Exception as e:
                    self.logger.warning(f"Error in Morris screening for {param}: {e}")
                    continue
            
            if elementary_effects:
                sensitivity_results[param] = {
                    'mean_elementary_effect': np.mean(elementary_effects),
                    'std_elementary_effect': np.std(elementary_effects),
                    'mean_absolute_effect': np.mean(np.abs(elementary_effects))
                }
            else:
                sensitivity_results[param] = {
                    'mean_elementary_effect': 0.0,
                    'std_elementary_effect': 0.0,
                    'mean_absolute_effect': 0.0
                }
        
        return sensitivity_results
    
    def sobol_indices(self, objective_function: Callable,
                     parameter_bounds: Dict[str, Tuple[float, float]],
                     n_samples: int = 1000) -> Dict[str, Dict[str, float]]:
        
        parameters = list(parameter_bounds.keys())
        n_params = len(parameters)
        
        sobol_results = {}
        
        for i, param in enumerate(parameters):
            first_order_samples = []
            total_order_samples = []
            
            for _ in range(n_samples):
                base_point = self._generate_random_point(parameter_bounds)
                resampled_point = self._generate_random_point(parameter_bounds)
                
                varied_point = base_point.copy()
                varied_point[param] = resampled_point[param]
                
                try:
                    base_result = objective_function(base_point)
                    varied_result = objective_function(varied_point)
                    resampled_result = objective_function(resampled_point)
                    
                    base_fitness = sum(base_result.objectives.values()) / len(base_result.objectives)
                    varied_fitness = sum(varied_result.objectives.values()) / len(varied_result.objectives)
                    resampled_fitness = sum(resampled_result.objectives.values()) / len(resampled_result.objectives)
                    
                    first_order_samples.append((base_fitness, varied_fitness, resampled_fitness))
                    
                except Exception as e:
                    self.logger.warning(f"Error in Sobol analysis for {param}: {e}")
                    continue
            
            if first_order_samples:
                first_order_index = self._compute_sobol_first_order(first_order_samples)
                total_order_index = self._compute_sobol_total_order(first_order_samples)
                
                sobol_results[param] = {
                    'first_order_index': first_order_index,
                    'total_order_index': total_order_index
                }
            else:
                sobol_results[param] = {
                    'first_order_index': 0.0,
                    'total_order_index': 0.0
                }
        
        return sobol_results
    
    def _generate_random_point(self, parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        point = {}
        for param, (min_val, max_val) in parameter_bounds.items():
            point[param] = np.random.uniform(min_val, max_val)
        return point
    
    def _compute_sobol_first_order(self, samples: List[Tuple[float, float, float]]) -> float:
        if not samples:
            return 0.0
        
        f_a = np.array([s[0] for s in samples])
        f_ab = np.array([s[1] for s in samples])
        f_b = np.array([s[2] for s in samples])
        
        var_y = np.var(np.concatenate([f_a, f_b]))
        if var_y == 0:
            return 0.0
        
        first_order = np.mean(f_b * (f_ab - f_a)) / var_y
        return max(0.0, min(1.0, first_order))
    
    def _compute_sobol_total_order(self, samples: List[Tuple[float, float, float]]) -> float:
        if not samples:
            return 0.0
        
        f_a = np.array([s[0] for s in samples])
        f_ab = np.array([s[1] for s in samples])
        f_b = np.array([s[2] for s in samples])
        
        var_y = np.var(np.concatenate([f_a, f_b]))
        if var_y == 0:
            return 0.0
        
        total_order = 1 - np.mean(f_a * (f_ab - f_b)) / var_y
        return max(0.0, min(1.0, total_order))


class RobustOptimizer:
    def __init__(self, uncertainty_quantification: UncertaintyQuantification):
        self.uq = uncertainty_quantification
        self.logger = logging.getLogger(f"{__name__}.RobustOptimizer")
        self.sensitivity_analyzer = SensitivityAnalysis()
        
    def robust_optimization(self, objective_function: Callable,
                          parameter_bounds: Dict[str, Tuple[float, float]],
                          robustness_measure: str = 'mean_std',
                          n_mc_samples: int = 100,
                          optimization_algorithm: str = 'nelder_mead') -> RobustOptimizationResult:
        
        import time
        start_time = time.time()
        
        def robust_objective(params_array):
            params_dict = self._array_to_dict(params_array, parameter_bounds)
            return self._evaluate_robustness(params_dict, objective_function, 
                                           robustness_measure, n_mc_samples)
        
        bounds = [(min_val, max_val) for min_val, max_val in parameter_bounds.values()]
        x0 = np.array([np.mean([min_val, max_val]) for min_val, max_val in bounds])
        
        result = minimize(robust_objective, x0, bounds=bounds, method='L-BFGS-B')
        
        optimal_params = self._array_to_dict(result.x, parameter_bounds)
        
        mc_results = self._monte_carlo_analysis(optimal_params, objective_function, n_mc_samples * 2)
        
        uncertainty_analysis = self._analyze_uncertainties(mc_results)
        
        sensitivity_results = self.sensitivity_analyzer.morris_screening(
            objective_function, parameter_bounds, n_trajectories=20
        )
        
        reliability_metrics = self._compute_reliability_metrics(mc_results)
        
        robust_objectives = {}
        if mc_results:
            all_objectives = {}
            for result in mc_results:
                for obj_name, obj_value in result['objectives'].items():
                    if obj_name not in all_objectives:
                        all_objectives[obj_name] = []
                    all_objectives[obj_name].append(obj_value)
            
            for obj_name, values in all_objectives.items():
                robust_objectives[obj_name] = np.mean(values)
        
        end_time = time.time()
        
        return RobustOptimizationResult(
            robust_parameters=optimal_params,
            robust_objectives=robust_objectives,
            uncertainty_analysis=uncertainty_analysis,
            monte_carlo_results=mc_results,
            sensitivity_analysis=sensitivity_results,
            reliability_metrics=reliability_metrics,
            total_evaluations=len(mc_results),
            computation_time=end_time - start_time
        )
    
    def _array_to_dict(self, params_array: np.ndarray, 
                      parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        params_dict = {}
        for i, param_name in enumerate(parameter_bounds.keys()):
            params_dict[param_name] = params_array[i]
        return params_dict
    
    def _evaluate_robustness(self, nominal_params: Dict[str, float],
                           objective_function: Callable,
                           robustness_measure: str,
                           n_samples: int) -> float:
        
        mc_results = self._monte_carlo_analysis(nominal_params, objective_function, n_samples)
        
        if not mc_results:
            return float('inf')
        
        objective_values = []
        for result in mc_results:
            obj_sum = sum(result['objectives'].values())
            objective_values.append(obj_sum)
        
        if robustness_measure == 'mean_std':
            mean_obj = np.mean(objective_values)
            std_obj = np.std(objective_values)
            return -(mean_obj - 2 * std_obj)
        
        elif robustness_measure == 'worst_case':
            return -np.min(objective_values)
        
        elif robustness_measure == 'mean':
            return -np.mean(objective_values)
        
        elif robustness_measure == 'cvar':
            alpha = 0.05
            sorted_values = np.sort(objective_values)
            n_cvar = max(1, int(alpha * len(sorted_values)))
            cvar = np.mean(sorted_values[:n_cvar])
            return -cvar
        
        else:
            return -np.mean(objective_values)
    
    def _monte_carlo_analysis(self, nominal_params: Dict[str, float],
                            objective_function: Callable,
                            n_samples: int) -> List[Dict[str, Any]]:
        
        uncertainty_samples = self.uq.sample_uncertainties(n_samples)
        
        mc_results = []
        
        for i, uncertainties in enumerate(uncertainty_samples):
            perturbed_params = nominal_params.copy()
            
            for param_name, uncertainty_value in uncertainties['parameters'].items():
                if param_name in perturbed_params:
                    perturbed_params[param_name] += uncertainty_value
            
            try:
                result = objective_function(perturbed_params)
                
                mc_results.append({
                    'sample_id': i,
                    'parameters': perturbed_params.copy(),
                    'objectives': result.objectives.copy(),
                    'uncertainties': uncertainties,
                    'constraints': getattr(result, 'constraints', {}),
                    'metadata': getattr(result, 'metadata', {})
                })
                
            except Exception as e:
                self.logger.warning(f"Monte Carlo sample {i} failed: {e}")
                continue
        
        return mc_results
    
    def _analyze_uncertainties(self, mc_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not mc_results:
            return {}
        
        analysis = {}
        
        objective_names = list(mc_results[0]['objectives'].keys())
        
        for obj_name in objective_names:
            obj_values = [result['objectives'][obj_name] for result in mc_results]
            
            statistics = self.uq.compute_statistics(obj_values)
            confidence_intervals = self.uq.compute_confidence_intervals(obj_values)
            
            analysis[obj_name] = {
                'statistics': statistics,
                'confidence_intervals': confidence_intervals
            }
        
        return analysis
    
    def _compute_reliability_metrics(self, mc_results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not mc_results:
            return {}
        
        n_samples = len(mc_results)
        
        constraint_violations = 0
        objective_failures = 0
        
        for result in mc_results:
            if result['constraints']:
                for constraint_name, constraint_value in result['constraints'].items():
                    if constraint_value < 0:
                        constraint_violations += 1
                        break
            
            objectives = list(result['objectives'].values())
            if any(obj < 0 for obj in objectives):
                objective_failures += 1
        
        reliability = 1 - (constraint_violations / n_samples)
        success_rate = 1 - (objective_failures / n_samples)
        
        objective_values = []
        for result in mc_results:
            obj_sum = sum(result['objectives'].values())
            objective_values.append(obj_sum)
        
        robustness_index = 1 - (np.std(objective_values) / (np.mean(objective_values) + 1e-10))
        
        return {
            'reliability': reliability,
            'success_rate': success_rate,
            'robustness_index': max(0.0, robustness_index),
            'constraint_violation_rate': constraint_violations / n_samples,
            'objective_failure_rate': objective_failures / n_samples
        }


class UncertaintyPropagation:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.UncertaintyPropagation")
    
    def polynomial_chaos_expansion(self, objective_function: Callable,
                                 parameter_bounds: Dict[str, Tuple[float, float]],
                                 polynomial_order: int = 2,
                                 n_samples: int = 100) -> Dict[str, Any]:
        
        parameters = list(parameter_bounds.keys())
        n_params = len(parameters)
        
        sample_points = []
        objective_evaluations = []
        
        for _ in range(n_samples):
            point = {}
            for param, (min_val, max_val) in parameter_bounds.items():
                point[param] = np.random.uniform(min_val, max_val)
            
            try:
                result = objective_function(point)
                obj_value = sum(result.objectives.values()) / len(result.objectives)
                
                sample_points.append([point[param] for param in parameters])
                objective_evaluations.append(obj_value)
                
            except Exception as e:
                self.logger.warning(f"PCE evaluation failed: {e}")
                continue
        
        if not sample_points:
            return {}
        
        X = np.array(sample_points)
        y = np.array(objective_evaluations)
        
        polynomial_features = self._generate_polynomial_features(X, polynomial_order)
        
        try:
            coefficients = np.linalg.lstsq(polynomial_features, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            self.logger.warning("PCE coefficient computation failed")
            return {}
        
        main_effects = self._compute_main_effects(coefficients, polynomial_features)
        interaction_effects = self._compute_interaction_effects(coefficients, polynomial_features)
        
        return {
            'coefficients': coefficients.tolist(),
            'main_effects': main_effects,
            'interaction_effects': interaction_effects,
            'polynomial_order': polynomial_order,
            'n_samples': len(sample_points)
        }
    
    def _generate_polynomial_features(self, X: np.ndarray, order: int) -> np.ndarray:
        n_samples, n_features = X.shape
        
        features = [np.ones(n_samples)]
        
        for i in range(n_features):
            for p in range(1, order + 1):
                features.append(X[:, i] ** p)
        
        if order >= 2:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    features.append(X[:, i] * X[:, j])
        
        return np.column_stack(features)
    
    def _compute_main_effects(self, coefficients: np.ndarray, 
                            polynomial_features: np.ndarray) -> Dict[str, float]:
        main_effects = {}
        
        for i in range(len(coefficients)):
            if i > 0:
                main_effects[f'feature_{i}'] = abs(coefficients[i])
        
        return main_effects
    
    def _compute_interaction_effects(self, coefficients: np.ndarray,
                                   polynomial_features: np.ndarray) -> Dict[str, float]:
        interaction_effects = {}
        
        n_linear = polynomial_features.shape[1] - 1
        for i in range(n_linear, len(coefficients)):
            interaction_effects[f'interaction_{i}'] = abs(coefficients[i])
        
        return interaction_effects