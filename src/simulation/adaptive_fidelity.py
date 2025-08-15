import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
from ..simulation.base import FidelityLevel, MultiFidelitySimulation, SimulationResult


class FidelitySwitchingStrategy(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


class FidelityDecisionCriteria:
    def __init__(self):
        self.convergence_window = 10
        self.uncertainty_threshold = 0.08
        self.computational_budget_fraction = 0.7
        self.min_low_fidelity_evaluations = 20
        self.max_consecutive_high_fidelity = 5
        self.improvement_threshold = 0.01
        self.confidence_threshold = 0.85


class AdaptiveFidelityManager:
    def __init__(self, strategy: FidelitySwitchingStrategy = FidelitySwitchingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.criteria = FidelityDecisionCriteria()
        self.logger = logging.getLogger(f"{__name__}.AdaptiveFidelityManager")
        
        self.evaluation_history = []
        self.fidelity_performance = {
            FidelityLevel.LOW: {'accuracy': 0.8, 'cost': 1.0, 'reliability': 0.9},
            FidelityLevel.MEDIUM: {'accuracy': 0.9, 'cost': 5.0, 'reliability': 0.95},
            FidelityLevel.HIGH: {'accuracy': 0.98, 'cost': 15.0, 'reliability': 0.99}
        }
        
        self.computational_budget = 3600.0
        self.used_budget = 0.0
        self.consecutive_high_fidelity = 0
        self.last_improvement = float('inf')
        
        self.surrogate_model = None
        self.trust_region_radius = 1.0
        self.exploration_factor = 0.1
    
    def select_fidelity(self, parameters: Dict[str, float], 
                       optimization_progress: Optional[Dict[str, Any]] = None,
                       multi_fidelity_sim: Optional[MultiFidelitySimulation] = None) -> FidelityLevel:
        
        if self.strategy == FidelitySwitchingStrategy.CONSERVATIVE:
            return self._conservative_selection(parameters, optimization_progress)
        elif self.strategy == FidelitySwitchingStrategy.AGGRESSIVE:
            return self._aggressive_selection(parameters, optimization_progress)
        elif self.strategy == FidelitySwitchingStrategy.BALANCED:
            return self._balanced_selection(parameters, optimization_progress)
        else:
            return self._adaptive_selection(parameters, optimization_progress, multi_fidelity_sim)
    
    def _conservative_selection(self, parameters: Dict[str, float], 
                              optimization_progress: Optional[Dict[str, Any]]) -> FidelityLevel:
        
        budget_fraction = self.used_budget / self.computational_budget
        
        if budget_fraction > 0.8:
            return FidelityLevel.LOW
        
        if len(self.evaluation_history) < self.criteria.min_low_fidelity_evaluations:
            return FidelityLevel.LOW
        
        if self._check_convergence() and budget_fraction < 0.6:
            return FidelityLevel.HIGH
        
        return FidelityLevel.LOW
    
    def _aggressive_selection(self, parameters: Dict[str, float], 
                            optimization_progress: Optional[Dict[str, Any]]) -> FidelityLevel:
        
        if len(self.evaluation_history) < 5:
            return FidelityLevel.LOW
        
        budget_fraction = self.used_budget / self.computational_budget
        
        if budget_fraction > 0.9:
            return FidelityLevel.LOW
        
        if self._is_promising_region(parameters):
            return FidelityLevel.HIGH
        
        return FidelityLevel.MEDIUM if budget_fraction < 0.7 else FidelityLevel.LOW
    
    def _balanced_selection(self, parameters: Dict[str, float], 
                          optimization_progress: Optional[Dict[str, Any]]) -> FidelityLevel:
        
        budget_fraction = self.used_budget / self.computational_budget
        
        if budget_fraction > 0.85:
            return FidelityLevel.LOW
        
        if len(self.evaluation_history) < self.criteria.min_low_fidelity_evaluations:
            return FidelityLevel.LOW
        
        if self.consecutive_high_fidelity >= self.criteria.max_consecutive_high_fidelity:
            return FidelityLevel.LOW
        
        promising_score = self._compute_promising_score(parameters)
        
        if promising_score > 0.8 and budget_fraction < 0.7:
            return FidelityLevel.HIGH
        elif promising_score > 0.5 and budget_fraction < 0.8:
            return FidelityLevel.MEDIUM
        else:
            return FidelityLevel.LOW
    
    def _adaptive_selection(self, parameters: Dict[str, float], 
                          optimization_progress: Optional[Dict[str, Any]],
                          multi_fidelity_sim: Optional[MultiFidelitySimulation]) -> FidelityLevel:
        
        budget_fraction = self.used_budget / self.computational_budget
        
        if budget_fraction > 0.9:
            return FidelityLevel.LOW
        
        if len(self.evaluation_history) < 10:
            return FidelityLevel.LOW
        
        decision_score = self._compute_adaptive_decision_score(
            parameters, optimization_progress, multi_fidelity_sim
        )
        
        uncertainty_level = self._estimate_local_uncertainty(parameters)
        expected_improvement = self._estimate_expected_improvement(parameters)
        
        cost_benefit_ratio = self._compute_cost_benefit_ratio(
            decision_score, uncertainty_level, expected_improvement
        )
        
        if cost_benefit_ratio > 2.0 and budget_fraction < 0.8:
            return FidelityLevel.HIGH
        elif cost_benefit_ratio > 1.0 and budget_fraction < 0.85:
            return FidelityLevel.MEDIUM
        else:
            return FidelityLevel.LOW
    
    def _compute_adaptive_decision_score(self, parameters: Dict[str, float],
                                       optimization_progress: Optional[Dict[str, Any]],
                                       multi_fidelity_sim: Optional[MultiFidelitySimulation]) -> float:
        
        score = 0.0
        
        convergence_score = self._compute_convergence_score()
        score += convergence_score * 0.3
        
        exploration_score = self._compute_exploration_score(parameters)
        score += exploration_score * 0.2
        
        uncertainty_score = self._compute_uncertainty_score()
        score += uncertainty_score * 0.25
        
        improvement_score = self._compute_improvement_score()
        score += improvement_score * 0.25
        
        return min(1.0, max(0.0, score))
    
    def _compute_convergence_score(self) -> float:
        
        if len(self.evaluation_history) < self.criteria.convergence_window:
            return 0.0
        
        recent_results = self.evaluation_history[-self.criteria.convergence_window:]
        
        objective_values = []
        for entry in recent_results:
            if 'result' in entry and hasattr(entry['result'], 'objectives'):
                values = list(entry['result'].objectives.values())
                if values:
                    objective_values.append(np.mean(values))
        
        if len(objective_values) < 3:
            return 0.0
        
        variations = []
        for i in range(1, len(objective_values)):
            if objective_values[i-1] != 0:
                variation = abs(objective_values[i] - objective_values[i-1]) / abs(objective_values[i-1])
                variations.append(variation)
        
        if not variations:
            return 0.0
        
        avg_variation = np.mean(variations)
        convergence_score = max(0.0, 1.0 - avg_variation / self.criteria.improvement_threshold)
        
        return convergence_score
    
    def _compute_exploration_score(self, parameters: Dict[str, float]) -> float:
        
        if len(self.evaluation_history) < 5:
            return 1.0
        
        distances = []
        for entry in self.evaluation_history[-20:]:
            if 'parameters' in entry:
                distance = self._parameter_distance(parameters, entry['parameters'])
                distances.append(distance)
        
        if not distances:
            return 1.0
        
        min_distance = min(distances)
        
        exploration_score = min(1.0, min_distance / self.trust_region_radius)
        
        return exploration_score
    
    def _compute_uncertainty_score(self) -> float:
        
        if len(self.evaluation_history) < 5:
            return 0.5
        
        recent_uncertainties = []
        for entry in self.evaluation_history[-10:]:
            if 'result' in entry and hasattr(entry['result'], 'uncertainty'):
                uncertainty_values = list(entry['result'].uncertainty.values())
                if uncertainty_values:
                    recent_uncertainties.append(np.mean(uncertainty_values))
        
        if not recent_uncertainties:
            return 0.5
        
        avg_uncertainty = np.mean(recent_uncertainties)
        
        uncertainty_score = min(1.0, avg_uncertainty / self.criteria.uncertainty_threshold)
        
        return uncertainty_score
    
    def _compute_improvement_score(self) -> float:
        
        if len(self.evaluation_history) < 10:
            return 0.5
        
        recent_objectives = []
        for entry in self.evaluation_history[-10:]:
            if 'result' in entry and hasattr(entry['result'], 'objectives'):
                values = list(entry['result'].objectives.values())
                if values:
                    recent_objectives.append(np.mean(values))
        
        if len(recent_objectives) < 5:
            return 0.5
        
        improvements = []
        for i in range(1, len(recent_objectives)):
            if recent_objectives[i-1] != 0:
                improvement = (recent_objectives[i] - recent_objectives[i-1]) / abs(recent_objectives[i-1])
                improvements.append(improvement)
        
        if not improvements:
            return 0.5
        
        avg_improvement = np.mean(improvements)
        improvement_score = min(1.0, max(0.0, avg_improvement / self.criteria.improvement_threshold))
        
        return improvement_score
    
    def _parameter_distance(self, params1: Dict[str, float], params2: Dict[str, float]) -> float:
        
        common_params = set(params1.keys()) & set(params2.keys())
        
        if not common_params:
            return 1.0
        
        squared_diffs = []
        for param in common_params:
            diff = abs(params1[param] - params2[param])
            squared_diffs.append(diff ** 2)
        
        return math.sqrt(sum(squared_diffs) / len(squared_diffs))
    
    def _check_convergence(self) -> bool:
        
        if len(self.evaluation_history) < self.criteria.convergence_window:
            return False
        
        recent_results = self.evaluation_history[-self.criteria.convergence_window:]
        
        objective_trends = {}
        for entry in recent_results:
            if 'result' in entry and hasattr(entry['result'], 'objectives'):
                for obj_name, obj_value in entry['result'].objectives.items():
                    if obj_name not in objective_trends:
                        objective_trends[obj_name] = []
                    objective_trends[obj_name].append(obj_value)
        
        for obj_name, values in objective_trends.items():
            if len(values) < 5:
                continue
            
            recent_variation = np.std(values[-5:]) / (np.mean(values[-5:]) + 1e-10)
            if recent_variation > self.criteria.improvement_threshold:
                return False
        
        return True
    
    def _is_promising_region(self, parameters: Dict[str, float]) -> bool:
        
        if len(self.evaluation_history) < 10:
            return False
        
        nearby_results = []
        for entry in self.evaluation_history:
            if 'parameters' in entry and 'result' in entry:
                distance = self._parameter_distance(parameters, entry['parameters'])
                if distance < self.trust_region_radius:
                    nearby_results.append(entry['result'])
        
        if len(nearby_results) < 3:
            return False
        
        nearby_objectives = []
        for result in nearby_results:
            if hasattr(result, 'objectives'):
                values = list(result.objectives.values())
                if values:
                    nearby_objectives.append(np.mean(values))
        
        if len(nearby_objectives) < 3:
            return False
        
        all_objectives = []
        for entry in self.evaluation_history:
            if 'result' in entry and hasattr(entry['result'], 'objectives'):
                values = list(entry['result'].objectives.values())
                if values:
                    all_objectives.append(np.mean(values))
        
        if not all_objectives:
            return False
        
        nearby_mean = np.mean(nearby_objectives)
        overall_mean = np.mean(all_objectives)
        
        return nearby_mean > overall_mean * 1.1
    
    def _compute_promising_score(self, parameters: Dict[str, float]) -> float:
        
        if len(self.evaluation_history) < 5:
            return 0.5
        
        exploration_score = self._compute_exploration_score(parameters)
        
        if self._is_promising_region(parameters):
            return min(1.0, 0.7 + exploration_score * 0.3)
        else:
            return exploration_score * 0.6
    
    def _estimate_local_uncertainty(self, parameters: Dict[str, float]) -> float:
        
        nearby_uncertainties = []
        for entry in self.evaluation_history:
            if 'parameters' in entry and 'result' in entry:
                distance = self._parameter_distance(parameters, entry['parameters'])
                if distance < self.trust_region_radius * 2:
                    if hasattr(entry['result'], 'uncertainty'):
                        uncertainty_values = list(entry['result'].uncertainty.values())
                        if uncertainty_values:
                            nearby_uncertainties.append(np.mean(uncertainty_values))
        
        if not nearby_uncertainties:
            return 0.1
        
        return np.mean(nearby_uncertainties)
    
    def _estimate_expected_improvement(self, parameters: Dict[str, float]) -> float:
        
        if len(self.evaluation_history) < 10:
            return 0.5
        
        recent_improvements = []
        for i in range(1, min(10, len(self.evaluation_history))):
            entry1 = self.evaluation_history[-(i+1)]
            entry2 = self.evaluation_history[-i]
            
            if ('result' in entry1 and 'result' in entry2 and 
                hasattr(entry1['result'], 'objectives') and 
                hasattr(entry2['result'], 'objectives')):
                
                obj1 = list(entry1['result'].objectives.values())
                obj2 = list(entry2['result'].objectives.values())
                
                if obj1 and obj2:
                    improvement = (np.mean(obj2) - np.mean(obj1)) / (abs(np.mean(obj1)) + 1e-10)
                    recent_improvements.append(improvement)
        
        if not recent_improvements:
            return 0.5
        
        return max(0.0, min(1.0, np.mean(recent_improvements)))
    
    def _compute_cost_benefit_ratio(self, decision_score: float, 
                                  uncertainty_level: float, 
                                  expected_improvement: float) -> float:
        
        benefit = decision_score * expected_improvement * (1 + uncertainty_level)
        
        cost_ratio = self.fidelity_performance[FidelityLevel.HIGH]['cost'] / self.fidelity_performance[FidelityLevel.LOW]['cost']
        
        return benefit / cost_ratio if cost_ratio > 0 else 0.0
    
    def update_evaluation_history(self, parameters: Dict[str, float], 
                                result: SimulationResult, 
                                computation_time: float):
        
        self.evaluation_history.append({
            'parameters': parameters.copy(),
            'result': result,
            'computation_time': computation_time,
            'timestamp': len(self.evaluation_history)
        })
        
        self.used_budget += computation_time
        
        if result.fidelity_level == FidelityLevel.HIGH:
            self.consecutive_high_fidelity += 1
        else:
            self.consecutive_high_fidelity = 0
        
        self._update_fidelity_performance(result)
        self._adapt_criteria()
    
    def _update_fidelity_performance(self, result: SimulationResult):
        
        fidelity = result.fidelity_level
        
        if len(self.evaluation_history) > 1:
            
            current_objectives = list(result.objectives.values())
            if current_objectives:
                current_score = np.mean(current_objectives)
                
                self.fidelity_performance[fidelity]['accuracy'] = (
                    self.fidelity_performance[fidelity]['accuracy'] * 0.9 + 
                    min(1.0, current_score / 100.0) * 0.1
                )
    
    def _adapt_criteria(self):
        
        if len(self.evaluation_history) < 20:
            return
        
        recent_history = self.evaluation_history[-20:]
        
        high_fidelity_count = sum(1 for entry in recent_history 
                                if entry['result'].fidelity_level == FidelityLevel.HIGH)
        
        if high_fidelity_count > 15:
            self.criteria.uncertainty_threshold *= 1.1
            self.criteria.improvement_threshold *= 0.9
        elif high_fidelity_count < 5:
            self.criteria.uncertainty_threshold *= 0.9
            self.criteria.improvement_threshold *= 1.1
    
    def get_fidelity_statistics(self) -> Dict[str, Any]:
        
        fidelity_counts = {level: 0 for level in FidelityLevel}
        total_time_by_fidelity = {level: 0.0 for level in FidelityLevel}
        
        for entry in self.evaluation_history:
            fidelity = entry['result'].fidelity_level
            fidelity_counts[fidelity] += 1
            total_time_by_fidelity[fidelity] += entry['computation_time']
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'fidelity_distribution': {level.value: count for level, count in fidelity_counts.items()},
            'computation_time_by_fidelity': {level.value: time for level, time in total_time_by_fidelity.items()},
            'total_computation_time': self.used_budget,
            'budget_utilization': self.used_budget / self.computational_budget,
            'consecutive_high_fidelity': self.consecutive_high_fidelity,
            'current_strategy': self.strategy.value
        }
    
    def reset(self):
        
        self.evaluation_history = []
        self.used_budget = 0.0
        self.consecutive_high_fidelity = 0
        self.last_improvement = float('inf')
        self.trust_region_radius = 1.0