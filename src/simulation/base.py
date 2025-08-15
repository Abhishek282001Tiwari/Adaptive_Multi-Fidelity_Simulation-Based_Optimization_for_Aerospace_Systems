from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from enum import Enum
import time
import logging


class FidelityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SimulationResult:
    def __init__(self, 
                 objectives: Dict[str, float],
                 constraints: Optional[Dict[str, float]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 fidelity_level: FidelityLevel = FidelityLevel.LOW,
                 computation_time: float = 0.0,
                 uncertainty: Optional[Dict[str, float]] = None):
        self.objectives = objectives
        self.constraints = constraints or {}
        self.metadata = metadata or {}
        self.fidelity_level = fidelity_level
        self.computation_time = computation_time
        self.uncertainty = uncertainty or {}
        self.timestamp = time.time()
    
    def __repr__(self):
        return f"SimulationResult(fidelity={self.fidelity_level.value}, objectives={self.objectives})"


class BaseSimulation(ABC):
    def __init__(self, name: str, fidelity_level: FidelityLevel):
        self.name = name
        self.fidelity_level = fidelity_level
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.computation_history = []
        self.parameter_bounds = {}
        self.objective_names = []
        self.constraint_names = []
    
    @abstractmethod
    def evaluate(self, parameters: Dict[str, float]) -> SimulationResult:
        pass
    
    @abstractmethod
    def get_computational_cost(self) -> float:
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, float]) -> bool:
        pass
    
    def set_parameter_bounds(self, bounds: Dict[str, Tuple[float, float]]):
        self.parameter_bounds = bounds
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        return self.parameter_bounds
    
    def get_expected_runtime(self, parameters: Dict[str, float]) -> float:
        base_cost = self.get_computational_cost()
        complexity_factor = self._estimate_complexity_factor(parameters)
        return base_cost * complexity_factor
    
    def _estimate_complexity_factor(self, parameters: Dict[str, float]) -> float:
        return 1.0
    
    def log_evaluation(self, parameters: Dict[str, float], result: SimulationResult):
        self.computation_history.append({
            'parameters': parameters.copy(),
            'result': result,
            'timestamp': time.time()
        })


class MultiFidelitySimulation:
    def __init__(self, name: str):
        self.name = name
        self.simulations = {}
        self.current_fidelity = FidelityLevel.LOW
        self.fidelity_history = []
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        self.fidelity_switching_criteria = {
            'max_low_fidelity_evals': 100,
            'convergence_threshold': 0.01,
            'uncertainty_threshold': 0.05,
            'computational_budget': 3600.0
        }
        
        self.total_computation_time = 0.0
        self.evaluation_count = {level: 0 for level in FidelityLevel}
    
    def add_simulation(self, simulation: BaseSimulation):
        self.simulations[simulation.fidelity_level] = simulation
        self.logger.info(f"Added {simulation.fidelity_level.value} fidelity simulation: {simulation.name}")
    
    def evaluate(self, parameters: Dict[str, float], 
                 force_fidelity: Optional[FidelityLevel] = None) -> SimulationResult:
        fidelity = force_fidelity or self._select_fidelity(parameters)
        
        if fidelity not in self.simulations:
            self.logger.warning(f"Requested fidelity {fidelity.value} not available, using {self.current_fidelity.value}")
            fidelity = self.current_fidelity
        
        simulation = self.simulations[fidelity]
        
        if not simulation.validate_parameters(parameters):
            raise ValueError(f"Invalid parameters for {fidelity.value} fidelity simulation")
        
        start_time = time.time()
        result = simulation.evaluate(parameters)
        end_time = time.time()
        
        computation_time = end_time - start_time
        result.computation_time = computation_time
        self.total_computation_time += computation_time
        self.evaluation_count[fidelity] += 1
        
        self.fidelity_history.append({
            'parameters': parameters.copy(),
            'fidelity': fidelity,
            'result': result,
            'timestamp': time.time()
        })
        
        simulation.log_evaluation(parameters, result)
        self.current_fidelity = fidelity
        
        return result
    
    def _select_fidelity(self, parameters: Dict[str, float]) -> FidelityLevel:
        if not self.fidelity_history:
            return FidelityLevel.LOW
        
        criteria = self.fidelity_switching_criteria
        
        if self.total_computation_time > criteria['computational_budget']:
            return FidelityLevel.LOW
        
        low_fidelity_count = self.evaluation_count[FidelityLevel.LOW]
        if low_fidelity_count < criteria['max_low_fidelity_evals']:
            return FidelityLevel.LOW
        
        if self._check_convergence():
            return FidelityLevel.HIGH
        
        if self._check_uncertainty_level():
            return FidelityLevel.HIGH
        
        return FidelityLevel.MEDIUM if FidelityLevel.MEDIUM in self.simulations else FidelityLevel.LOW
    
    def _check_convergence(self) -> bool:
        if len(self.fidelity_history) < 10:
            return False
        
        recent_results = [entry['result'] for entry in self.fidelity_history[-10:]]
        if not recent_results:
            return False
        
        objective_names = list(recent_results[0].objectives.keys())
        if not objective_names:
            return False
        
        for obj_name in objective_names:
            values = [result.objectives[obj_name] for result in recent_results]
            if len(values) > 1:
                relative_change = abs(values[-1] - values[0]) / (abs(values[0]) + 1e-10)
                if relative_change > self.fidelity_switching_criteria['convergence_threshold']:
                    return False
        
        return True
    
    def _check_uncertainty_level(self) -> bool:
        if len(self.fidelity_history) < 5:
            return False
        
        recent_results = [entry['result'] for entry in self.fidelity_history[-5:]]
        for result in recent_results:
            if result.uncertainty:
                avg_uncertainty = np.mean(list(result.uncertainty.values()))
                if avg_uncertainty > self.fidelity_switching_criteria['uncertainty_threshold']:
                    return True
        
        return False
    
    def get_fidelity_statistics(self) -> Dict[str, Any]:
        return {
            'total_evaluations': sum(self.evaluation_count.values()),
            'evaluations_by_fidelity': dict(self.evaluation_count),
            'total_computation_time': self.total_computation_time,
            'current_fidelity': self.current_fidelity.value,
            'fidelity_switches': len(set(entry['fidelity'] for entry in self.fidelity_history))
        }
    
    def reset_statistics(self):
        self.total_computation_time = 0.0
        self.evaluation_count = {level: 0 for level in FidelityLevel}
        self.fidelity_history = []
        for simulation in self.simulations.values():
            simulation.computation_history = []