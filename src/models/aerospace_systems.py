import numpy as np
from typing import Dict, List, Optional, Any
from ..simulation.base import MultiFidelitySimulation, FidelityLevel
from ..simulation.adaptive_fidelity import AdaptiveFidelityManager, FidelitySwitchingStrategy
from .aerodynamics import LowFidelityAerodynamics, SpacecraftLowFidelity
from .high_fidelity import HighFidelityAerodynamics, SpacecraftHighFidelity
import logging


class AircraftOptimizationSystem:
    def __init__(self, fidelity_strategy: FidelitySwitchingStrategy = FidelitySwitchingStrategy.ADAPTIVE):
        self.logger = logging.getLogger(f"{__name__}.AircraftOptimizationSystem")
        
        self.multi_fidelity_sim = MultiFidelitySimulation("AircraftSystem")
        
        self.low_fidelity_model = LowFidelityAerodynamics()
        self.high_fidelity_model = HighFidelityAerodynamics()
        
        self.multi_fidelity_sim.add_simulation(self.low_fidelity_model)
        self.multi_fidelity_sim.add_simulation(self.high_fidelity_model)
        
        self.fidelity_manager = AdaptiveFidelityManager(fidelity_strategy)
        
        self.design_constraints = {
            'max_takeoff_weight': 100000.0,
            'min_range': 1000.0,
            'max_stall_speed': 70.0,
            'min_fuel_efficiency': 10.0,
            'min_payload': 5000.0
        }
        
        self.environmental_conditions = {
            'temperature_variation': 20.0,
            'pressure_variation': 0.1,
            'wind_conditions': 'moderate',
            'turbulence_level': 'low'
        }
        
        self.mission_profiles = {
            'commercial': {
                'cruise_altitude': 11000.0,
                'cruise_mach': 0.78,
                'range_requirement': 5000.0,
                'payload_requirement': 15000.0
            },
            'regional': {
                'cruise_altitude': 8000.0,
                'cruise_mach': 0.65,
                'range_requirement': 2000.0,
                'payload_requirement': 8000.0
            },
            'business_jet': {
                'cruise_altitude': 12000.0,
                'cruise_mach': 0.80,
                'range_requirement': 6000.0,
                'payload_requirement': 2000.0
            }
        }
    
    def evaluate_design(self, parameters: Dict[str, float], 
                       mission_profile: str = 'commercial',
                       optimization_progress: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        if mission_profile in self.mission_profiles:
            mission_params = self.mission_profiles[mission_profile]
            parameters.update(mission_params)
        
        fidelity = self.fidelity_manager.select_fidelity(
            parameters, optimization_progress, self.multi_fidelity_sim
        )
        
        result = self.multi_fidelity_sim.evaluate(parameters, force_fidelity=fidelity)
        
        self.fidelity_manager.update_evaluation_history(
            parameters, result, result.computation_time
        )
        
        performance_metrics = self._compute_performance_metrics(result, mission_profile)
        
        constraint_violations = self._check_constraints(result, parameters)
        
        system_reliability = self._assess_system_reliability(result, parameters)
        
        return {
            'simulation_result': result,
            'performance_metrics': performance_metrics,
            'constraint_violations': constraint_violations,
            'system_reliability': system_reliability,
            'fidelity_used': fidelity,
            'mission_profile': mission_profile
        }
    
    def _compute_performance_metrics(self, result, mission_profile: str) -> Dict[str, float]:
        objectives = result.objectives
        
        if mission_profile == 'commercial':
            performance_score = (
                objectives.get('fuel_efficiency', 0) * 0.3 +
                objectives.get('range', 0) / 100.0 * 0.3 +
                objectives.get('payload_capacity', 0) / 1000.0 * 0.25 +
                objectives.get('lift_to_drag_ratio', 0) * 2.0 * 0.15
            )
        elif mission_profile == 'regional':
            performance_score = (
                objectives.get('fuel_efficiency', 0) * 0.25 +
                objectives.get('range', 0) / 100.0 * 0.25 +
                objectives.get('payload_capacity', 0) / 1000.0 * 0.3 +
                objectives.get('lift_to_drag_ratio', 0) * 2.0 * 0.2
            )
        else:
            performance_score = (
                objectives.get('fuel_efficiency', 0) * 0.35 +
                objectives.get('range', 0) / 100.0 * 0.35 +
                objectives.get('lift_to_drag_ratio', 0) * 2.0 * 0.3
            )
        
        efficiency_rating = min(10.0, max(0.0, objectives.get('fuel_efficiency', 0) / 5.0))
        
        range_rating = min(10.0, max(0.0, objectives.get('range', 0) / 1000.0))
        
        safety_margin = 1.0
        if hasattr(result, 'constraints'):
            stall_speed = result.constraints.get('stall_speed', 60.0)
            safety_margin = max(0.0, 1.0 - stall_speed / 100.0)
        
        return {
            'overall_performance': performance_score,
            'efficiency_rating': efficiency_rating,
            'range_rating': range_rating,
            'safety_margin': safety_margin
        }
    
    def _check_constraints(self, result, parameters: Dict[str, float]) -> Dict[str, float]:
        violations = {}
        
        constraints = getattr(result, 'constraints', {})
        objectives = result.objectives
        
        weight = parameters.get('weight', 50000.0)
        if weight > self.design_constraints['max_takeoff_weight']:
            violations['weight_violation'] = (weight - self.design_constraints['max_takeoff_weight']) / self.design_constraints['max_takeoff_weight']
        
        range_km = objectives.get('range', 0)
        if range_km < self.design_constraints['min_range']:
            violations['range_violation'] = (self.design_constraints['min_range'] - range_km) / self.design_constraints['min_range']
        
        stall_speed = constraints.get('stall_speed', 60.0)
        if stall_speed > self.design_constraints['max_stall_speed']:
            violations['stall_speed_violation'] = (stall_speed - self.design_constraints['max_stall_speed']) / self.design_constraints['max_stall_speed']
        
        fuel_efficiency = objectives.get('fuel_efficiency', 0)
        if fuel_efficiency < self.design_constraints['min_fuel_efficiency']:
            violations['efficiency_violation'] = (self.design_constraints['min_fuel_efficiency'] - fuel_efficiency) / self.design_constraints['min_fuel_efficiency']
        
        payload = objectives.get('payload_capacity', 0)
        if payload < self.design_constraints['min_payload']:
            violations['payload_violation'] = (self.design_constraints['min_payload'] - payload) / self.design_constraints['min_payload']
        
        return violations
    
    def _assess_system_reliability(self, result, parameters: Dict[str, float]) -> Dict[str, float]:
        base_reliability = 0.95
        
        structural_margin = getattr(result, 'constraints', {}).get('structural_margin', 1.0)
        reliability_structural = base_reliability * structural_margin
        
        aspect_ratio = parameters.get('aspect_ratio', 8.0)
        if aspect_ratio > 12:
            reliability_structural *= 0.95
        elif aspect_ratio < 6:
            reliability_structural *= 0.90
        
        sweep_angle = parameters.get('sweep_angle', 0.0)
        if sweep_angle > 35:
            reliability_structural *= 0.92
        
        mach = parameters.get('cruise_mach', 0.7)
        if mach > 0.85:
            reliability_aerodynamic = base_reliability * 0.88
        else:
            reliability_aerodynamic = base_reliability
        
        overall_reliability = min(reliability_structural, reliability_aerodynamic)
        
        return {
            'structural_reliability': reliability_structural,
            'aerodynamic_reliability': reliability_aerodynamic,
            'overall_reliability': overall_reliability
        }
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        fidelity_stats = self.fidelity_manager.get_fidelity_statistics()
        simulation_stats = self.multi_fidelity_sim.get_fidelity_statistics()
        
        return {
            'fidelity_manager': fidelity_stats,
            'simulation_system': simulation_stats
        }


class SpacecraftOptimizationSystem:
    def __init__(self, fidelity_strategy: FidelitySwitchingStrategy = FidelitySwitchingStrategy.ADAPTIVE):
        self.logger = logging.getLogger(f"{__name__}.SpacecraftOptimizationSystem")
        
        self.multi_fidelity_sim = MultiFidelitySimulation("SpacecraftSystem")
        
        self.low_fidelity_model = SpacecraftLowFidelity()
        self.high_fidelity_model = SpacecraftHighFidelity()
        
        self.multi_fidelity_sim.add_simulation(self.low_fidelity_model)
        self.multi_fidelity_sim.add_simulation(self.high_fidelity_model)
        
        self.fidelity_manager = AdaptiveFidelityManager(fidelity_strategy)
        
        self.mission_constraints = {
            'max_total_mass': 250000.0,
            'min_delta_v': 3000.0,
            'min_mission_duration': 365.0,
            'min_power_margin': 0.2,
            'min_thermal_stability': 0.7
        }
        
        self.orbital_environments = {
            'leo': {
                'altitude': 400.0,
                'radiation_level': 'moderate',
                'debris_density': 'high',
                'atmospheric_drag': 'significant'
            },
            'meo': {
                'altitude': 10000.0,
                'radiation_level': 'high',
                'debris_density': 'moderate',
                'atmospheric_drag': 'negligible'
            },
            'geo': {
                'altitude': 35786.0,
                'radiation_level': 'very_high',
                'debris_density': 'low',
                'atmospheric_drag': 'none'
            }
        }
        
        self.mission_types = {
            'earth_observation': {
                'target_orbit_altitude': 700.0,
                'mission_duration': 1825.0,
                'payload_mass': 2000.0,
                'pointing_accuracy': 'high'
            },
            'communication': {
                'target_orbit_altitude': 35786.0,
                'mission_duration': 5475.0,
                'payload_mass': 3000.0,
                'power_requirement': 'high'
            },
            'deep_space': {
                'target_orbit_altitude': 1000.0,
                'mission_duration': 3650.0,
                'payload_mass': 1500.0,
                'delta_v_requirement': 'very_high'
            }
        }
    
    def evaluate_design(self, parameters: Dict[str, float], 
                       mission_type: str = 'earth_observation',
                       optimization_progress: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        if mission_type in self.mission_types:
            mission_params = self.mission_types[mission_type]
            parameters.update(mission_params)
        
        fidelity = self.fidelity_manager.select_fidelity(
            parameters, optimization_progress, self.multi_fidelity_sim
        )
        
        result = self.multi_fidelity_sim.evaluate(parameters, force_fidelity=fidelity)
        
        self.fidelity_manager.update_evaluation_history(
            parameters, result, result.computation_time
        )
        
        performance_metrics = self._compute_performance_metrics(result, mission_type)
        
        constraint_violations = self._check_constraints(result, parameters)
        
        mission_success_probability = self._assess_mission_success(result, parameters, mission_type)
        
        return {
            'simulation_result': result,
            'performance_metrics': performance_metrics,
            'constraint_violations': constraint_violations,
            'mission_success_probability': mission_success_probability,
            'fidelity_used': fidelity,
            'mission_type': mission_type
        }
    
    def _compute_performance_metrics(self, result, mission_type: str) -> Dict[str, float]:
        objectives = result.objectives
        
        if mission_type == 'earth_observation':
            performance_score = (
                objectives.get('mass_efficiency', 0) * 0.25 +
                objectives.get('power_efficiency', 0) * 0.3 +
                objectives.get('thermal_stability', 0) * 0.25 +
                objectives.get('delta_v_capability', 0) / 1000.0 * 0.2
            )
        elif mission_type == 'communication':
            performance_score = (
                objectives.get('power_efficiency', 0) * 0.4 +
                objectives.get('thermal_stability', 0) * 0.3 +
                objectives.get('mass_efficiency', 0) * 0.2 +
                objectives.get('delta_v_capability', 0) / 1000.0 * 0.1
            )
        else:
            performance_score = (
                objectives.get('delta_v_capability', 0) / 1000.0 * 0.4 +
                objectives.get('mass_efficiency', 0) * 0.3 +
                objectives.get('power_efficiency', 0) * 0.2 +
                objectives.get('thermal_stability', 0) * 0.1
            )
        
        propulsion_efficiency = min(10.0, max(0.0, objectives.get('delta_v_capability', 0) / 500.0))
        
        power_rating = min(10.0, max(0.0, objectives.get('power_efficiency', 0) * 10.0))
        
        survivability = objectives.get('thermal_stability', 0.5) * 10.0
        
        return {
            'overall_performance': performance_score,
            'propulsion_efficiency': propulsion_efficiency,
            'power_rating': power_rating,
            'survivability': survivability
        }
    
    def _check_constraints(self, result, parameters: Dict[str, float]) -> Dict[str, float]:
        violations = {}
        
        constraints = getattr(result, 'constraints', {})
        objectives = result.objectives
        
        total_mass = (parameters.get('dry_mass', 5000) + 
                     parameters.get('fuel_mass', 20000) + 
                     parameters.get('payload_mass', 1000))
        
        if total_mass > self.mission_constraints['max_total_mass']:
            violations['mass_violation'] = (total_mass - self.mission_constraints['max_total_mass']) / self.mission_constraints['max_total_mass']
        
        delta_v = objectives.get('delta_v_capability', 0)
        if delta_v < self.mission_constraints['min_delta_v']:
            violations['delta_v_violation'] = (self.mission_constraints['min_delta_v'] - delta_v) / self.mission_constraints['min_delta_v']
        
        mission_duration = parameters.get('mission_duration', 365.0)
        if mission_duration < self.mission_constraints['min_mission_duration']:
            violations['duration_violation'] = (self.mission_constraints['min_mission_duration'] - mission_duration) / self.mission_constraints['min_mission_duration']
        
        power_balance = constraints.get('power_balance', 1.0)
        required_margin = 1.0 + self.mission_constraints['min_power_margin']
        if power_balance < required_margin:
            violations['power_violation'] = (required_margin - power_balance) / required_margin
        
        thermal_stability = objectives.get('thermal_stability', 0)
        if thermal_stability < self.mission_constraints['min_thermal_stability']:
            violations['thermal_violation'] = (self.mission_constraints['min_thermal_stability'] - thermal_stability) / self.mission_constraints['min_thermal_stability']
        
        return violations
    
    def _assess_mission_success(self, result, parameters: Dict[str, float], mission_type: str) -> Dict[str, float]:
        base_success_probability = 0.85
        
        constraints = getattr(result, 'constraints', {})
        
        structural_load = constraints.get('structural_load', 1.0)
        structural_success = base_success_probability * structural_load
        
        power_balance = constraints.get('power_balance', 1.0)
        power_success = base_success_probability * min(1.0, power_balance)
        
        orbit_altitude = parameters.get('target_orbit_altitude', 400.0)
        if orbit_altitude < 300:
            orbital_success = base_success_probability * 0.7
        elif orbit_altitude > 35000:
            orbital_success = base_success_probability * 0.9
        else:
            orbital_success = base_success_probability * 0.85
        
        mission_duration = parameters.get('mission_duration', 365.0)
        duration_factor = min(1.0, max(0.5, 1.0 - (mission_duration - 365) / 3650 * 0.3))
        duration_success = base_success_probability * duration_factor
        
        thermal_stability = result.objectives.get('thermal_stability', 0.5)
        thermal_success = base_success_probability * thermal_stability
        
        overall_success = min(structural_success, power_success, orbital_success, 
                            duration_success, thermal_success)
        
        return {
            'structural_success': structural_success,
            'power_success': power_success,
            'orbital_success': orbital_success,
            'duration_success': duration_success,
            'thermal_success': thermal_success,
            'overall_success': overall_success
        }
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        fidelity_stats = self.fidelity_manager.get_fidelity_statistics()
        simulation_stats = self.multi_fidelity_sim.get_fidelity_statistics()
        
        return {
            'fidelity_manager': fidelity_stats,
            'simulation_system': simulation_stats
        }


class IntegratedAerospaceSystem:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.IntegratedAerospaceSystem")
        
        self.aircraft_system = AircraftOptimizationSystem()
        self.spacecraft_system = SpacecraftOptimizationSystem()
        
        self.comparative_metrics = {}
        
    def evaluate_aircraft_design(self, parameters: Dict[str, float], 
                               mission_profile: str = 'commercial') -> Dict[str, Any]:
        return self.aircraft_system.evaluate_design(parameters, mission_profile)
    
    def evaluate_spacecraft_design(self, parameters: Dict[str, float], 
                                 mission_type: str = 'earth_observation') -> Dict[str, Any]:
        return self.spacecraft_system.evaluate_design(parameters, mission_type)
    
    def compare_designs(self, aircraft_params: Dict[str, float], 
                       spacecraft_params: Dict[str, float]) -> Dict[str, Any]:
        
        aircraft_result = self.evaluate_aircraft_design(aircraft_params)
        spacecraft_result = self.evaluate_spacecraft_design(spacecraft_params)
        
        comparison = {
            'aircraft': {
                'performance': aircraft_result['performance_metrics']['overall_performance'],
                'constraints_satisfied': len(aircraft_result['constraint_violations']) == 0,
                'reliability': aircraft_result['system_reliability']['overall_reliability']
            },
            'spacecraft': {
                'performance': spacecraft_result['performance_metrics']['overall_performance'],
                'constraints_satisfied': len(spacecraft_result['constraint_violations']) == 0,
                'mission_success': spacecraft_result['mission_success_probability']['overall_success']
            }
        }
        
        return {
            'aircraft_evaluation': aircraft_result,
            'spacecraft_evaluation': spacecraft_result,
            'comparison': comparison
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        return {
            'aircraft_system': self.aircraft_system.get_optimization_statistics(),
            'spacecraft_system': self.spacecraft_system.get_optimization_statistics()
        }