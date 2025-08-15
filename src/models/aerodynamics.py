import numpy as np
import math
from typing import Dict, Tuple, Optional
from ..simulation.base import BaseSimulation, SimulationResult, FidelityLevel


class LowFidelityAerodynamics(BaseSimulation):
    def __init__(self):
        super().__init__("LowFidelityAerodynamics", FidelityLevel.LOW)
        
        self.parameter_bounds = {
            'wingspan': (10.0, 80.0),
            'wing_area': (20.0, 500.0),
            'aspect_ratio': (5.0, 15.0),
            'sweep_angle': (0.0, 45.0),
            'taper_ratio': (0.3, 1.0),
            'thickness_ratio': (0.08, 0.18),
            'cruise_altitude': (1000.0, 15000.0),
            'cruise_mach': (0.1, 0.9),
            'weight': (5000.0, 100000.0)
        }
        
        self.objective_names = ['lift_to_drag_ratio', 'fuel_efficiency', 'range', 'payload_capacity']
        self.constraint_names = ['stall_speed', 'structural_margin']
    
    def validate_parameters(self, parameters: Dict[str, float]) -> bool:
        for param, value in parameters.items():
            if param in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param]
                if not (min_val <= value <= max_val):
                    return False
        
        if 'wing_area' in parameters and 'wingspan' in parameters:
            aspect_ratio_calc = (parameters['wingspan'] ** 2) / parameters['wing_area']
            if 'aspect_ratio' in parameters:
                if abs(aspect_ratio_calc - parameters['aspect_ratio']) > 0.5:
                    return False
        
        return True
    
    def get_computational_cost(self) -> float:
        return 0.1
    
    def evaluate(self, parameters: Dict[str, float]) -> SimulationResult:
        wingspan = parameters.get('wingspan', 40.0)
        wing_area = parameters.get('wing_area', 200.0)
        aspect_ratio = parameters.get('aspect_ratio', (wingspan**2)/wing_area)
        sweep_angle = parameters.get('sweep_angle', 0.0)
        taper_ratio = parameters.get('taper_ratio', 0.6)
        thickness_ratio = parameters.get('thickness_ratio', 0.12)
        altitude = parameters.get('cruise_altitude', 10000.0)
        mach = parameters.get('cruise_mach', 0.7)
        weight = parameters.get('weight', 50000.0)
        
        results = self._compute_aerodynamic_performance(
            wingspan, wing_area, aspect_ratio, sweep_angle, 
            taper_ratio, thickness_ratio, altitude, mach, weight
        )
        
        objectives = {
            'lift_to_drag_ratio': results['L_D'],
            'fuel_efficiency': results['fuel_efficiency'],
            'range': results['range'],
            'payload_capacity': results['payload']
        }
        
        constraints = {
            'stall_speed': results['stall_speed'],
            'structural_margin': results['structural_margin']
        }
        
        uncertainty = self._compute_uncertainty(parameters)
        
        return SimulationResult(
            objectives=objectives,
            constraints=constraints,
            fidelity_level=self.fidelity_level,
            uncertainty=uncertainty,
            metadata={'method': 'simplified_aerodynamics', 'model_version': '1.0'}
        )
    
    def _compute_aerodynamic_performance(self, wingspan: float, wing_area: float, 
                                       aspect_ratio: float, sweep_angle: float,
                                       taper_ratio: float, thickness_ratio: float,
                                       altitude: float, mach: float, weight: float) -> Dict[str, float]:
        
        sweep_rad = math.radians(sweep_angle)
        
        rho = self._atmospheric_density(altitude)
        
        cl_design = 0.4 + 0.3 * (1 - mach)
        
        cd0 = 0.015 + 0.01 * thickness_ratio + 0.005 * (sweep_angle / 45.0)
        
        e = 0.85 - 0.1 * (sweep_angle / 45.0) - 0.05 * (1 - taper_ratio)
        k = 1 / (math.pi * aspect_ratio * e)
        
        cd = cd0 + k * (cl_design ** 2)
        
        L_D = cl_design / cd
        
        velocity = mach * self._speed_of_sound(altitude)
        dynamic_pressure = 0.5 * rho * (velocity ** 2)
        
        lift = cl_design * dynamic_pressure * wing_area
        thrust_required = cd * dynamic_pressure * wing_area
        
        sfc = 0.5 + 0.2 * mach
        fuel_flow = thrust_required * sfc / 3600.0
        fuel_efficiency = velocity / fuel_flow if fuel_flow > 0 else 0
        
        fuel_fraction = 0.85
        range_km = (L_D * velocity * fuel_fraction * weight) / (sfc * 9.81 * thrust_required)
        range_km = max(0, range_km / 1000.0)
        
        cl_max = 1.2 + 0.3 * math.cos(sweep_rad) - 0.1 * thickness_ratio
        stall_speed = math.sqrt((2 * weight) / (rho * wing_area * cl_max))
        
        load_factor = weight / wing_area
        structural_margin = max(0, 1 - load_factor / 5000.0)
        
        payload_capacity = max(0, weight * 0.25 - 2000)
        
        return {
            'L_D': L_D,
            'fuel_efficiency': fuel_efficiency,
            'range': range_km,
            'stall_speed': stall_speed,
            'structural_margin': structural_margin,
            'payload': payload_capacity
        }
    
    def _atmospheric_density(self, altitude: float) -> float:
        if altitude < 11000:
            temp = 288.15 - 0.0065 * altitude
            pressure = 101325 * ((temp / 288.15) ** 5.2561)
        else:
            temp = 216.65
            pressure = 22632 * math.exp(-0.0001577 * (altitude - 11000))
        
        return pressure / (287 * temp)
    
    def _speed_of_sound(self, altitude: float) -> float:
        if altitude < 11000:
            temp = 288.15 - 0.0065 * altitude
        else:
            temp = 216.65
        
        return math.sqrt(1.4 * 287 * temp)
    
    def _compute_uncertainty(self, parameters: Dict[str, float]) -> Dict[str, float]:
        base_uncertainty = {
            'lift_to_drag_ratio': 0.15,
            'fuel_efficiency': 0.20,
            'range': 0.25,
            'payload_capacity': 0.10
        }
        
        mach = parameters.get('cruise_mach', 0.7)
        if mach > 0.8:
            for key in base_uncertainty:
                base_uncertainty[key] *= 1.5
        
        return base_uncertainty


class SpacecraftLowFidelity(BaseSimulation):
    def __init__(self):
        super().__init__("SpacecraftLowFidelity", FidelityLevel.LOW)
        
        self.parameter_bounds = {
            'dry_mass': (1000.0, 50000.0),
            'fuel_mass': (5000.0, 200000.0),
            'specific_impulse': (200.0, 450.0),
            'thrust': (1000.0, 100000.0),
            'solar_panel_area': (10.0, 200.0),
            'thermal_mass': (500.0, 10000.0),
            'target_orbit_altitude': (200.0, 35786.0),
            'mission_duration': (30.0, 3650.0)
        }
        
        self.objective_names = ['delta_v_capability', 'mass_efficiency', 'power_efficiency', 'thermal_stability']
        self.constraint_names = ['structural_load', 'power_balance']
    
    def validate_parameters(self, parameters: Dict[str, float]) -> bool:
        for param, value in parameters.items():
            if param in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param]
                if not (min_val <= value <= max_val):
                    return False
        
        if 'fuel_mass' in parameters and 'dry_mass' in parameters:
            if parameters['fuel_mass'] > parameters['dry_mass'] * 10:
                return False
        
        return True
    
    def get_computational_cost(self) -> float:
        return 0.05
    
    def evaluate(self, parameters: Dict[str, float]) -> SimulationResult:
        dry_mass = parameters.get('dry_mass', 5000.0)
        fuel_mass = parameters.get('fuel_mass', 20000.0)
        isp = parameters.get('specific_impulse', 300.0)
        thrust = parameters.get('thrust', 10000.0)
        solar_area = parameters.get('solar_panel_area', 50.0)
        thermal_mass = parameters.get('thermal_mass', 2000.0)
        orbit_alt = parameters.get('target_orbit_altitude', 400.0)
        mission_duration = parameters.get('mission_duration', 365.0)
        
        results = self._compute_spacecraft_performance(
            dry_mass, fuel_mass, isp, thrust, solar_area, 
            thermal_mass, orbit_alt, mission_duration
        )
        
        objectives = {
            'delta_v_capability': results['delta_v'],
            'mass_efficiency': results['mass_efficiency'],
            'power_efficiency': results['power_efficiency'],
            'thermal_stability': results['thermal_stability']
        }
        
        constraints = {
            'structural_load': results['structural_load'],
            'power_balance': results['power_balance']
        }
        
        uncertainty = self._compute_uncertainty(parameters)
        
        return SimulationResult(
            objectives=objectives,
            constraints=constraints,
            fidelity_level=self.fidelity_level,
            uncertainty=uncertainty,
            metadata={'method': 'simplified_orbital_mechanics', 'model_version': '1.0'}
        )
    
    def _compute_spacecraft_performance(self, dry_mass: float, fuel_mass: float, 
                                      isp: float, thrust: float, solar_area: float,
                                      thermal_mass: float, orbit_alt: float,
                                      mission_duration: float) -> Dict[str, float]:
        
        total_mass = dry_mass + fuel_mass
        mass_ratio = total_mass / dry_mass
        
        g0 = 9.81
        delta_v = isp * g0 * math.log(mass_ratio)
        
        mass_efficiency = fuel_mass / total_mass
        
        solar_flux = 1361.0
        solar_efficiency = 0.28
        power_generated = solar_area * solar_flux * solar_efficiency
        
        baseline_power_req = dry_mass * 0.05
        power_efficiency = power_generated / baseline_power_req if baseline_power_req > 0 else 0
        
        orbital_period = 2 * math.pi * math.sqrt(((6371 + orbit_alt) * 1000) ** 3 / (3.986e14))
        eclipse_fraction = 0.35 if orbit_alt < 1000 else 0.0
        
        thermal_variation = 200 * eclipse_fraction
        thermal_stability = max(0, 1 - thermal_variation / (thermal_mass * 0.1))
        
        thrust_to_weight = thrust / (total_mass * g0)
        structural_load = min(1.0, thrust_to_weight / 3.0)
        
        power_balance = min(1.0, power_generated / baseline_power_req) if baseline_power_req > 0 else 0
        
        return {
            'delta_v': delta_v,
            'mass_efficiency': mass_efficiency,
            'power_efficiency': power_efficiency,
            'thermal_stability': thermal_stability,
            'structural_load': structural_load,
            'power_balance': power_balance
        }
    
    def _compute_uncertainty(self, parameters: Dict[str, float]) -> Dict[str, float]:
        base_uncertainty = {
            'delta_v_capability': 0.10,
            'mass_efficiency': 0.05,
            'power_efficiency': 0.15,
            'thermal_stability': 0.20
        }
        
        mission_duration = parameters.get('mission_duration', 365.0)
        if mission_duration > 1000:
            for key in base_uncertainty:
                base_uncertainty[key] *= 1.3
        
        return base_uncertainty