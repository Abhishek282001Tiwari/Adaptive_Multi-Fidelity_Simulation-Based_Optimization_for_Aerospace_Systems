import numpy as np
import math
from typing import Dict, Tuple, Optional, List
from ..simulation.base import BaseSimulation, SimulationResult, FidelityLevel


class HighFidelityAerodynamics(BaseSimulation):
    def __init__(self):
        super().__init__("HighFidelityAerodynamics", FidelityLevel.HIGH)
        
        self.parameter_bounds = {
            'wingspan': (10.0, 80.0),
            'wing_area': (20.0, 500.0),
            'aspect_ratio': (5.0, 15.0),
            'sweep_angle': (0.0, 45.0),
            'taper_ratio': (0.3, 1.0),
            'thickness_ratio': (0.08, 0.18),
            'cruise_altitude': (1000.0, 15000.0),
            'cruise_mach': (0.1, 0.9),
            'weight': (5000.0, 100000.0),
            'wing_twist': (-5.0, 5.0),
            'dihedral_angle': (-10.0, 10.0),
            'reynolds_number': (1e6, 1e8)
        }
        
        self.objective_names = ['lift_to_drag_ratio', 'fuel_efficiency', 'range', 'payload_capacity']
        self.constraint_names = ['stall_speed', 'structural_margin', 'flutter_margin']
        
        self.grid_density = 'fine'
        self.turbulence_model = 'k_omega_sst'
        self.boundary_conditions = 'no_slip_wall'
    
    def validate_parameters(self, parameters: Dict[str, float]) -> bool:
        for param, value in parameters.items():
            if param in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param]
                if not (min_val <= value <= max_val):
                    return False
        
        if 'wing_area' in parameters and 'wingspan' in parameters:
            aspect_ratio_calc = (parameters['wingspan'] ** 2) / parameters['wing_area']
            if 'aspect_ratio' in parameters:
                if abs(aspect_ratio_calc - parameters['aspect_ratio']) > 0.1:
                    return False
        
        return True
    
    def get_computational_cost(self) -> float:
        return 15.0
    
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
        wing_twist = parameters.get('wing_twist', 0.0)
        dihedral = parameters.get('dihedral_angle', 0.0)
        
        reynolds = self._calculate_reynolds_number(mach, altitude, wingspan)
        
        results = self._compute_cfd_performance(
            wingspan, wing_area, aspect_ratio, sweep_angle, 
            taper_ratio, thickness_ratio, altitude, mach, weight,
            wing_twist, dihedral, reynolds
        )
        
        objectives = {
            'lift_to_drag_ratio': results['L_D'],
            'fuel_efficiency': results['fuel_efficiency'],
            'range': results['range'],
            'payload_capacity': results['payload']
        }
        
        constraints = {
            'stall_speed': results['stall_speed'],
            'structural_margin': results['structural_margin'],
            'flutter_margin': results['flutter_margin']
        }
        
        uncertainty = self._compute_uncertainty(parameters)
        
        return SimulationResult(
            objectives=objectives,
            constraints=constraints,
            fidelity_level=self.fidelity_level,
            uncertainty=uncertainty,
            metadata={
                'method': 'cfd_approximation', 
                'model_version': '2.0',
                'grid_density': self.grid_density,
                'turbulence_model': self.turbulence_model,
                'reynolds_number': reynolds
            }
        )
    
    def _calculate_reynolds_number(self, mach: float, altitude: float, char_length: float) -> float:
        rho = self._atmospheric_density(altitude)
        velocity = mach * self._speed_of_sound(altitude)
        mu = self._dynamic_viscosity(altitude)
        
        return rho * velocity * char_length / mu
    
    def _dynamic_viscosity(self, altitude: float) -> float:
        if altitude < 11000:
            temp = 288.15 - 0.0065 * altitude
        else:
            temp = 216.65
        
        mu_ref = 1.716e-5
        temp_ref = 273.15
        sutherland_const = 110.4
        
        return mu_ref * ((temp / temp_ref) ** 1.5) * ((temp_ref + sutherland_const) / (temp + sutherland_const))
    
    def _compute_cfd_performance(self, wingspan: float, wing_area: float, 
                               aspect_ratio: float, sweep_angle: float,
                               taper_ratio: float, thickness_ratio: float,
                               altitude: float, mach: float, weight: float,
                               wing_twist: float, dihedral: float, reynolds: float) -> Dict[str, float]:
        
        sweep_rad = math.radians(sweep_angle)
        twist_rad = math.radians(wing_twist)
        dihedral_rad = math.radians(dihedral)
        
        rho = self._atmospheric_density(altitude)
        velocity = mach * self._speed_of_sound(altitude)
        
        cl_design = self._compute_cl_with_viscous_effects(mach, reynolds, aspect_ratio, sweep_angle, thickness_ratio)
        
        cd0 = self._compute_viscous_drag(reynolds, mach, thickness_ratio, sweep_angle, wing_area)
        
        e_factor = self._compute_span_efficiency(aspect_ratio, sweep_angle, taper_ratio, wing_twist, dihedral)
        k = 1 / (math.pi * aspect_ratio * e_factor)
        
        cd_induced = k * (cl_design ** 2)
        cd_compressibility = self._compute_compressibility_drag(mach, thickness_ratio, sweep_angle)
        
        cd_total = cd0 + cd_induced + cd_compressibility
        
        L_D = cl_design / cd_total
        
        dynamic_pressure = 0.5 * rho * (velocity ** 2)
        lift = cl_design * dynamic_pressure * wing_area
        thrust_required = cd_total * dynamic_pressure * wing_area
        
        sfc = self._compute_specific_fuel_consumption(mach, altitude)
        fuel_flow = thrust_required * sfc / 3600.0
        fuel_efficiency = velocity / fuel_flow if fuel_flow > 0 else 0
        
        range_factor = self._compute_range_factor(L_D, mach, altitude)
        fuel_fraction = 0.87
        range_km = range_factor * fuel_fraction * weight / 1000.0
        
        cl_max = self._compute_cl_max_viscous(reynolds, aspect_ratio, sweep_angle, thickness_ratio, wing_twist)
        stall_speed = math.sqrt((2 * weight) / (rho * wing_area * cl_max))
        
        flutter_speed = self._compute_flutter_speed(wingspan, aspect_ratio, sweep_angle, thickness_ratio)
        flutter_margin = flutter_speed / velocity if velocity > 0 else float('inf')
        
        structural_margin = self._compute_structural_margin(weight, wing_area, aspect_ratio, sweep_angle)
        
        payload_capacity = max(0, weight * 0.28 - 1500)
        
        return {
            'L_D': L_D,
            'fuel_efficiency': fuel_efficiency,
            'range': range_km,
            'stall_speed': stall_speed,
            'structural_margin': structural_margin,
            'flutter_margin': flutter_margin,
            'payload': payload_capacity
        }
    
    def _compute_cl_with_viscous_effects(self, mach: float, reynolds: float, 
                                       aspect_ratio: float, sweep_angle: float, 
                                       thickness_ratio: float) -> float:
        
        cl_inviscid = 0.45 + 0.25 * (1 - mach)
        
        reynolds_factor = 1 + 0.1 * math.log10(reynolds / 1e6)
        
        aspect_ratio_factor = 1 + 0.05 * (aspect_ratio - 8) / 8
        
        sweep_factor = math.cos(math.radians(sweep_angle)) ** 0.5
        
        thickness_factor = 1 - 0.3 * (thickness_ratio - 0.12) / 0.06
        
        return cl_inviscid * reynolds_factor * aspect_ratio_factor * sweep_factor * thickness_factor
    
    def _compute_viscous_drag(self, reynolds: float, mach: float, 
                            thickness_ratio: float, sweep_angle: float, 
                            wing_area: float) -> float:
        
        cf = 0.455 / ((math.log10(reynolds)) ** 2.58)
        
        form_factor = 1 + 2 * thickness_ratio + 100 * (thickness_ratio ** 4)
        
        sweep_factor = 1 + 0.2 * (sweep_angle / 45.0)
        
        compressibility_factor = 1 + 0.34 * (mach ** 2) if mach < 0.8 else 1 + 2.0 * ((mach - 0.8) ** 2)
        
        wetted_area_ratio = 2.05
        
        return cf * form_factor * sweep_factor * compressibility_factor * wetted_area_ratio
    
    def _compute_span_efficiency(self, aspect_ratio: float, sweep_angle: float, 
                               taper_ratio: float, wing_twist: float, dihedral: float) -> float:
        
        e_basic = 0.9 - 0.05 * (sweep_angle / 45.0)
        
        taper_correction = 1 - 0.1 * (1 - taper_ratio)
        
        twist_correction = 1 + 0.02 * abs(wing_twist)
        
        dihedral_correction = 1 - 0.01 * abs(dihedral)
        
        return e_basic * taper_correction * twist_correction * dihedral_correction
    
    def _compute_compressibility_drag(self, mach: float, thickness_ratio: float, sweep_angle: float) -> float:
        
        mach_crit = 0.7 + 0.1 * math.cos(math.radians(sweep_angle)) - 0.5 * thickness_ratio
        
        if mach < mach_crit:
            return 0.0
        else:
            drag_rise = 20 * ((mach - mach_crit) ** 2)
            return drag_rise * thickness_ratio
    
    def _compute_specific_fuel_consumption(self, mach: float, altitude: float) -> float:
        
        base_sfc = 0.5
        mach_factor = 1 + 0.3 * mach
        altitude_factor = 1 - 0.00002 * altitude
        
        return base_sfc * mach_factor * altitude_factor
    
    def _compute_range_factor(self, L_D: float, mach: float, altitude: float) -> float:
        
        velocity = mach * self._speed_of_sound(altitude)
        return L_D * velocity * 0.001
    
    def _compute_cl_max_viscous(self, reynolds: float, aspect_ratio: float, 
                              sweep_angle: float, thickness_ratio: float, wing_twist: float) -> float:
        
        cl_max_basic = 1.4 + 0.4 * thickness_ratio
        
        reynolds_effect = 1 + 0.15 * math.log10(reynolds / 5e6)
        
        sweep_effect = math.cos(math.radians(sweep_angle)) ** 0.8
        
        twist_effect = 1 + 0.05 * abs(wing_twist)
        
        return cl_max_basic * reynolds_effect * sweep_effect * twist_effect
    
    def _compute_flutter_speed(self, wingspan: float, aspect_ratio: float, 
                             sweep_angle: float, thickness_ratio: float) -> float:
        
        base_flutter = 250.0
        
        stiffness_factor = thickness_ratio / 0.12
        sweep_factor = math.cos(math.radians(sweep_angle))
        aspect_ratio_factor = math.sqrt(8.0 / aspect_ratio)
        
        return base_flutter * stiffness_factor * sweep_factor * aspect_ratio_factor
    
    def _compute_structural_margin(self, weight: float, wing_area: float, 
                                 aspect_ratio: float, sweep_angle: float) -> float:
        
        wing_loading = weight / wing_area
        
        allowable_loading = 4000 * (1 + 0.1 * math.cos(math.radians(sweep_angle)))
        allowable_loading *= math.sqrt(aspect_ratio / 8.0)
        
        return max(0, 1 - wing_loading / allowable_loading)
    
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
            'lift_to_drag_ratio': 0.05,
            'fuel_efficiency': 0.08,
            'range': 0.10,
            'payload_capacity': 0.05
        }
        
        mach = parameters.get('cruise_mach', 0.7)
        reynolds = parameters.get('reynolds_number', 1e7)
        
        if mach > 0.8:
            for key in base_uncertainty:
                base_uncertainty[key] *= 1.2
        
        if reynolds < 5e6:
            for key in base_uncertainty:
                base_uncertainty[key] *= 1.3
        
        return base_uncertainty


class SpacecraftHighFidelity(BaseSimulation):
    def __init__(self):
        super().__init__("SpacecraftHighFidelity", FidelityLevel.HIGH)
        
        self.parameter_bounds = {
            'dry_mass': (1000.0, 50000.0),
            'fuel_mass': (5000.0, 200000.0),
            'specific_impulse': (200.0, 450.0),
            'thrust': (1000.0, 100000.0),
            'solar_panel_area': (10.0, 200.0),
            'thermal_mass': (500.0, 10000.0),
            'target_orbit_altitude': (200.0, 35786.0),
            'mission_duration': (30.0, 3650.0),
            'propellant_efficiency': (0.8, 0.98),
            'attitude_control_mass': (50.0, 1000.0),
            'payload_mass': (100.0, 10000.0)
        }
        
        self.objective_names = ['delta_v_capability', 'mass_efficiency', 'power_efficiency', 'thermal_stability']
        self.constraint_names = ['structural_load', 'power_balance', 'orbital_decay']
    
    def validate_parameters(self, parameters: Dict[str, float]) -> bool:
        for param, value in parameters.items():
            if param in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param]
                if not (min_val <= value <= max_val):
                    return False
        
        total_mass = (parameters.get('dry_mass', 5000) + 
                     parameters.get('fuel_mass', 20000) + 
                     parameters.get('payload_mass', 1000) + 
                     parameters.get('attitude_control_mass', 200))
        
        if total_mass > 300000:
            return False
        
        return True
    
    def get_computational_cost(self) -> float:
        return 25.0
    
    def evaluate(self, parameters: Dict[str, float]) -> SimulationResult:
        dry_mass = parameters.get('dry_mass', 5000.0)
        fuel_mass = parameters.get('fuel_mass', 20000.0)
        isp = parameters.get('specific_impulse', 300.0)
        thrust = parameters.get('thrust', 10000.0)
        solar_area = parameters.get('solar_panel_area', 50.0)
        thermal_mass = parameters.get('thermal_mass', 2000.0)
        orbit_alt = parameters.get('target_orbit_altitude', 400.0)
        mission_duration = parameters.get('mission_duration', 365.0)
        prop_efficiency = parameters.get('propellant_efficiency', 0.9)
        attitude_mass = parameters.get('attitude_control_mass', 200.0)
        payload_mass = parameters.get('payload_mass', 1000.0)
        
        results = self._compute_detailed_spacecraft_performance(
            dry_mass, fuel_mass, isp, thrust, solar_area, thermal_mass,
            orbit_alt, mission_duration, prop_efficiency, attitude_mass, payload_mass
        )
        
        objectives = {
            'delta_v_capability': results['delta_v'],
            'mass_efficiency': results['mass_efficiency'],
            'power_efficiency': results['power_efficiency'],
            'thermal_stability': results['thermal_stability']
        }
        
        constraints = {
            'structural_load': results['structural_load'],
            'power_balance': results['power_balance'],
            'orbital_decay': results['orbital_decay']
        }
        
        uncertainty = self._compute_uncertainty(parameters)
        
        return SimulationResult(
            objectives=objectives,
            constraints=constraints,
            fidelity_level=self.fidelity_level,
            uncertainty=uncertainty,
            metadata={
                'method': 'detailed_orbital_mechanics', 
                'model_version': '2.0',
                'includes_perturbations': True,
                'thermal_analysis': 'detailed'
            }
        )
    
    def _compute_detailed_spacecraft_performance(self, dry_mass: float, fuel_mass: float, 
                                               isp: float, thrust: float, solar_area: float,
                                               thermal_mass: float, orbit_alt: float,
                                               mission_duration: float, prop_efficiency: float,
                                               attitude_mass: float, payload_mass: float) -> Dict[str, float]:
        
        total_mass = dry_mass + fuel_mass + payload_mass + attitude_mass
        effective_fuel = fuel_mass * prop_efficiency
        
        mass_ratio = total_mass / (total_mass - effective_fuel)
        
        g0 = 9.81
        delta_v_ideal = isp * g0 * math.log(mass_ratio)
        
        gravity_losses = self._compute_gravity_losses(orbit_alt, thrust, total_mass)
        atmospheric_losses = self._compute_atmospheric_losses(orbit_alt)
        steering_losses = self._compute_steering_losses(orbit_alt)
        
        delta_v_actual = delta_v_ideal - gravity_losses - atmospheric_losses - steering_losses
        
        mass_efficiency = effective_fuel / total_mass
        
        solar_flux = self._compute_solar_flux(orbit_alt)
        degradation_factor = self._compute_solar_degradation(mission_duration)
        solar_efficiency = 0.30 * degradation_factor
        
        eclipse_fraction = self._compute_eclipse_fraction(orbit_alt)
        battery_efficiency = 0.85
        
        average_power = solar_area * solar_flux * solar_efficiency * (1 - eclipse_fraction * (1 - battery_efficiency))
        
        power_requirements = self._compute_power_requirements(dry_mass, payload_mass, attitude_mass, mission_duration)
        power_efficiency = average_power / power_requirements if power_requirements > 0 else 0
        
        thermal_stability = self._compute_thermal_stability(thermal_mass, orbit_alt, solar_area, eclipse_fraction)
        
        thrust_to_weight = thrust / (total_mass * g0)
        structural_load = min(1.0, thrust_to_weight / 5.0)
        
        power_margin = (average_power - power_requirements) / power_requirements if power_requirements > 0 else 0
        power_balance = min(1.0, max(0.0, 1 + power_margin))
        
        orbital_decay_rate = self._compute_orbital_decay(orbit_alt, total_mass, solar_area)
        orbital_decay = max(0, 1 - orbital_decay_rate * mission_duration / orbit_alt)
        
        return {
            'delta_v': delta_v_actual,
            'mass_efficiency': mass_efficiency,
            'power_efficiency': power_efficiency,
            'thermal_stability': thermal_stability,
            'structural_load': structural_load,
            'power_balance': power_balance,
            'orbital_decay': orbital_decay
        }
    
    def _compute_gravity_losses(self, orbit_alt: float, thrust: float, mass: float) -> float:
        
        earth_radius = 6371000
        orbit_radius = earth_radius + orbit_alt * 1000
        
        g = 9.81 * (earth_radius / orbit_radius) ** 2
        acceleration = thrust / mass
        
        burn_time = 300.0
        
        return g * burn_time * 0.5
    
    def _compute_atmospheric_losses(self, orbit_alt: float) -> float:
        
        if orbit_alt > 150:
            return 0.0
        else:
            return max(0, 200 * (150 - orbit_alt) / 150)
    
    def _compute_steering_losses(self, orbit_alt: float) -> float:
        
        base_loss = 50.0
        altitude_factor = math.sqrt(orbit_alt / 400.0)
        return base_loss / altitude_factor
    
    def _compute_solar_flux(self, orbit_alt: float) -> float:
        
        base_flux = 1361.0
        earth_radius = 6371000
        orbit_radius = earth_radius + orbit_alt * 1000
        
        distance_factor = (1.496e11 / 1.496e11) ** 2
        
        return base_flux * distance_factor
    
    def _compute_solar_degradation(self, mission_duration: float) -> float:
        
        annual_degradation = 0.02
        years = mission_duration / 365.25
        return (1 - annual_degradation) ** years
    
    def _compute_eclipse_fraction(self, orbit_alt: float) -> float:
        
        earth_radius = 6371000
        orbit_radius = earth_radius + orbit_alt * 1000
        
        if orbit_radius > 3 * earth_radius:
            return 0.0
        else:
            beta_angle = math.asin(earth_radius / orbit_radius)
            return beta_angle / math.pi
    
    def _compute_power_requirements(self, dry_mass: float, payload_mass: float, 
                                  attitude_mass: float, mission_duration: float) -> float:
        
        baseline_power = dry_mass * 0.08
        payload_power = payload_mass * 0.15
        attitude_power = attitude_mass * 0.5
        
        aging_factor = 1 + 0.0001 * mission_duration
        
        return (baseline_power + payload_power + attitude_power) * aging_factor
    
    def _compute_thermal_stability(self, thermal_mass: float, orbit_alt: float, 
                                 solar_area: float, eclipse_fraction: float) -> float:
        
        solar_heating = solar_area * 1361 * 0.7
        earth_ir = solar_area * 240 * 0.8
        
        heat_input = solar_heating + earth_ir
        
        thermal_capacity = thermal_mass * 900
        
        eclipse_time = eclipse_fraction * 5400
        
        temp_swing = (heat_input * eclipse_time) / thermal_capacity
        
        return max(0, 1 - temp_swing / 100.0)
    
    def _compute_orbital_decay(self, orbit_alt: float, mass: float, area: float) -> float:
        
        if orbit_alt > 500:
            return 0.0
        
        atmospheric_density = 1e-12 * math.exp(-(orbit_alt - 300) / 50)
        
        drag_coefficient = 2.2
        velocity = math.sqrt(3.986e14 / ((6371 + orbit_alt) * 1000))
        
        drag_force = 0.5 * atmospheric_density * velocity ** 2 * drag_coefficient * area
        
        orbital_energy = -3.986e14 * mass / (2 * (6371 + orbit_alt) * 1000)
        
        energy_loss_rate = drag_force * velocity
        
        decay_rate = abs(energy_loss_rate / orbital_energy) * (6371 + orbit_alt) * 1000
        
        return decay_rate / 1000.0
    
    def _compute_uncertainty(self, parameters: Dict[str, float]) -> Dict[str, float]:
        
        base_uncertainty = {
            'delta_v_capability': 0.03,
            'mass_efficiency': 0.02,
            'power_efficiency': 0.08,
            'thermal_stability': 0.12
        }
        
        mission_duration = parameters.get('mission_duration', 365.0)
        orbit_alt = parameters.get('target_orbit_altitude', 400.0)
        
        if mission_duration > 1000:
            for key in base_uncertainty:
                base_uncertainty[key] *= 1.2
        
        if orbit_alt < 300:
            for key in base_uncertainty:
                base_uncertainty[key] *= 1.4
        
        return base_uncertainty