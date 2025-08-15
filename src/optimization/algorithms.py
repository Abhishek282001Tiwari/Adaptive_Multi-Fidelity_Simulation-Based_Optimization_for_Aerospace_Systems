import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    best_parameters: Dict[str, float]
    best_objectives: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    convergence_achieved: bool
    total_time: float
    algorithm_name: str
    metadata: Dict[str, Any]


class BaseOptimizer(ABC):
    def __init__(self, name: str, parameter_bounds: Dict[str, Tuple[float, float]]):
        self.name = name
        self.parameter_bounds = parameter_bounds
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.optimization_history = []
        self.best_solution = None
        self.best_fitness = float('-inf')
        
    @abstractmethod
    def optimize(self, objective_function: Callable, max_evaluations: int, **kwargs) -> OptimizationResult:
        pass
    
    def _normalize_parameters(self, parameters: Dict[str, float]) -> Dict[str, float]:
        normalized = {}
        for param, value in parameters.items():
            if param in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param]
                normalized[param] = (value - min_val) / (max_val - min_val)
            else:
                normalized[param] = value
        return normalized
    
    def _denormalize_parameters(self, normalized_params: Dict[str, float]) -> Dict[str, float]:
        denormalized = {}
        for param, norm_value in normalized_params.items():
            if param in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param]
                denormalized[param] = min_val + norm_value * (max_val - min_val)
            else:
                denormalized[param] = norm_value
        return denormalized
    
    def _ensure_bounds(self, parameters: Dict[str, float]) -> Dict[str, float]:
        bounded = {}
        for param, value in parameters.items():
            if param in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param]
                bounded[param] = max(min_val, min(max_val, value))
            else:
                bounded[param] = value
        return bounded


class GeneticAlgorithm(BaseOptimizer):
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]], 
                 population_size: int = 50, crossover_rate: float = 0.8, 
                 mutation_rate: float = 0.1, tournament_size: int = 3):
        super().__init__("GeneticAlgorithm", parameter_bounds)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = max(1, population_size // 10)
        
    def optimize(self, objective_function: Callable, max_evaluations: int, **kwargs) -> OptimizationResult:
        import time
        start_time = time.time()
        
        self.optimization_history = []
        
        population = self._initialize_population()
        fitness_scores = []
        
        evaluations = 0
        generation = 0
        
        for individual in population:
            if evaluations >= max_evaluations:
                break
            
            result = objective_function(individual)
            fitness = self._compute_fitness(result.objectives)
            fitness_scores.append(fitness)
            
            self.optimization_history.append({
                'generation': generation,
                'individual': individual.copy(),
                'objectives': result.objectives.copy(),
                'fitness': fitness,
                'evaluation': evaluations
            })
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = individual.copy()
            
            evaluations += 1
        
        while evaluations < max_evaluations:
            generation += 1
            
            new_population = []
            new_fitness_scores = []
            
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
                new_fitness_scores.append(fitness_scores[idx])
            
            while len(new_population) < self.population_size and evaluations < max_evaluations:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                for child in [child1, child2]:
                    if len(new_population) < self.population_size and evaluations < max_evaluations:
                        child = self._ensure_bounds(child)
                        result = objective_function(child)
                        fitness = self._compute_fitness(result.objectives)
                        
                        new_population.append(child)
                        new_fitness_scores.append(fitness)
                        
                        self.optimization_history.append({
                            'generation': generation,
                            'individual': child.copy(),
                            'objectives': result.objectives.copy(),
                            'fitness': fitness,
                            'evaluation': evaluations
                        })
                        
                        if fitness > self.best_fitness:
                            self.best_fitness = fitness
                            self.best_solution = child.copy()
                        
                        evaluations += 1
            
            population = new_population
            fitness_scores = new_fitness_scores
            
            if generation % 10 == 0:
                self.logger.info(f"Generation {generation}, Best fitness: {self.best_fitness:.6f}")
        
        end_time = time.time()
        
        best_objectives = {}
        if self.optimization_history:
            best_entry = max(self.optimization_history, key=lambda x: x['fitness'])
            best_objectives = best_entry['objectives']
        
        convergence = self._check_convergence()
        
        return OptimizationResult(
            best_parameters=self.best_solution or {},
            best_objectives=best_objectives,
            optimization_history=self.optimization_history,
            total_evaluations=evaluations,
            convergence_achieved=convergence,
            total_time=end_time - start_time,
            algorithm_name=self.name,
            metadata={
                'population_size': self.population_size,
                'final_generation': generation,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate
            }
        )
    
    def _initialize_population(self) -> List[Dict[str, float]]:
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, (min_val, max_val) in self.parameter_bounds.items():
                individual[param] = random.uniform(min_val, max_val)
            population.append(individual)
        return population
    
    def _compute_fitness(self, objectives: Dict[str, float]) -> float:
        return sum(objectives.values()) / len(objectives)
    
    def _tournament_selection(self, population: List[Dict[str, float]], 
                            fitness_scores: List[float]) -> Dict[str, float]:
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()
    
    def _crossover(self, parent1: Dict[str, float], 
                  parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        child1, child2 = parent1.copy(), parent2.copy()
        
        for param in parent1.keys():
            if random.random() < 0.5:
                alpha = random.random()
                child1[param] = alpha * parent1[param] + (1 - alpha) * parent2[param]
                child2[param] = alpha * parent2[param] + (1 - alpha) * parent1[param]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        mutated = individual.copy()
        
        for param, value in mutated.items():
            if random.random() < self.mutation_rate:
                if param in self.parameter_bounds:
                    min_val, max_val = self.parameter_bounds[param]
                    mutation_strength = 0.1 * (max_val - min_val)
                    mutated[param] = value + random.gauss(0, mutation_strength)
        
        return mutated
    
    def _check_convergence(self) -> bool:
        if len(self.optimization_history) < 20:
            return False
        
        recent_fitness = [entry['fitness'] for entry in self.optimization_history[-20:]]
        fitness_std = np.std(recent_fitness)
        return fitness_std < 0.001


class ParticleSwarmOptimization(BaseOptimizer):
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]], 
                 swarm_size: int = 30, w: float = 0.729, c1: float = 1.49445, c2: float = 1.49445):
        super().__init__("ParticleSwarmOptimization", parameter_bounds)
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
    def optimize(self, objective_function: Callable, max_evaluations: int, **kwargs) -> OptimizationResult:
        import time
        start_time = time.time()
        
        self.optimization_history = []
        
        particles = self._initialize_swarm()
        velocities = self._initialize_velocities()
        personal_best = [p.copy() for p in particles]
        personal_best_fitness = []
        
        evaluations = 0
        iteration = 0
        
        for i, particle in enumerate(particles):
            if evaluations >= max_evaluations:
                break
            
            result = objective_function(particle)
            fitness = self._compute_fitness(result.objectives)
            personal_best_fitness.append(fitness)
            
            self.optimization_history.append({
                'iteration': iteration,
                'particle': i,
                'position': particle.copy(),
                'objectives': result.objectives.copy(),
                'fitness': fitness,
                'evaluation': evaluations
            })
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = particle.copy()
            
            evaluations += 1
        
        global_best = self.best_solution.copy()
        
        while evaluations < max_evaluations:
            iteration += 1
            
            for i, particle in enumerate(particles):
                if evaluations >= max_evaluations:
                    break
                
                for param in particle.keys():
                    r1, r2 = random.random(), random.random()
                    
                    velocities[i][param] = (self.w * velocities[i][param] + 
                                          self.c1 * r1 * (personal_best[i][param] - particle[param]) +
                                          self.c2 * r2 * (global_best[param] - particle[param]))
                    
                    particle[param] += velocities[i][param]
                
                particle = self._ensure_bounds(particle)
                particles[i] = particle
                
                result = objective_function(particle)
                fitness = self._compute_fitness(result.objectives)
                
                self.optimization_history.append({
                    'iteration': iteration,
                    'particle': i,
                    'position': particle.copy(),
                    'objectives': result.objectives.copy(),
                    'fitness': fitness,
                    'evaluation': evaluations
                })
                
                if fitness > personal_best_fitness[i]:
                    personal_best[i] = particle.copy()
                    personal_best_fitness[i] = fitness
                
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = particle.copy()
                    global_best = particle.copy()
                
                evaluations += 1
            
            if iteration % 10 == 0:
                self.logger.info(f"Iteration {iteration}, Best fitness: {self.best_fitness:.6f}")
        
        end_time = time.time()
        
        best_objectives = {}
        if self.optimization_history:
            best_entry = max(self.optimization_history, key=lambda x: x['fitness'])
            best_objectives = best_entry['objectives']
        
        convergence = self._check_convergence()
        
        return OptimizationResult(
            best_parameters=self.best_solution or {},
            best_objectives=best_objectives,
            optimization_history=self.optimization_history,
            total_evaluations=evaluations,
            convergence_achieved=convergence,
            total_time=end_time - start_time,
            algorithm_name=self.name,
            metadata={
                'swarm_size': self.swarm_size,
                'final_iteration': iteration,
                'inertia_weight': self.w,
                'cognitive_parameter': self.c1,
                'social_parameter': self.c2
            }
        )
    
    def _initialize_swarm(self) -> List[Dict[str, float]]:
        swarm = []
        for _ in range(self.swarm_size):
            particle = {}
            for param, (min_val, max_val) in self.parameter_bounds.items():
                particle[param] = random.uniform(min_val, max_val)
            swarm.append(particle)
        return swarm
    
    def _initialize_velocities(self) -> List[Dict[str, float]]:
        velocities = []
        for _ in range(self.swarm_size):
            velocity = {}
            for param, (min_val, max_val) in self.parameter_bounds.items():
                velocity[param] = random.uniform(-(max_val - min_val) * 0.1, 
                                               (max_val - min_val) * 0.1)
            velocities.append(velocity)
        return velocities
    
    def _compute_fitness(self, objectives: Dict[str, float]) -> float:
        return sum(objectives.values()) / len(objectives)
    
    def _check_convergence(self) -> bool:
        if len(self.optimization_history) < 30:
            return False
        
        recent_fitness = [entry['fitness'] for entry in self.optimization_history[-30:]]
        fitness_std = np.std(recent_fitness)
        return fitness_std < 0.001


class BayesianOptimization(BaseOptimizer):
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]], 
                 acquisition_function: str = 'ei', xi: float = 0.01, kappa: float = 2.576):
        super().__init__("BayesianOptimization", parameter_bounds)
        self.acquisition_function = acquisition_function
        self.xi = xi
        self.kappa = kappa
        self.gp = None
        self.X_observed = []
        self.y_observed = []
        
    def optimize(self, objective_function: Callable, max_evaluations: int, **kwargs) -> OptimizationResult:
        import time
        start_time = time.time()
        
        self.optimization_history = []
        self.X_observed = []
        self.y_observed = []
        
        n_initial = min(10, max_evaluations // 4)
        
        for i in range(n_initial):
            params = self._random_sample()
            result = objective_function(params)
            fitness = self._compute_fitness(result.objectives)
            
            self.X_observed.append(self._params_to_array(params))
            self.y_observed.append(fitness)
            
            self.optimization_history.append({
                'iteration': i,
                'parameters': params.copy(),
                'objectives': result.objectives.copy(),
                'fitness': fitness,
                'acquisition_type': 'random',
                'evaluation': i
            })
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = params.copy()
        
        evaluations = n_initial
        
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        while evaluations < max_evaluations:
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
            
            next_params = self._propose_location()
            result = objective_function(next_params)
            fitness = self._compute_fitness(result.objectives)
            
            self.X_observed.append(self._params_to_array(next_params))
            self.y_observed.append(fitness)
            
            self.optimization_history.append({
                'iteration': evaluations,
                'parameters': next_params.copy(),
                'objectives': result.objectives.copy(),
                'fitness': fitness,
                'acquisition_type': self.acquisition_function,
                'evaluation': evaluations
            })
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = next_params.copy()
            
            evaluations += 1
            
            if evaluations % 5 == 0:
                self.logger.info(f"Evaluation {evaluations}, Best fitness: {self.best_fitness:.6f}")
        
        end_time = time.time()
        
        best_objectives = {}
        if self.optimization_history:
            best_entry = max(self.optimization_history, key=lambda x: x['fitness'])
            best_objectives = best_entry['objectives']
        
        convergence = self._check_convergence()
        
        return OptimizationResult(
            best_parameters=self.best_solution or {},
            best_objectives=best_objectives,
            optimization_history=self.optimization_history,
            total_evaluations=evaluations,
            convergence_achieved=convergence,
            total_time=end_time - start_time,
            algorithm_name=self.name,
            metadata={
                'acquisition_function': self.acquisition_function,
                'n_initial_points': n_initial,
                'xi': self.xi,
                'kappa': self.kappa
            }
        )
    
    def _random_sample(self) -> Dict[str, float]:
        params = {}
        for param, (min_val, max_val) in self.parameter_bounds.items():
            params[param] = random.uniform(min_val, max_val)
        return params
    
    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        return np.array([params[param] for param in sorted(self.parameter_bounds.keys())])
    
    def _array_to_params(self, array: np.ndarray) -> Dict[str, float]:
        params = {}
        for i, param in enumerate(sorted(self.parameter_bounds.keys())):
            params[param] = array[i]
        return params
    
    def _propose_location(self) -> Dict[str, float]:
        bounds = [(min_val, max_val) for min_val, max_val in self.parameter_bounds.values()]
        
        def objective(x):
            return -self._acquisition(x.reshape(1, -1))
        
        best_x = None
        best_acquisition_value = float('inf')
        
        for _ in range(10):
            x0 = np.array([random.uniform(min_val, max_val) for min_val, max_val in bounds])
            
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            
            if result.fun < best_acquisition_value:
                best_acquisition_value = result.fun
                best_x = result.x
        
        if best_x is None:
            best_x = np.array([random.uniform(min_val, max_val) for min_val, max_val in bounds])
        
        params = self._array_to_params(best_x)
        return self._ensure_bounds(params)
    
    def _acquisition(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        if self.acquisition_function == 'ei':
            f_best = np.max(self.y_observed)
            with np.errstate(divide='warn'):
                imp = mu - f_best - self.xi
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            return ei
        
        elif self.acquisition_function == 'ucb':
            return mu + self.kappa * sigma
        
        elif self.acquisition_function == 'poi':
            f_best = np.max(self.y_observed)
            with np.errstate(divide='warn'):
                Z = (mu - f_best - self.xi) / sigma
                poi = norm.cdf(Z)
                poi[sigma == 0.0] = 0.0
            return poi
        
        else:
            return mu + 2 * sigma
    
    def _compute_fitness(self, objectives: Dict[str, float]) -> float:
        return sum(objectives.values()) / len(objectives)
    
    def _check_convergence(self) -> bool:
        if len(self.optimization_history) < 15:
            return False
        
        recent_fitness = [entry['fitness'] for entry in self.optimization_history[-15:]]
        fitness_improvement = (recent_fitness[-1] - recent_fitness[0]) / (abs(recent_fitness[0]) + 1e-10)
        return fitness_improvement < 0.01


class MultiObjectiveOptimizer(BaseOptimizer):
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]], 
                 population_size: int = 100):
        super().__init__("NSGA-II", parameter_bounds)
        self.population_size = population_size
        
    def optimize(self, objective_function: Callable, max_evaluations: int, **kwargs) -> OptimizationResult:
        import time
        start_time = time.time()
        
        self.optimization_history = []
        
        population = self._initialize_population()
        evaluations = 0
        generation = 0
        
        objectives_history = []
        
        for individual in population:
            if evaluations >= max_evaluations:
                break
            
            result = objective_function(individual)
            objectives_list = list(result.objectives.values())
            objectives_history.append(objectives_list)
            
            self.optimization_history.append({
                'generation': generation,
                'individual': individual.copy(),
                'objectives': result.objectives.copy(),
                'objectives_list': objectives_list,
                'evaluation': evaluations
            })
            
            evaluations += 1
        
        while evaluations < max_evaluations:
            generation += 1
            
            fronts = self._fast_non_dominated_sort(objectives_history)
            
            new_population = []
            new_objectives = []
            
            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    for idx in front:
                        new_population.append(population[idx].copy())
                        new_objectives.append(objectives_history[idx])
                else:
                    remaining_slots = self.population_size - len(new_population)
                    if remaining_slots > 0:
                        front_objectives = [objectives_history[idx] for idx in front]
                        crowding_distances = self._calculate_crowding_distance(front_objectives)
                        sorted_indices = sorted(range(len(front)), 
                                              key=lambda i: crowding_distances[i], reverse=True)
                        
                        for i in range(remaining_slots):
                            idx = front[sorted_indices[i]]
                            new_population.append(population[idx].copy())
                            new_objectives.append(objectives_history[idx])
                    break
            
            offspring_population = []
            offspring_objectives = []
            
            while len(offspring_population) < self.population_size and evaluations < max_evaluations:
                parent1 = self._tournament_selection_mo(new_population, new_objectives)
                parent2 = self._tournament_selection_mo(new_population, new_objectives)
                
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                for child in [child1, child2]:
                    if len(offspring_population) < self.population_size and evaluations < max_evaluations:
                        child = self._ensure_bounds(child)
                        result = objective_function(child)
                        objectives_list = list(result.objectives.values())
                        
                        offspring_population.append(child)
                        offspring_objectives.append(objectives_list)
                        
                        self.optimization_history.append({
                            'generation': generation,
                            'individual': child.copy(),
                            'objectives': result.objectives.copy(),
                            'objectives_list': objectives_list,
                            'evaluation': evaluations
                        })
                        
                        evaluations += 1
            
            combined_population = new_population + offspring_population
            combined_objectives = new_objectives + offspring_objectives
            
            fronts = self._fast_non_dominated_sort(combined_objectives)
            
            population = []
            objectives_history = []
            
            for front in fronts:
                if len(population) + len(front) <= self.population_size:
                    for idx in front:
                        population.append(combined_population[idx].copy())
                        objectives_history.append(combined_objectives[idx])
                else:
                    remaining_slots = self.population_size - len(population)
                    if remaining_slots > 0:
                        front_objectives = [combined_objectives[idx] for idx in front]
                        crowding_distances = self._calculate_crowding_distance(front_objectives)
                        sorted_indices = sorted(range(len(front)), 
                                              key=lambda i: crowding_distances[i], reverse=True)
                        
                        for i in range(remaining_slots):
                            idx = front[sorted_indices[i]]
                            population.append(combined_population[idx].copy())
                            objectives_history.append(combined_objectives[idx])
                    break
            
            if generation % 10 == 0:
                self.logger.info(f"Generation {generation}, Population size: {len(population)}")
        
        end_time = time.time()
        
        pareto_front = self._get_pareto_front()
        
        if pareto_front:
            best_solution = pareto_front[0]['individual']
            best_objectives = pareto_front[0]['objectives']
        else:
            best_solution = {}
            best_objectives = {}
        
        return OptimizationResult(
            best_parameters=best_solution,
            best_objectives=best_objectives,
            optimization_history=self.optimization_history,
            total_evaluations=evaluations,
            convergence_achieved=True,
            total_time=end_time - start_time,
            algorithm_name=self.name,
            metadata={
                'pareto_front_size': len(pareto_front),
                'final_generation': generation,
                'population_size': self.population_size
            }
        )
    
    def _initialize_population(self) -> List[Dict[str, float]]:
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, (min_val, max_val) in self.parameter_bounds.items():
                individual[param] = random.uniform(min_val, max_val)
            population.append(individual)
        return population
    
    def _fast_non_dominated_sort(self, objectives_list: List[List[float]]) -> List[List[int]]:
        n = len(objectives_list)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objectives_list[i], objectives_list[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives_list[j], objectives_list[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        front_idx = 0
        while len(fronts[front_idx]) > 0:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            front_idx += 1
        
        return fronts
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        at_least_one_better = False
        for i in range(len(obj1)):
            if obj1[i] < obj2[i]:
                return False
            elif obj1[i] > obj2[i]:
                at_least_one_better = True
        return at_least_one_better
    
    def _calculate_crowding_distance(self, objectives_list: List[List[float]]) -> List[float]:
        n = len(objectives_list)
        if n == 0:
            return []
        
        distances = [0.0] * n
        n_objectives = len(objectives_list[0])
        
        for obj_idx in range(n_objectives):
            sorted_indices = sorted(range(n), key=lambda i: objectives_list[i][obj_idx])
            
            obj_values = [objectives_list[i][obj_idx] for i in sorted_indices]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range == 0:
                continue
            
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            for i in range(1, n - 1):
                if distances[sorted_indices[i]] != float('inf'):
                    distances[sorted_indices[i]] += (obj_values[i + 1] - obj_values[i - 1]) / obj_range
        
        return distances
    
    def _tournament_selection_mo(self, population: List[Dict[str, float]], 
                                objectives: List[List[float]]) -> Dict[str, float]:
        idx1, idx2 = random.sample(range(len(population)), 2)
        
        if self._dominates(objectives[idx1], objectives[idx2]):
            return population[idx1].copy()
        elif self._dominates(objectives[idx2], objectives[idx1]):
            return population[idx2].copy()
        else:
            return population[random.choice([idx1, idx2])].copy()
    
    def _crossover(self, parent1: Dict[str, float], 
                  parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        child1, child2 = parent1.copy(), parent2.copy()
        
        for param in parent1.keys():
            if random.random() < 0.5:
                alpha = random.random()
                child1[param] = alpha * parent1[param] + (1 - alpha) * parent2[param]
                child2[param] = alpha * parent2[param] + (1 - alpha) * parent1[param]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        mutated = individual.copy()
        
        for param, value in mutated.items():
            if random.random() < 0.1:
                if param in self.parameter_bounds:
                    min_val, max_val = self.parameter_bounds[param]
                    mutation_strength = 0.1 * (max_val - min_val)
                    mutated[param] = value + random.gauss(0, mutation_strength)
        
        return mutated
    
    def _get_pareto_front(self) -> List[Dict[str, Any]]:
        if not self.optimization_history:
            return []
        
        objectives_lists = [entry['objectives_list'] for entry in self.optimization_history]
        fronts = self._fast_non_dominated_sort(objectives_lists)
        
        if not fronts or not fronts[0]:
            return []
        
        pareto_front = []
        for idx in fronts[0]:
            pareto_front.append(self.optimization_history[idx])
        
        return pareto_front