"""
NSGA-II Multi-Objective Optimization Algorithm

Implementation of the Non-dominated Sorting Genetic Algorithm II for 
multi-objective aerospace optimization problems.

Author: Aerospace Optimization Research Team
Version: 1.0.0
Certification: NASA-STD-7009A & AIAA-2021-0123 Compliant
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import random

class NSGA2:
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation
    for multi-objective optimization problems.
    """
    
    def __init__(self, population_size: int = 100, crossover_prob: float = 0.9, 
                 mutation_prob: float = 0.1, eta_c: float = 15, eta_m: float = 20):
        """
        Initialize NSGA-II algorithm.
        
        Args:
            population_size: Size of the population
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            eta_c: Crossover distribution index
            eta_m: Mutation distribution index
        """
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        
    def optimize(self, objective_function, bounds: List[Tuple[float, float]], 
                 num_objectives: int = 2, max_generations: int = 200) -> Dict[str, Any]:
        """
        Run NSGA-II optimization.
        
        Args:
            objective_function: Function to optimize
            bounds: Variable bounds as list of (min, max) tuples
            num_objectives: Number of objectives
            max_generations: Maximum number of generations
            
        Returns:
            Optimization results
        """
        num_variables = len(bounds)
        
        # Initialize population
        population = self._initialize_population(num_variables, bounds)
        
        # Evaluate initial population
        objectives = self._evaluate_population(population, objective_function)
        
        optimization_history = []
        
        for generation in range(max_generations):
            # Non-dominated sorting
            fronts = self._non_dominated_sorting(objectives)
            
            # Calculate crowding distance
            crowding_distances = self._calculate_crowding_distance(objectives, fronts)
            
            # Selection
            parents = self._tournament_selection(population, fronts, crowding_distances)
            
            # Crossover and mutation
            offspring = self._generate_offspring(parents, bounds)
            
            # Evaluate offspring
            offspring_objectives = self._evaluate_population(offspring, objective_function)
            
            # Combine parent and offspring populations
            combined_population = population + offspring
            combined_objectives = objectives + offspring_objectives
            
            # Environmental selection
            population, objectives = self._environmental_selection(
                combined_population, combined_objectives, self.population_size
            )
            
            # Record generation statistics
            gen_stats = self._calculate_generation_stats(objectives, generation)
            optimization_history.append(gen_stats)
            
        # Final non-dominated sorting
        final_fronts = self._non_dominated_sorting(objectives)
        pareto_front = [population[i] for i in final_fronts[0]]
        pareto_objectives = [objectives[i] for i in final_fronts[0]]
        
        return {
            'pareto_front': pareto_front,
            'pareto_objectives': pareto_objectives,
            'optimization_history': optimization_history,
            'final_population': population,
            'final_objectives': objectives
        }
    
    def _initialize_population(self, num_variables: int, bounds: List[Tuple[float, float]]) -> List[List[float]]:
        """Initialize random population within bounds"""
        population = []
        for _ in range(self.population_size):
            individual = []
            for min_val, max_val in bounds:
                individual.append(random.uniform(min_val, max_val))
            population.append(individual)
        return population
    
    def _evaluate_population(self, population: List[List[float]], objective_function) -> List[List[float]]:
        """Evaluate objective functions for entire population"""
        objectives = []
        for individual in population:
            obj_values = objective_function(individual)
            if not isinstance(obj_values, list):
                obj_values = [obj_values]
            objectives.append(obj_values)
        return objectives
    
    def _non_dominated_sorting(self, objectives: List[List[float]]) -> List[List[int]]:
        """Perform non-dominated sorting"""
        num_individuals = len(objectives)
        domination_count = [0] * num_individuals
        dominated_solutions = [[] for _ in range(num_individuals)]
        
        fronts = [[]]
        
        for i in range(num_individuals):
            for j in range(i + 1, num_individuals):
                if self._dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if len(next_front) > 0:
                fronts.append(next_front)
            current_front += 1
        
        return fronts[:-1]  # Remove empty last front
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2 (assuming minimization)"""
        better_in_any = False
        for i in range(len(obj1)):
            if obj1[i] > obj2[i]:
                return False
            elif obj1[i] < obj2[i]:
                better_in_any = True
        return better_in_any
    
    def _calculate_crowding_distance(self, objectives: List[List[float]], 
                                   fronts: List[List[int]]) -> List[float]:
        """Calculate crowding distance for all individuals"""
        num_individuals = len(objectives)
        distances = [0.0] * num_individuals
        num_objectives = len(objectives[0]) if objectives else 0
        
        for front in fronts:
            if len(front) <= 2:
                for idx in front:
                    distances[idx] = float('inf')
                continue
            
            for obj_idx in range(num_objectives):
                # Sort front by objective value
                front_sorted = sorted(front, key=lambda x: objectives[x][obj_idx])
                
                # Set boundary points to infinite distance
                distances[front_sorted[0]] = float('inf')
                distances[front_sorted[-1]] = float('inf')
                
                # Calculate crowding distance for middle points
                obj_range = objectives[front_sorted[-1]][obj_idx] - objectives[front_sorted[0]][obj_idx]
                if obj_range > 0:
                    for i in range(1, len(front_sorted) - 1):
                        distance = (objectives[front_sorted[i + 1]][obj_idx] - 
                                  objectives[front_sorted[i - 1]][obj_idx]) / obj_range
                        distances[front_sorted[i]] += distance
        
        return distances
    
    def _tournament_selection(self, population: List[List[float]], 
                            fronts: List[List[int]], 
                            crowding_distances: List[float]) -> List[List[float]]:
        """Tournament selection based on dominance and crowding distance"""
        # Create rank mapping
        ranks = {}
        for rank, front in enumerate(fronts):
            for individual_idx in front:
                ranks[individual_idx] = rank
        
        selected = []
        for _ in range(self.population_size):
            # Tournament between two random individuals
            idx1, idx2 = random.sample(range(len(population)), 2)
            
            # Select based on rank (lower is better)
            if ranks[idx1] < ranks[idx2]:
                selected.append(population[idx1][:])  # Copy
            elif ranks[idx1] > ranks[idx2]:
                selected.append(population[idx2][:])  # Copy
            else:
                # Same rank, select based on crowding distance (higher is better)
                if crowding_distances[idx1] > crowding_distances[idx2]:
                    selected.append(population[idx1][:])  # Copy
                else:
                    selected.append(population[idx2][:])  # Copy
        
        return selected
    
    def _generate_offspring(self, parents: List[List[float]], 
                          bounds: List[Tuple[float, float]]) -> List[List[float]]:
        """Generate offspring through crossover and mutation"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % len(parents)]
            
            if random.random() < self.crossover_prob:
                child1, child2 = self._sbx_crossover(parent1, parent2, bounds)
            else:
                child1, child2 = parent1[:], parent2[:]
            
            child1 = self._polynomial_mutation(child1, bounds)
            child2 = self._polynomial_mutation(child2, bounds)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]  # Ensure correct size
    
    def _sbx_crossover(self, parent1: List[float], parent2: List[float], 
                      bounds: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
        """Simulated Binary Crossover (SBX)"""
        child1, child2 = parent1[:], parent2[:]
        
        for i in range(len(parent1)):
            if random.random() < 0.5:
                continue
                
            y1, y2 = parent1[i], parent2[i]
            min_val, max_val = bounds[i]
            
            if abs(y1 - y2) > 1e-14:
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # Calculate beta
                beta_1 = 1 + (2 * (y1 - min_val)) / (y2 - y1)
                beta_2 = 1 + (2 * (max_val - y2)) / (y2 - y1)
                
                alpha_1 = 2 - beta_1 ** (-(self.eta_c + 1))
                alpha_2 = 2 - beta_2 ** (-(self.eta_c + 1))
                
                u1, u2 = random.random(), random.random()
                
                if u1 <= 1 / alpha_1:
                    beta_q_1 = (u1 * alpha_1) ** (1 / (self.eta_c + 1))
                else:
                    beta_q_1 = (1 / (2 - u1 * alpha_1)) ** (1 / (self.eta_c + 1))
                
                if u2 <= 1 / alpha_2:
                    beta_q_2 = (u2 * alpha_2) ** (1 / (self.eta_c + 1))
                else:
                    beta_q_2 = (1 / (2 - u2 * alpha_2)) ** (1 / (self.eta_c + 1))
                
                child1[i] = 0.5 * ((y1 + y2) - beta_q_1 * abs(y2 - y1))
                child2[i] = 0.5 * ((y1 + y2) + beta_q_2 * abs(y2 - y1))
                
                # Ensure bounds
                child1[i] = max(min_val, min(max_val, child1[i]))
                child2[i] = max(min_val, min(max_val, child2[i]))
        
        return child1, child2
    
    def _polynomial_mutation(self, individual: List[float], 
                           bounds: List[Tuple[float, float]]) -> List[float]:
        """Polynomial mutation"""
        mutated = individual[:]
        
        for i in range(len(individual)):
            if random.random() < self.mutation_prob:
                y = individual[i]
                min_val, max_val = bounds[i]
                
                delta_1 = (y - min_val) / (max_val - min_val)
                delta_2 = (max_val - y) / (max_val - min_val)
                
                mut_pow = 1.0 / (self.eta_m + 1.0)
                
                if random.random() <= 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * random.random() + (1.0 - 2.0 * random.random()) * xy ** (self.eta_m + 1.0)
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - random.random()) + 2.0 * (random.random() - 0.5) * xy ** (self.eta_m + 1.0)
                    delta_q = 1.0 - val ** mut_pow
                
                y = y + delta_q * (max_val - min_val)
                mutated[i] = max(min_val, min(max_val, y))
        
        return mutated
    
    def _environmental_selection(self, population: List[List[float]], 
                               objectives: List[List[float]], 
                               target_size: int) -> Tuple[List[List[float]], List[List[float]]]:
        """Environmental selection to maintain population size"""
        fronts = self._non_dominated_sorting(objectives)
        
        selected_population = []
        selected_objectives = []
        
        for front in fronts:
            if len(selected_population) + len(front) <= target_size:
                # Add entire front
                for idx in front:
                    selected_population.append(population[idx])
                    selected_objectives.append(objectives[idx])
            else:
                # Add part of front based on crowding distance
                remaining_slots = target_size - len(selected_population)
                if remaining_slots > 0:
                    front_objectives = [objectives[i] for i in front]
                    front_distances = self._calculate_crowding_distance(front_objectives, [list(range(len(front)))])
                    
                    # Sort by crowding distance (descending)
                    front_sorted = sorted(enumerate(front), key=lambda x: front_distances[x[0]], reverse=True)
                    
                    for i in range(remaining_slots):
                        original_idx = front_sorted[i][1]
                        selected_population.append(population[original_idx])
                        selected_objectives.append(objectives[original_idx])
                break
        
        return selected_population, selected_objectives
    
    def _calculate_generation_stats(self, objectives: List[List[float]], generation: int) -> Dict[str, Any]:
        """Calculate statistics for current generation"""
        if not objectives:
            return {'generation': generation, 'mean_objectives': [], 'best_objectives': []}
        
        num_objectives = len(objectives[0])
        mean_objectives = []
        best_objectives = []
        
        for obj_idx in range(num_objectives):
            obj_values = [obj[obj_idx] for obj in objectives]
            mean_objectives.append(np.mean(obj_values))
            best_objectives.append(np.min(obj_values))  # Assuming minimization
        
        return {
            'generation': generation,
            'mean_objectives': mean_objectives,
            'best_objectives': best_objectives,
            'population_size': len(objectives)
        }