from typing import List

import numpy as np
from finchge.algorithm import BaseAlgorithm
import itertools

def dominates(ind1, ind2, maximize_flags: list[bool]) -> bool:
    """
    Determines whether ind1 dominates ind2 in a multi-objective context.

    An individual dominates another if it is no worse in all objectives and
    strictly better in at least one, according to the specified optimization directions.

    Args:
        ind1: First individual with a 'fitness' attribute (scalar or list).
        ind2: Second individual with a 'fitness' attribute (scalar or list).
        maximize_flags (list of bool): List indicating whether each objective should
            be maximized (True) or minimized (False).

    Returns:
        bool: True if ind1 dominates ind2, False otherwise.
    """
    better_in_any = False

    # Get objectives from fitness values
    f1 = ind1.fitness if isinstance(ind1.fitness, list) else [ind1.fitness]
    f2 = ind2.fitness if isinstance(ind2.fitness, list) else [ind2.fitness]

    if len(f1) != len(maximize_flags) or len(f2) != len(maximize_flags):
        raise ValueError("Length of fitness values and maximize_flags must match.")

    for i, (val1, val2) in enumerate(zip(f1, f2)):
        # Compare based on optimization direction
        if maximize_flags[i]:
            if val1 < val2:  # ind1 is worse
                return False
            elif val1 > val2:  # ind1 is better
                better_in_any = True
        else:
            if val1 > val2:  # ind1 is worse
                return False
            elif val1 < val2:  # ind1 is better
                better_in_any = True

    return better_in_any


def fast_non_dominated_sort(individuals, maximize_flags: list[bool]) -> List[List]:
    """
    Implement NSGA-II fast non-dominated sorting
    Returns list of fronts, where each front is a list of indices
    """

    if not individuals:
        return [[]]  # Return empty front if population is empty


    fronts = [[]]  # Initialize with empty first front
    dominated_sets = [[] for _ in range(len(individuals))]
    domination_counts = [0] * len(individuals)

    # Calculate domination for each individual
    for i, p in enumerate(individuals):
        for j, q in enumerate(individuals):
            if i != j:
                if dominates(p, q, maximize_flags):
                    dominated_sets[i].append(j)
                elif dominates(q, p, maximize_flags):
                    domination_counts[i] += 1

        # Add to first front if not dominated
        if domination_counts[i] == 0:
            p.rank = 0
            fronts[0].append(i)

    # If first front is empty, something is wrong with domination comparison
    if not fronts[0]:
        print("Warning: No individuals in first front. Check domination comparison.")
        # Add all individuals to first front as fallback
        fronts[0] = list(range(len(individuals)))
        return fronts

    # Generate remaining fronts
    i = 0
    while i < len(fronts):
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in dominated_sets[p_idx]:
                domination_counts[q_idx] -= 1
                if domination_counts[q_idx] == 0:
                    individuals[q_idx].rank = i + 1
                    next_front.append(q_idx)
        if next_front:
            fronts.append(next_front)
        i += 1

    return fronts


def calculate_crowding_distance(individuals: List, front: List, maximize_flags: list[bool]) -> None:
        """Calculate crowding distance for individuals in a front"""
        if len(front) <= 2:
            for idx in front:
                individuals[idx].crowding_distance = float('inf')
            return

        front_size = len(front)

        # Initialize distances
        for idx in front:
            individuals[idx].crowding_distance = 0

        # Number of objectives
        num_objectives = len(individuals[front[0]].fitness) if isinstance(
            individuals[front[0]].fitness, list) else 1

        # Calculate crowding distance for each objective
        for m in range(num_objectives):
            # Sort front by mth objective
            front = sorted(front,
                           key=lambda idx: individuals[idx].fitness[m] if isinstance(
                               individuals[idx].fitness, list) else individuals[idx].fitness)

            # Set boundary points to infinity
            individuals[front[0]].crowding_distance = float('inf')
            individuals[front[-1]].crowding_distance = float('inf')

            # Get fitness values for this objective
            if isinstance(individuals[0].fitness, list):
                f_values = [individuals[i].fitness[m] for i in front]
            else:
                f_values = [individuals[i].fitness for i in front]

            # Calculate crowding distances
            f_range = f_values[-1] - f_values[0]
            if f_range > 0:
                for i in range(1, front_size - 1):
                    distance = (f_values[i + 1] - f_values[i - 1]) / f_range
                    individuals[front[i]].crowding_distance += distance

class NSGA2(BaseAlgorithm):
    def __init__(self, selection, crossover, mutation, replacement, elite_size: int):
        """
        NSGA2  class.

        Args:
            selection: Selection strategy instance or function.
            crossover: Crossover strategy instance or function.
            mutation: Mutation strategy instance or function.
            replacement: Replacement strategy instance or function.
        """
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement
        self.elite_size = elite_size


    def evolve_one_generation(self, population) -> 'Population':
        """
        Perform one generation of evolution on the given population.

        Args:
            population (Population): The population to evolve.

        Returns:
            Population: The evolved population.
        """
        valid_individuals = [ind for ind in population.individuals if not ind.invalid]
        if len(valid_individuals) < 2:
            raise Exception(f"Not enough valid individuals. Valid count: {len(valid_individuals)}")

        # Selection
        selected_pop = self.selection.select(
            population.config['population_size'],
            valid_individuals
        )

        # Crossover
        offspring_genome = []
        while len(offspring_genome) < population.config['population_size']:
            parent_indices = np.random.randint(0, len(selected_pop), size=2)
            parents = [selected_pop[i] for i in parent_indices]
            offsprings = parents[0].cross_with(parents[1], crossover_strategy=self.crossover)
            offspring_genome.extend(offsprings)

        # Create new population from offspring
        new_population = population.__class__(
            population.config,
            fitness_evaluator=population.fitness_evaluator,
            grammar=population.grammar,
            genome=offspring_genome,
            cache_manager=population.cache_manager
        )

        # Mutation
        for individual in new_population.individuals:
            individual.mutate(mutation_strategy=self.mutation)

        new_population.eval()
        self.sort_population(new_population)

        # Replacement
        new_population.individuals = self.replacement.replace(
            new_population=new_population.individuals,
            old_population=population.individuals,
            elite_size=self.elite_size,
            population_size=population.config['population_size'],
            maximize_flags=population.fitness_evaluator.get_maximize_flags()
        )

        new_population.eval()
        return new_population

    def sort_population(self, population) -> None:
        """Sort population using NSGA-II criteria"""
        maximize_flags = population.fitness_evaluator.get_maximize_flags()
        fronts = fast_non_dominated_sort(population.individuals, maximize_flags)
        # Calculate crowding distance for each front
        for front in fronts:
            calculate_crowding_distance(population.individuals, front, maximize_flags)
        # Sort population based on rank and crowding distance
        population.individuals.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

    def get_best(self, population) -> List:
        """Return the Pareto front (individuals with rank 0)"""
        fronts = fast_non_dominated_sort(population.individuals, population.fitness_evaluator.get_maximize_flags())
        return [population.individuals[idx] for idx in fronts[0]]


class NSGA3(BaseAlgorithm):
    def __init__(self, selection, crossover, mutation, replacement, elite_size: int, num_divisions: int = 12):
        """
        NSGA3  class.

        Args:
            selection: Selection strategy instance or function.
            crossover: Crossover strategy instance or function.
            mutation: Mutation strategy instance or function.
            replacement: Replacement strategy instance or function.
        """
        self.num_divisions = num_divisions
        self.reference_points = None
        self.ideal_point = None
        self.worst_point = None
        self.epsilon = 1e-10

        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement
        self.elite_size = elite_size


    def evolve_one_generation(self, population) -> 'Population':
        """
        Perform one generation of evolution on the given population.

        Args:
            population (Population): The population to evolve.

        Returns:
            Population: The evolved population.
        """
        valid_individuals = [ind for ind in population.individuals if not ind.invalid]
        if len(valid_individuals) < 2:
            raise Exception(f"Not enough valid individuals. Valid count: {len(valid_individuals)}")

        # Selection
        selected_pop = self.selection.select(
            population.config['population_size'],
            valid_individuals
        )

        # Crossover
        offspring_genome = []
        while len(offspring_genome) < population.config['population_size']:
            parent_indices = np.random.randint(0, len(selected_pop), size=2)
            parents = [selected_pop[i] for i in parent_indices]
            offsprings = parents[0].cross_with(parents[1], crossover_strategy=self.crossover)
            offspring_genome.extend(offsprings)

        # Create new population from offspring
        new_population = population.__class__(
            population.config,
            fitness_evaluator=population.fitness_evaluator,
            grammar=population.grammar,
            genome=offspring_genome,
            cache_manager=population.cache_manager
        )

        # Mutation
        for individual in new_population.individuals:
            individual.mutate(mutation_strategy=self.mutation)

        new_population.eval()
        maximize_flags = [fn.maximize for fn in population.fitness_evaluator.get_fitness_functions()]
        self.sort_population(new_population, maximize_flags=maximize_flags)

        # Replacement
        new_population.individuals = self.replacement.replace(
            new_population=new_population.individuals,
            old_population=population.individuals,
            elite_size=self.elite_size,
            population_size=population.config['population_size'],
            maximize_flags=maximize_flags
        )

        new_population.eval()
        return new_population

    def sort_population(self, population, maximize_flags) -> None:
        """Sort population using NSGA-III criteria"""
        fronts = fast_non_dominated_sort(population.individuals, maximize_flags)

        #   Calculating crowding distance for selection functions (even though NSGA-III primarily uses reference points)
        for front in fronts:
            calculate_crowding_distance(population.individuals, front, maximize_flags)

        num_objectives = len(population.individuals[0].fitness) if isinstance(population.individuals[0].fitness, list) else 1
        if self.reference_points is None:
            self.reference_points = self.generate_reference_points(num_objectives, self.num_divisions)

        population.individuals.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

    def get_best(self, population, maximize_flags) -> List:
        """Return the Pareto front"""
        fronts = fast_non_dominated_sort(population.individuals, maximize_flags)
        return [population.individuals[idx] for idx in fronts[0]]

    def generate_reference_points(self, num_objectives, num_divisions):
        """
        Generate structured reference points

        Args:
            num_objectives (int): Number of objectives.
            num_divisions (int): Number of divisions for each objective.

        Returns:
            np.ndarray: Array of reference points.
        """

        if num_objectives == 2 and num_divisions < 2:
            raise ValueError("For 2 objectives, num_divisions must be at least 2 to avoid degenerate reference points.")

        def recursive_combinations(n, k):
            """Helper function to generate integer combinations."""
            for c in itertools.combinations_with_replacement(range(n + 1), k - 1):
                yield [c[0]] + \
                    [c[i] - c[i - 1] for i in range(1, len(c))] + \
                    [n - c[-1]]

        reference_points = []
        for partition in recursive_combinations(num_divisions, num_objectives):
            reference_points.append(np.array(partition) / num_divisions)

        return np.array(reference_points)

    def get_objective_values(self, individuals, maximize_flags, normalized=False):
        """Extract objective values from population"""
        if isinstance(individuals[0].fitness, list):
            objectives = np.array([ind.fitness for ind in individuals])
        else:
            objectives = np.array([[ind.fitness] for ind in individuals])

        if normalized:
            return self.normalize_objectives(objectives, maximize_flags=maximize_flags)
        return objectives

    def normalize_objectives(self, objectives, maximize_flags):
        """Normalize objective values considering optimization direction"""
        if self.ideal_point is None or self.worst_point is None:
            self.ideal_point = np.min(objectives, axis=0)
            self.worst_point = np.max(objectives, axis=0)

        # Adjust normalization based on optimization direction
        normalized = np.zeros_like(objectives, dtype=float)
        for i in range(objectives.shape[1]):
            if maximize_flags[i]:
                # For maximization, reverse the normalization
                normalized[:, i] = (self.worst_point[i] - objectives[:, i]) / \
                                   (self.worst_point[i] - self.ideal_point[i] + self.epsilon)
            else:
                # For minimization, normal normalization
                normalized[:, i] = (objectives[:, i] - self.ideal_point[i]) / \
                                   (self.worst_point[i] - self.ideal_point[i] + self.epsilon)

        return normalized

    def associate_to_reference_points(self, normalized_objectives, reference_points):
        """Associate solutions to reference points"""
        # Calculate perpendicular distances
        distances = []
        for objective in normalized_objectives:
            d = []
            for ref_point in reference_points:
                # Project objective onto reference line
                proj = np.dot(objective, ref_point) / np.dot(ref_point, ref_point)
                # Calculate perpendicular distance
                d.append(np.linalg.norm(objective - proj * ref_point))
            distances.append(d)
        return np.array(distances)

    def select_from_front(self, front_indices, individuals, maximize_flags, reference_points):
        """Select individuals from front using reference points"""
        if len(front_indices) == 0:
            return []

        front_individuals = [individuals[i] for i in front_indices]
        front_objectives = self.get_objective_values(front_individuals, maximize_flags=maximize_flags, normalized=True)

        distances = self.associate_to_reference_points(front_objectives, reference_points)

        selected = []
        remaining_points = list(range(len(reference_points)))

        while len(selected) < len(front_indices) and remaining_points:
            min_dist = float('inf')
            best_idx_in_front = None
            best_point_idx = None

            for i, point_idx in enumerate(front_indices):
                if point_idx in selected:
                    continue

                for ref_idx in remaining_points:
                    dist = distances[i][ref_idx]
                    if dist < min_dist:
                        min_dist = dist
                        best_idx_in_front = i  # this is index in front_individuals
                        best_point_idx = ref_idx

            if best_idx_in_front is not None:
                selected.append(front_indices[best_idx_in_front])
                remaining_points.remove(best_point_idx)
            else:
                break  # No more selections possible

            return selected

