from abc import ABC, abstractmethod
from random import random
from typing import List
from finchge.algorithm.nsga import fast_non_dominated_sort, calculate_crowding_distance


class GEReplacementStrategy(ABC):
    """
    Base class for replacement strategy.

    Args:
        max_best (bool): If True, higher fitness is better. If False, lower is better.
    """
    def __init__(self, max_best: bool):
        if not isinstance(max_best, bool):
            raise TypeError("max_best must be a boolean")

        self.max_best = max_best

    @abstractmethod
    def replace(self, new_population: list, old_population: list, elite_size: int, population_size: int):
        pass


class GenerationalReplacement(GEReplacementStrategy):
    def __init__(self, max_best: bool):
        super().__init__(max_best=max_best)

    def replace(self, new_population: list, old_population: list, elite_size: int, population_size: int):
        """
        Replaces old population with new population, preserving elite individuals.

        Args:
            new_population: List of new individuals
            old_population: List of current individuals
            elite_size: Number of elite individuals to preserve
            population_size: Target size of the population

        Returns:
            list: New population with elites
        """
        old_population.sort(key=lambda ind: ind.fitness, reverse=self.max_best)
        elites = old_population[:elite_size]
        combined_population = new_population + elites
        combined_population.sort(key=lambda ind: ind.fitness, reverse=self.max_best)
        return combined_population[:population_size]


class SteadyStateReplacement(GEReplacementStrategy):
    def __init__(self, max_best: bool):
        super().__init__(max_best=max_best)

    def replace(self, new_population: list, old_population: list, elite_size: int, population_size: int):
        """
        Replaces worst individuals in old population with new individuals.
        Preserves elite_size best individuals.
        """
        old_population.sort(key=lambda ind: ind.fitness, reverse=self.max_best)
        preserved = old_population[:elite_size] # preserve best individuals (elites)

        new_population.sort(key=lambda ind: ind.fitness, reverse=self.max_best)
        replacements = new_population[:population_size - elite_size] # take elites
        return preserved + replacements


class RandomReplacement(GEReplacementStrategy):
    """
    Randomly replaces individuals from old population, but preserves elites.
    """
    def __init__(self, max_best: bool):
        super().__init__(max_best=max_best)

    def replace(self, new_population: list, old_population: list, elite_size: int, population_size: int):
        """
        Randomly replaces individuals from old population, but preserves elites.
        """
        old_population.sort(key=lambda ind: ind.fitness, reverse=self.max_best)
        elites = old_population[:elite_size]

        eligible = old_population[elite_size:] + new_population
        random_selection = random.sample(eligible, population_size - elite_size)

        return elites + random_selection


class CrowdingReplacement(GEReplacementStrategy):
    """
    CrowdingReplacement
    """
    def __init__(self, max_best: bool, distance_metric=None):
        super().__init__(max_best=max_best)
        self.distance_metric = distance_metric or self._default_distance

    def _default_distance(self, ind1, ind2):
        """Default distance metric based on genotype length"""
        return abs(len(ind1.genotype) - len(ind2.genotype))

    def replace(self, new_population: list, old_population: list, elite_size: int, population_size: int):
        """
        Replaces individuals with most similar ones from new population if better.
        Preserves elites.
        """
        old_population.sort(key=lambda ind: ind.fitness, reverse=self.max_best)
        result = old_population[:elite_size]
        remaining_old = old_population[elite_size:]

        #  replace most similar old one if better
        for new_ind in new_population:
            if len(result) >= population_size:
                break

            if not remaining_old:
                result.append(new_ind)
                continue

            most_similar = min(remaining_old,
                               key=lambda x: self.distance_metric(x, new_ind))

            # Replace if better
            if ((self.max_best and new_ind.fitness > most_similar.fitness) or
                    (not self.max_best and new_ind.fitness < most_similar.fitness)):
                result.append(new_ind)
                remaining_old.remove(most_similar)
            else:
                result.append(most_similar)
                remaining_old.remove(most_similar)

        return result

"""

REPLACEMENT STRATEGY FOR MULTI-OBJECTIVE OPTIMIZATION

"""

class MODiverseReplacement:
    def replace(self, new_population: List, old_population: List,
                elite_size: int, population_size: int, maximize_flags) -> List:
        """
        Replacement strategy that preserves elite solutions and maintains diversity.
        """

        combined = old_population + new_population
        fronts = fast_non_dominated_sort(individuals=combined, maximize_flags=maximize_flags)
        # Sort by rank (lower rank is better)
        combined.sort(key=lambda x: (x.rank if x.rank is not None else float('inf')))
        selected = []
        # Step 1: Preserve the top elite solutions (best solutions from the old population)
        elite_individuals = sorted(old_population, key=lambda x: x.rank)[:elite_size]
        selected.extend(elite_individuals)

        current_rank = 0
        # Step 2: Fill remaining population using Pareto fronts and crowding distance
        while len(selected) < population_size:
            current_front = [ind for ind in combined if ind.rank == current_rank and ind not in selected]

            if not current_front:
                break

            if len(selected) + len(current_front) <= population_size:
                selected.extend(current_front)
            else:
                # Step 3: Use crowding distance to pick the most diverse solutions
                needed = population_size - len(selected)
                calculate_crowding_distance(current_front, range(len(current_front)), maximize_flags)

                # Sort front by crowding distance (higher is better)
                current_front.sort(key=lambda x: x.crowding_distance, reverse=True)
                selected.extend(current_front[:needed])
                break

            current_rank += 1

        return selected