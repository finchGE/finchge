import random
from typing import List
import numpy as np
from abc import ABC, abstractmethod
import warnings


class GESelectionStrategy(ABC):
    def __init__(self, max_best: bool):
        if max_best is None:
            raise ValueError("max_best parameter is required")
        self.max_best = max_best

    @abstractmethod
    def select(self, population_size: int, population: list) -> list:
        pass


class TournamentSelection(GESelectionStrategy):
    """
    Selection strategy that chooses individuals using tournament selection.

    In tournament selection, a fixed number of individuals are randomly chosen from
    the population, and the best among them (based on fitness) is selected. This is
    repeated until the desired number of individuals is selected. The selection pressure
    can be controlled by adjusting the tournament size.

    Parameters
    ----------
    max_best : bool
        Whether higher fitness values are better (True for maximization, False for minimization).
    tournament_size : int, optional
        The number of individuals competing in each tournament. Must be >= 2. Default is 3.
    """
    def __init__(self, max_best: bool, tournament_size: int = 3):
        super().__init__(max_best=max_best)
        self.tournament_size = tournament_size

    def select(self, population_size: int, population: list) -> list:
        """
        Select a subset of individuals using tournament selection.

        Repeatedly selects a group of individuals (a "tournament") at random from the
        population, and chooses the best among them based on fitness. This process
        continues until the desired number of individuals is selected. Whether the
        best individual is determined by maximizing or minimizing fitness depends on
        the `max_best` setting.

        Parameters
        ----------
        population_size : int
            The number of individuals to select for the next generation.
        population : list
            A list of individuals to select from. Each must have a `fitness` attribute,
            which is expected to be a list containing a single float value (e.g., [fitness]).

        Returns
        -------
        list
            A list of selected individuals based on tournament outcomes.

        Raises
        ------
        ValueError
            If the population size is smaller than the tournament size.
        """
        if len(population) < self.tournament_size:
            raise ValueError(f"Population size ({len(population)}) must be >= tournament size ({self.tournament_size})")

        selected_pop = []
        while len(selected_pop) < population_size:
            participants = random.sample(population, self.tournament_size)
            winner = max(participants, key=lambda ind: ind.fitness) if self.max_best \
                else min(participants, key=lambda ind: ind.fitness)
            selected_pop.append(winner)
        return selected_pop


class RouletteWheelSelection(GESelectionStrategy):
    """
    Selection strategy that selects individuals based on roulette wheel (fitness proportionate) selection.

    Roulette wheel selection assigns selection probabilities to individuals based on their
    fitness values. Individuals with higher fitness values are more likely to be selected,
    but all individuals have a chance to be chosen. The selection process involves a proportional
    allocation of fitness values, where the "wheel" is spun to randomly select individuals based
    on their relative fitness.

    If the fitness values contain negative numbers, they are shifted to ensure all fitness values
    are non-negative. If the total weight of the fitness values becomes zero, uniform selection
    is used as a fallback.

    Parameters
    ----------
    max_best : bool
        Whether higher fitness values are better (True for maximization, False for minimization).
    """
    def select(self, population_size: int, population: list) -> list:
        """
        Select a subset of individuals using roulette wheel selection.

        This method performs fitness-proportionate selection, where individuals are assigned
        selection probabilities based on their fitness values. The fitness values are shifted
        to ensure they are non-negative, and if necessary, inverted for minimization problems.
        If the total weight of the shifted fitness values is zero or negative, the method falls
        back to uniform random selection.

        Parameters
        ----------
        population_size : int
            The number of individuals to select for the next generation.
        population : list
            A list of individuals to select from. Each individual must have a `fitness` attribute,
            which is expected to be a list containing a single float value (e.g., [fitness]).

        Returns
        -------
        list
            A list of selected individuals based on roulette wheel selection probabilities.

        Warnings
        --------
        If the total fitness values are zero or negative, a warning will be issued, and uniform
        random selection will be used.
        """

        # Since we are using fitness as a list, just extracting the fitness value beforehand.
        raw_fitness = [ind.fitness[0] for ind in population]
        min_fitness = min(raw_fitness)
        if min_fitness < 0:
            shifted_fitness = [fit - min_fitness + 1e-10 for fit in raw_fitness]
        else:
            shifted_fitness = [fit + 1e-10 for fit in raw_fitness]
        if not self.max_best:
            max_fitness = max(shifted_fitness)
            shifted_fitness = [max_fitness - fit for fit in shifted_fitness]

        total_weight = sum(shifted_fitness)
        if total_weight <= 0:
            warnings.warn("Roulette selection fell back to uniform due to zero total weights.")
            return random.choices(population, k=population_size)

        return random.choices(population, weights=shifted_fitness, k=population_size)


class RankSelection(GESelectionStrategy):
    """
    Selection strategy that selects individuals based on their rank in the population.

    In rank selection, individuals are first sorted by fitness, and then assigned
    selection probabilities based on their rank rather than their raw fitness value.
    The selection pressure can be adjusted using the `selection_pressure` parameter,
    which influences how strongly the best individuals are favored. A higher selection
    pressure makes the selection more biased toward higher-ranked individuals.

    Parameters
    ----------
    max_best : bool
        Whether higher fitness values are better (True for maximization, False for minimization).
    selection_pressure : float, optional
        The selection pressure that controls the bias toward higher-ranked individuals.
        Must be between 1.0 and 2.0. Default is 1.5.
    """

    def __init__(self, max_best: bool, selection_pressure: float = 1.5):
        super().__init__(max_best=max_best)
        if not 1.0 <= selection_pressure <= 2.0:
            raise ValueError("Selection pressure must be between 1.0 and 2.0")
        self.selection_pressure = selection_pressure

    def select(self, population_size: int, population: list) -> list:
        """
        Select a subset of individuals using linear rank-based selection.

        This method applies rank selection by sorting individuals based on fitness and assigning
        selection probabilities based on their rank rather than their raw fitness values. The
        selection pressure parameter controls how much more likely the best-ranked individuals
        are to be selected compared to lower-ranked ones.

        Parameters
        ----------
        population_size : int
            The number of individuals to select.
        population : list
            A list of individuals to select from. Each individual must have a `fitness` attribute,
            which is a list containing a single float value representing its fitness.

        Returns
        -------
        list
            A list of selected individuals from the population based on rank-weighted probabilities.
        """
        sorted_pop = sorted(population, key=lambda ind: ind.fitness[0], reverse=self.max_best)
        n = len(population)
        weights = []
        for rank in range(n):
            weight = 2 - self.selection_pressure + \
                     (2 * (self.selection_pressure - 1) * (n - 1 - rank)) / (n - 1)
            weights.append(weight) # not sure if weights need normalization. think about this later.
        return random.choices(sorted_pop, weights=weights, k=population_size)


class StochasticUniversalSampling(GESelectionStrategy):
    def select(self, population_size: int, population: list) -> list:
        # Handle negative fitness values and minimization
        min_fitness = min(ind.fitness for ind in population)
        if min_fitness < 0:
            shifted_fitness = [ind.fitness - min_fitness + 1e-10 for ind in population]
        else:
            shifted_fitness = [ind.fitness + 1e-10 for ind in population]

        if not self.max_best:
            max_fitness = max(shifted_fitness)
            shifted_fitness = [max_fitness - fit for fit in shifted_fitness]

        # Calculate points
        total_fitness = sum(shifted_fitness)
        step = total_fitness / population_size
        start = random.uniform(0, step)
        points = [start + i * step for i in range(population_size)]

        # Select individuals
        selected = []
        for point in points:
            sum_fitness = 0
            for ind, fitness in zip(population, shifted_fitness):
                sum_fitness += fitness
                if sum_fitness > point:
                    selected.append(ind)
                    break

        return selected


class TruncationSelection(GESelectionStrategy):
    def __init__(self, max_best: bool, truncation_threshold: float = 0.5):
        super().__init__(max_best=max_best)
        if not 0.0 < truncation_threshold <= 1.0:
            raise ValueError("Truncation threshold must be between 0.0 and 1.0")
        self.truncation_threshold = truncation_threshold

    def select(self, population_size: int, population: list) -> list:
        # Sort population
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=self.max_best)

        # Select top individuals based on threshold
        cutoff = max(2, int(len(population) * self.truncation_threshold))
        eligible = sorted_pop[:cutoff]

        # Randomly select from truncated population
        return random.choices(eligible, k=population_size)


"""

SELECTION FUNCTIONS FOR  MULTI OBJECTIVE OPTIMISATION

"""

class MOTournamentSelection:
    def __init__(self, tournament_size=2, exploration_prob=0.1):
        self.tournament_size = tournament_size
        self.exploration_prob = exploration_prob

    def select(self, population_size: int, individuals: List) -> List:

        selected = []
        for _ in range(population_size):
            actual_tournament_size = min(self.tournament_size, len(individuals))
            tournament = np.random.choice(individuals, actual_tournament_size, replace=False)
            if np.random.rand() < self.exploration_prob:
                winner = np.random.choice(tournament)
            else:
                winner = self.crowded_comparison_operator(tournament)
            selected.append(winner)
        return selected

    def crowded_comparison_operator(self, tournament):
        """
        Implements crowded comparison operator:
        - Prefer better rank
        - If same rank, prefer larger crowding distance
        """
        if len(tournament) == 1:
            return tournament[0]

        winner = tournament[0]
        for candidate in tournament[1:]:
            if (candidate.rank < winner.rank or  # Lower rank is better
                    (candidate.rank == winner.rank and
                     candidate.crowding_distance > winner.crowding_distance)):
                winner = candidate
        return winner

class MOBinaryTournamentSelection:
    def select(self, population_size: int, individuals: List) -> List:
        selected = []
        while len(selected) < population_size:
            # Select two individuals randomly
            i1, i2 = np.random.choice(individuals, 2, replace=False)

            # If one dominates the other, select the dominating one
            if i1.rank < i2.rank:
                selected.append(i1)
            elif i2.rank < i1.rank:
                selected.append(i2)
            # If same rank, select the one with larger crowding distance
            else:
                if i1.crowding_distance > i2.crowding_distance:
                    selected.append(i1)
                else:
                    selected.append(i2)
        return selected

class MORandomSelection:
    def __init__(self, elite_ratio=0.1):
        self.elite_ratio = elite_ratio

    def select(self, population_size: int, individuals: List) -> List:
        # Sort by rank and crowding distance
        sorted_individuals = sorted(
            individuals,
            key=lambda x: (x.rank, -x.crowding_distance)
        )

        # Select elites
        n_elites = int(population_size * self.elite_ratio)
        selected = sorted_individuals[:n_elites]

        # Random selection for the rest
        remaining = population_size - n_elites
        selected.extend(np.random.choice(individuals, remaining, replace=True))

        return selected