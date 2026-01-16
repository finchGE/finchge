from typing import List
import numpy as np
from finchge.algorithm import BaseAlgorithm


class GeneticAlgorithm(BaseAlgorithm):
    """
    Genetic Algorithm

    Args:
        selection: Selection strategy instance or function.
        crossover: Crossover strategy instance or function.
        mutation: Mutation strategy instance or function.
        replacement: Replacement strategy instance or function.
        elite_size (int): Number of elite individuals to carry over.
    """
    def __init__(self, selection, crossover, mutation, replacement, elite_size: int):
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
            population_size=population.config['population_size']
        )


        new_population.eval()
        return new_population

    def get_best(self, population) -> List:
        fitness_functions = population.fitness_evaluator.get_fitness_functions()
        if not isinstance(fitness_functions, list):
            fitness_functions = [fitness_functions]

        if len(fitness_functions) != 1:
            raise ValueError("GeneticAlgorithm only supports single-objective optimization. "
                             f"Provided {len(fitness_functions)} fitness functions.")

        fitness_fn = fitness_functions[0]
        if fitness_fn.maximize:
            return max(population.individuals, key=lambda ind: ind.fitness)
        return min(population.individuals, key=lambda ind: ind.fitness)


    def sort_population(self, population) -> None:
        """
        Sorts the population based on fitness values.
        Supports the fitness_evaluator with single fitness function.

        Args:
            population (Population): The population to sort.
        """
        fitness_functions = population.fitness_evaluator.get_fitness_functions()
        if not isinstance(fitness_functions, list):
            fitness_functions = [fitness_functions]

        if len(fitness_functions) != 1:
            raise ValueError("GeneticAlgorithm only supports single-objective optimization. "
                             f"Provided {len(fitness_functions)} fitness functions.")

        fitness_fn = fitness_functions[0]

        reverse = fitness_fn.maximize if hasattr(fitness_fn, 'maximize') else False

        population.individuals.sort(
            key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'),
            reverse=reverse
        )

class CrosslessGA(BaseAlgorithm):
    """
    CrossLess Genetic Algorithm
    The offsprings are generated from by mutating the selected individuals without crossover.

    Args:
        selection: Selection strategy instance or function.
        mutation: Mutation strategy instance or function.
        replacement: Replacement strategy instance or function.
        elite_size (int): Number of elite individuals to carry over.
    """

    def __init__(self, selection, mutation, replacement, elite_size: int):

        self.selection = selection
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

        offspring_genome = []
        while len(offspring_genome) < population.config['population_size']:
            for individual in selected_pop:
                individual.mutate(mutation_strategy=self.mutation)
                offspring_genome.append(individual.genotype)
                if len(offspring_genome) >= population.config['population_size']:
                    break

        # Create new population from mutated individuals
        new_population = population.__class__(
            population.config,
            fitness_fn=population.fitness_fn,
            grammar=population.grammar,
            genome=offspring_genome,
            cache_manager=population.cache_manager
        )

        new_population.eval()
        self.sort_population(new_population, fitness_functions=new_population.fitness_fn)

        # Replacement
        new_population.individuals = self.replacement.replace(
            new_population=new_population.individuals,
            old_population=population.individuals,
            elite_size=self.elite_size,
            population_size=population.config['population_size']
        )

        new_population.eval()
        return new_population

    def get_best(self, population, fitness_functions) -> List:
        if not isinstance(fitness_functions, list):
            fitness_functions = [fitness_functions]

        if len(fitness_functions) != 1:
            raise ValueError("GeneticAlgorithm only supports single-objective optimization. "
                             f"Provided {len(fitness_functions)} fitness functions.")

        fitness_fn = fitness_functions[0]
        if fitness_fn.maximize:
            return max(population.individuals, key=lambda ind: ind.fitness)
        return min(population.individuals, key=lambda ind: ind.fitness)

    def sort_population(self, population, fitness_functions) -> None:
        """
        Sorts the population based on fitness values.
        Supports both single fitness function instance or a list with one function.

        Args:
            population (Population): The population to sort.
            fitness_functions: A single fitness function or a list of fitness functions.
        """
        if not isinstance(fitness_functions, list):
            fitness_functions = [fitness_functions]

        if len(fitness_functions) != 1:
            raise ValueError("GeneticAlgorithm only supports single-objective optimization. "
                             f"Provided {len(fitness_functions)} fitness functions.")

        fitness_fn = fitness_functions[0]

        reverse = fitness_fn.maximize if hasattr(fitness_fn, 'maximize') else False

        population.individuals.sort(
            key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'),
            reverse=reverse
        )


