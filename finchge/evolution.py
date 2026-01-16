from finchge.utils.loghelper import get_log_dir, get_logger
from pathlib import Path
import timeit
import csv
import os
import numpy as np
from tqdm import tqdm
from pprint import pprint
from tabulate import tabulate
from finchge.fitness import FitnessEvaluator
from finchge.population import Population
from finchge.utils.results import ResultHelper
from finchge.utils.cache_manager import CacheManager
from finchge.utils.loghelper import setup_logging
from finchge.utils.config_helper import load_config
from finchge.grammar import BNFGrammar
from finchge.algorithm import GeneticAlgorithm
from finchge.operators.replacement import GenerationalReplacement
from finchge.operators.selection import TournamentSelection, RouletteWheelSelection
from finchge.operators.mutation import IntFlipMutation
from finchge.operators.crossover import OnePointCrossover
from finchge.utils.constants import ALGORITHMS

class GrammaticalEvolution:
    """
    Grammatical evolution class for running the evolution.
    This implementation is based on:
    Grammatical Genetic Evolution (GE).
    M. O'Neill and C. Ryan, "Grammatical evolution", IEEE Trans. Evol. Comput., vol. 5, no. 4, pp. 349-358, Aug. 2001.

    Args:
        fitness_evaluator (FitnessEvaluator): Evaluator to evaluate the fitness of individuals.
        grammar (dict): BNF Grammar to be used. If not provided config must be available and must contain grammar_file value
        config (dict): Configuration settings for the GE algorithm.
        algorithm (BaseAlgorithm): Evolutionary algorithm to be used (e.g., GA, NSGA). If agorithm is not provided, GA will be used (provided that config is available).
    """

    def __init__(self, fitness_evaluator, grammar=None, config=None, algorithm=None):
        # setup logging for the project
        setup_logging()
        self.logger = get_logger()

        if config:
            self.config = config
        else:
            self.config = load_config()

            '''
            if not self.config:
                raise ValueError("GE hyperparameters not found. Please make sure the config.ini file exists. The parames can either be passed as an orgument or in the config.ini file")
            '''


        self.logger.info("GE Parameters: " + str(self.config))

        self.fitness_evaluator = fitness_evaluator
        self.objective_names = self.fitness_evaluator.get_objective_names()
        self.is_moge = self.fitness_evaluator.is_multi_objective()

        if algorithm is not None:
            self.algorithm = algorithm
            print(f"GE Using {algorithm.__class__.__name__}")
            self.logger.info(f"GE Using {algorithm.__class__.__name__}")

        else:
            # If algorithm is not defined ,
            # we use GeneticAlgorithm by default,
            # But it requires GE parameters to be astored in config.ini
            if not self.config:
                raise ValueError(
                    "Algorithim not specified.")


            self.logger.info(
                "Default GeneticAlgorithm setup: Selection = TournamentSelection, "
                "Crossover = OnePointCrossover, Mutation = IntflipMutation, "
                "Replacement = GenerationalReplacement. To customize, pass an 'algorithm' argument."
            )

            required_ga_configs = ['crossover_probability', 'codon_size', 'mutation_probability', 'elite_size']
            missing = [key for key in required_ga_configs if key not in self.config]
            if missing:
                self.logger.error(f"Missing required GeneticAlgorithm configs: {missing}")
                raise ValueError(f"Missing required GeneticAlgorithm configs in config.ini: {missing}")

            # Default algorithm is single-objective
            max_best = self.fitness_evaluator.get_fitness_functions()[0].maximize

            self.algorithm = GeneticAlgorithm(
                selection=TournamentSelection(max_best=max_best),
                crossover=OnePointCrossover(codon_size=self.config['codon_size'],
                                            crossover_proba=self.config['crossover_probability']),
                mutation=IntFlipMutation(self.config['mutation_probability'],
                                         codon_size=self.config['codon_size']),
                replacement=GenerationalReplacement(max_best=max_best),
                elite_size=self.config['elite_size']
            )

        self.validate_algorithm_fitness_match()

        cache = 'lru' if not self.config else self.config.get('cache', 'lru')
        cache_size = 128 if not self.config else self.config.get('cache_size', 128)

        # Initialize components
        self.cache_manager = CacheManager(
            cache_type=cache,
            maxsize=cache_size
        )

        if not grammar:
            no_grammar_error = ("Grammar Undefined. "
                                "The GrammaticalEvolution process cannot start because the evolutionary grammar is missing. "
                                "Please supply the grammar by either passing the BNFGrammar instance as the 'grammar' argument during class instantiation,"
                                " or defining the file path using the 'grammar_file' key within the configuration settings.")
            if not self.config:
                raise ValueError(no_grammar_error)

            try:
                grammar_str = Path(self.config['grammar_file']).read_text()
                print(grammar_str)
                self.grammar = BNFGrammar(
                    grammar_str,
                    self.config['max_recursion_depth'],
                    self.config['max_wraps']
                )
                print(self.grammar)
                print(self.grammar.describe())
            except(KeyError):
                raise ValueError(no_grammar_error)
        else:
            self.grammar = grammar

        self.result_helper = ResultHelper()

    def print_params(self):
        pprint(self.config)

    def initialize_population(self):
        # Initialize population
        population = Population(
            config=self.config,
            fitness_evaluator=self.fitness_evaluator,
            grammar=self.grammar,
            cache_manager=self.cache_manager
        )
        return population


    def find_fittest(self):
        """Finds the fittest individual in the population and logs the results."""
        start = timeit.default_timer()

        population = self.initialize_population()

        # Run evolution
        fittest_ = self.fit(population)
        stop = timeit.default_timer()

        if not self.is_moge:
            # Save fittest tree
            with open(f"{get_log_dir()}/fittest_tree.json", "w") as file:
                file.write(fittest_.tree.to_json())

            print("-------------------------------------------------")
            rounded_fitness = [round(val, 4) for val in fittest_.fitness]
            print(f"Best Phenotype (Fitness: {rounded_fitness}): {fittest_.phenotype}")
            self.logger.info(f"Best Phenotype (Fitness: {rounded_fitness}): {fittest_.phenotype}")


        else:
            # Save results
            self.result_helper.save_pareto_front(fittest_, self.objective_names)

            print("-------------------------------------------------")
            print(f"Results Saved")
            self.logger.info(f"Results Saved")

        self.result_helper.generate_summary(objective_names=self.objective_names)
        self.logger.info(f'Total time taken: {stop - start :.4f} seconds')
        return fittest_

    def fit(self, population):

        # Create log directories if they are not excluded
        exclude_log_config = self.config.get("exclude_log", [])
        if not "phenotypes" in exclude_log_config:
            phenotypes_dir = os.path.join(get_log_dir(), "phenotypes")
            os.makedirs(phenotypes_dir, exist_ok=True)
        if not "genotypes" in exclude_log_config:
            genotypes_dir = os.path.join(get_log_dir(), "genotypes")
            os.makedirs(genotypes_dir, exist_ok=True)
        if not "trees" in exclude_log_config:
            trees_dir = os.path.join(get_log_dir(), "trees")
            os.makedirs(trees_dir, exist_ok=True)

        # Initial evaluation
        population.eval()


        self.algorithm.sort_population(population)
        fittest = self.algorithm.get_best(population) # individual or front for multi-obj

        # Track generation data
        csv_file_path = f"{get_log_dir()}/generations.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["generation"] + self.objective_names + ["used_codons", "tree_depth"])


        # Evolution Loop
        generation_progress = tqdm(range(1, (self.config['num_generations'] + 1)))
        for generation in generation_progress:

            population = self.algorithm.evolve_one_generation(population)
            fittest = self.algorithm.get_best(population)

            if not self.is_moge:
                # Update progress
                rounded_fitness = [round(val, 4) for val in fittest.fitness]
                generation_progress.set_description(
                    f"Generation: {generation} Best Fitness: {rounded_fitness}"
                )

                # Save Phenotype
                if not "phenotypes" in exclude_log_config:
                    phenotype_path = os.path.join(phenotypes_dir, f"{generation}.txt")
                    with open(phenotype_path, 'w') as f:
                        f.write(fittest.phenotype)

                # Save genotype to text file
                if not "genotypes" in exclude_log_config:
                    genotype_path = os.path.join(genotypes_dir, f"{generation}.txt")
                    with open(genotype_path, 'w') as f:
                        # Convert numpy array to string if needed
                        if hasattr(fittest.genotype, 'tolist'):
                            genotype_data = fittest.genotype.tolist()
                        else:
                            genotype_data = fittest.genotype
                        f.write(str(genotype_data))

                # Save tree structure to JSON
                if not 'trees' in exclude_log_config:
                    tree_path = os.path.join(trees_dir, f"{generation}.json")
                    with open(tree_path, 'w') as f:
                        f.write(fittest.tree.to_json())

                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        generation,
                        *fittest.fitness,
                        fittest.used_codons,
                        fittest.tree.max_depth
                    ])
            else:
                # Update progress
                avg_fitness = np.mean([ind.fitness for ind in fittest], axis=0)
                generation_progress.set_description(
                    f"Generation: {generation} Avg Front Fitness: {avg_fitness}"
                )

                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    for ind in fittest:
                        writer.writerow([
                            generation,
                            *ind.fitness,
                            ind.used_codons,
                            ind.tree.max_depth
                        ])

        # Clear cache before returning
        self.cache_manager.clear() # Important because if we run with different seed later same phenotype can be taken from cache,

        return fittest

    def validate_algorithm_fitness_match(self):
        num_objectives = self.fitness_evaluator.count_objectives()
        algorithm_name = self.algorithm.__class__.__name__
        expectations = ALGORITHMS.get(algorithm_name)

        if expectations:
            if 'min_objectives' in expectations and num_objectives < expectations['min_objectives']:
                raise ValueError(
                    f"{algorithm_name.upper()} requires at least {expectations['min_objectives']} objectives, got {num_objectives}.")
            if 'max_objectives' in expectations and num_objectives > expectations['max_objectives']:
                raise ValueError(
                    f"{algorithm_name.upper()} supports at most {expectations['max_objectives']} objectives, got {num_objectives}. Check fitness evaluator.")
        else:
            print(
                f"Warning: Unknown algorithm '{self.algorithm}'. Skipping fitness-function validation. Ensure your algorithm handles {num_objectives} objectives correctly.")

