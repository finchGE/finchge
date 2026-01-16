import numpy as np
from array import array
from typing import List, Union
from tabulate import tabulate

class Individual:
    """
    Individual in a population.

    Args:
        genotype: Genetic material (array or list of integers)
        phenotype: Expressed form of the genotype (optional)
        used_codons: Number of used codons (optional)
        invalid: Whether the individual is invalid (optional)
        fitness: Fitness value (optional)
        tree: Tree representation (optional)
        evaluator: Fitness evaluator (optional) Used ofr multi-objective only
    """
    def __init__(self, genotype: Union[array, list], phenotype: str = None,
                 used_codons: int = None, invalid: bool = False,
                 fitness: Union[float, List[float]] = None, tree: str = None, evaluator = None):

        #   convert if it's not already an array or if it's a string
        if isinstance(genotype, str):
            # Handle string case that might come from JSON
            genotype = [int(x) for x in genotype.strip('[]').split(',')]
            self.genotype = array('i', genotype)
        elif isinstance(genotype, list):
            self.genotype = array('i', genotype)
        elif isinstance(genotype, array):
            self.genotype = genotype
        else:
            raise TypeError(f"Unsupported genotype type: {type(genotype)}")

        self.phenotype = phenotype
        self.used_codons = used_codons if used_codons is not None else len(genotype)
        self.invalid = invalid
        if fitness is None:
            self.fitness = float('-inf')
        else:
            self.fitness = fitness  # Can be float or list of floats

        self.tree = tree

        # Multi-objective specific attributes, only set when needed
        self.rank = None
        self.crowding_distance = None
        self.dominated_solutions = None
        self.domination_count = None

    def mutate(self, mutation_strategy) -> None:
        """Apply mutation to this individual"""
        self.genotype = mutation_strategy.mutate(self.genotype)

    def cross_with(self, p2, crossover_strategy) -> List[array]:
        """
        Perform crossover with another individual. 
        Crossover is only supported for genotype of type array.

        Returns:
            List[array]: List of offspring genotypes
        """
        return crossover_strategy.cross(
            self.genotype,
            p2.genotype,
            self.used_codons,
            p2.used_codons
        )

    def clone(self) -> 'Individual':
        """Create a deep copy of this individual"""
        return Individual(
            genotype=array('i', self.genotype),  # Create new array
            phenotype=self.phenotype,
            used_codons=self.used_codons,
            invalid=self.invalid,
            fitness=self.fitness,
            tree=self.tree
        )


    def __str__(self):
        """
        Returns string representation of grammar rules in BNF format.

        Returns:
            str: BNF grammar as a formatted string
        """
        return f"Individual\n Phenotype: {self.phenotype} \n Fitness: {self.fitness}"

    def __repr__(self):
        """
        Returns representation shown in Jupyter when object is evaluated.
        """
        return f"Individual\n Phenotype: {self.phenotype} \n Fitness: {self.fitness}"

    def _repr_html_(self):
        in_jupyter = False
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                in_jupyter = hasattr(get_ipython(), 'config')
        except:
            pass
        tablefmt = 'simple'
        if in_jupyter:
            tablefmt = 'html'

        if in_jupyter:
            info_str = "Individual:<br />"
        else:
            info_str = "Individual:\n"
        table_data = [
            ["Phenotype", self.phenotype],
            ["Genotype", self.genotype],
            ["Used Codons", self.used_codons],
            ["Fitness", self.fitness],
        ]
        headers = ["Attribute", "Value"]
        info_str += tabulate(table_data, headers=headers, tablefmt=tablefmt)
        return info_str


class Population:
    """
    Population class.

    Args:
        config: Configuration dictionary
        fitness_evaluator: Fitness evaluator
        grammar: Grammar for genotype-to-phenotype mapping
        genome: Initial genome (optional)
        phenome: Initial phenome (optional)
        used_codons: Initial used codons (optional)
        trees: Initial trees (optional)
        cache_manager: Cache manager (optional)
         **kwargs  # All custom fields go here
    """

    def __init__(self, config: dict, fitness_evaluator, grammar, genome: List[array] = None,
                 phenome: List[str] = None, used_codons: List[int] = None,
                 trees: List[str] = None, cache_manager=None, **kwargs ):

        self.config = config
        self.fitness_evaluator = fitness_evaluator
        self.grammar = grammar
        self.cache_manager = cache_manager
        self.individuals = []

        # Initialize genome if not provided
        if genome is None:
            genome = [
                array('i', np.random.randint(0, self.config['codon_size'] + 1,
                                             size=self.config['genome_length']))
                for _ in range(self.config['population_size'])
            ]
        else:
            genome = [array('i', g) if not isinstance(g, array) else g for g in genome]

        # Initialize other components
        if phenome is None:
            phenome = [None] * self.config['population_size']
        if used_codons is None:
            used_codons = [len(g) for g in genome]
        if trees is None:
            trees = ["" for _ in genome]

        # Initialize population
        fitness_value = [
            float('-inf') if fn.maximize else float('inf')
            for fn in self.fitness_evaluator.get_fitness_functions()
        ]

        for gen, phen, uc, tree_ in zip(genome, phenome, used_codons, trees):
            self.individuals.append(Individual(
                genotype=gen,
                phenotype=phen,
                used_codons=uc,
                invalid=False,
                fitness=fitness_value,
                tree=tree_
            ))

        for key, value in kwargs.items():
            setattr(self, key, value)

    def eval(self) -> None:
        """Evaluate all individuals in the population"""
        for individual in self.individuals:
            individual.phenotype, individual.used_genotype, individual.used_codons, \
                individual.invalid, individual.tree = self.grammar.map_genotype_to_phenotype(individual.genotype)

            if not individual.invalid:
                # fitness are always represented as a list even for single objective
                cache_key = f"{individual.phenotype}_fitness"
                fitness_values = self.cache_manager.get(cache_key) if self.cache_manager else None

                if fitness_values is None:
                    fitness_values = self.fitness_evaluator.evaluate(individual.phenotype)
                    if self.cache_manager:
                        self.cache_manager.set(cache_key, fitness_values)
                individual.fitness = fitness_values

            else:
                individual.fitness = [
                    float('-inf') if fn.maximize else float('inf')
                    for fn in self.fitness_evaluator.get_fitness_functions()
                ]


    def add_foreign_genotypes(self, foreign_genotypes: List[array]) -> None:
        """Add foreign genotypes to the population"""
        # TODO CHECK THIS LATER MAY NOT WORK
        for foreign_genotype in foreign_genotypes:
            new_individual = Individual(
                genotype=foreign_genotype,
                phenotype=None,
                used_codons=len(foreign_genotype),
                invalid=False,
                fitness=float('-inf'),
                tree=""
            )
            self.individuals.append(new_individual)
        self.eval()

    def replace_least_performing(self, foreign_genotypes: List[array]) -> None:
        # TODO CHECK THIS LATER MAY NOT WORK
        """Replace least performing individuals"""
        self.eval()
        self.sort()

        num_to_replace = min(len(foreign_genotypes), len(self.individuals))
        new_individuals = [
            Individual(
                genotype=genotype,
                phenotype=None,
                used_codons=len(genotype),
                invalid=False,
                fitness=float('-inf'),
                tree=""
            ) for genotype in foreign_genotypes
        ]

        self.individuals[-num_to_replace:] = new_individuals
        self.eval()

