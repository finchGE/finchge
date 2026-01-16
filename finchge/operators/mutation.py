import numpy as np
from abc import ABC, abstractmethod


class GEMutationStrategy(ABC):
    """Base class for mutation strategies"""

    def __init__(self):
        pass

    @abstractmethod
    def mutate(self, genome):
        """
        Abstract method for mutation implementation.

        Args:
            genome: Genome to mutate

        Returns:
            Mutated genome
        """
        pass


class IntFlipMutation(GEMutationStrategy):
    """
    Integer flip mutation strategy
    Args:
        mutation_probability (float): Probability of mutation per gene
        codon_size (int): Maximum value for codons
    """

    def __init__(self, mutation_probability: float, codon_size: int):
        super().__init__()
        self.mutation_probability = mutation_probability
        self.codon_size = codon_size

    def mutate(self, genome):
        """
        Perform mutation on the genome.

        Args:
            genome (np.ndarray or list): Genome to mutate

        Returns:
            np.ndarray: Mutated genome
        """
        genome = np.asarray(genome) # Convert to numpy array if not already
        mutation_mask = np.random.random(len(genome)) < self.mutation_probability
        random_genes = np.random.randint(0, self.codon_size + 1, len(genome))
        mutated_genome = np.where(mutation_mask, random_genes, genome)

        return mutated_genome


class SwapMutation(GEMutationStrategy):
    """
    Swap mutation strategy
    Args:
        mutation_probability (float): Probability of mutation
    """

    def __init__(self, mutation_probability: float):
        super().__init__()
        self.mutation_probability = mutation_probability

    def mutate(self, genome):
        """
        Perform swap mutation on the genome.

        Args:
            genome (np.ndarray or list): Genome to mutate

        Returns:
            np.ndarray: Mutated genome
        """
        genome = np.asarray(genome)
        mutated_genome = genome.copy()

        # Determine which positions to swap
        mutation_mask = np.random.random(len(genome)) < self.mutation_probability
        swap_positions = np.where(mutation_mask)[0]

        # Perform swaps
        if len(swap_positions) >= 2:
            for i in range(0, len(swap_positions) - 1, 2):
                pos1, pos2 = swap_positions[i], swap_positions[i + 1]
                mutated_genome[pos1], mutated_genome[pos2] = mutated_genome[pos2], mutated_genome[pos1]

        return mutated_genome


class GaussianMutation(GEMutationStrategy):
    """
    Gaussian noise mutation strategy
    Args:
        mutation_probability (float): Probability of mutation per gene
        std_dev (float): Standard deviation of Gaussian noise
    """

    def __init__(self, mutation_probability: float, std_dev: float = 1.0):
        super().__init__()
        self.mutation_probability = mutation_probability
        self.std_dev = std_dev

    def mutate(self, genome):
        """
        Add Gaussian noise to selected genes.

        Args:
            genome (np.ndarray or list): Genome to mutate

        Returns:
            np.ndarray: Mutated genome
        """
        genome = np.asarray(genome)
        mutation_mask = np.random.random(len(genome)) < self.mutation_probability
        noise = np.random.normal(0, self.std_dev, len(genome)) # Generate Gaussian noise

        # Apply mutation with rounding and clamping
        mutated_values = genome + noise
        mutated_genome = np.where(mutation_mask,
                                  np.clip(np.round(mutated_values), 0, None),
                                  genome)

        return mutated_genome


class InversionMutation(GEMutationStrategy):
    """
    Inversion mutation strategy
    Args:
        segment_probability (float): Probability of inverting a segment
    """

    def __init__(self, segment_probability: float):
        super().__init__()
        self.segment_probability = segment_probability

    def mutate(self, genome):
        """
        Reverse a random segment of the genome.

        Args:
            genome (np.ndarray or list): Genome to mutate

        Returns:
            np.ndarray: Mutated genome
        """
        genome = np.asarray(genome)

        # Decide whether to apply mutation
        if np.random.random() < self.segment_probability:
            length = len(genome)
            start = np.random.randint(0, length - 1)
            end = np.random.randint(start + 1, length)

            mutated_genome = genome.copy()
            mutated_genome[start:end] = genome[start:end][::-1]
            return mutated_genome

        return genome


class CyclicMutation(GEMutationStrategy):
    """
    Cyclic mutation strategy
    Args:
        mutation_probability (float): Probability of mutation per gene
        segment_size (int): Size of segment to rotate
    """

    def __init__(self, mutation_probability: float, segment_size: int = 3):
        super().__init__()
        self.mutation_probability = mutation_probability
        self.segment_size = segment_size

    def mutate(self, genome):
        """
        Rotate small segments of the genome.

        Args:
            genome (np.ndarray or list): Genome to mutate

        Returns:
            np.ndarray: Mutated genome
        """
        genome = np.asarray(genome)
        mutated_genome = genome.copy()

        # mutation positions
        mutation_mask = np.random.random(len(genome)) < self.mutation_probability
        positions = np.where(mutation_mask)[0]

        # cyclic
        for pos in positions:
            if pos + self.segment_size <= len(genome):
                segment = mutated_genome[pos:pos + self.segment_size]
                mutated_genome[pos:pos + self.segment_size] = np.roll(segment, 1)

        return mutated_genome


class DuplicationMutation(GEMutationStrategy):
    """
    Duplication mutation strategy
    Args:
        mutation_probability (float): Probability of mutation
        segment_size (int): Size of segment to duplicate
    """

    def __init__(self, mutation_probability: float, segment_size: int = 2):
        super().__init__()
        self.mutation_probability = mutation_probability
        self.segment_size = segment_size

    def mutate(self, genome):
        """
        Duplicate a segment and replace another segment.

        Args:
            genome (np.ndarray or list): Genome to mutate

        Returns:
            np.ndarray: Mutated genome
        """
        genome = np.asarray(genome)

        # apply or not
        if np.random.random() < self.mutation_probability:
            length = len(genome)
            if length >= 2 * self.segment_size:
                source_start = np.random.randint(0, length - self.segment_size) #source

                # target
                possible_targets = np.concatenate([
                    np.arange(0, source_start),
                    np.arange(source_start + self.segment_size, length - self.segment_size + 1)
                ])

                if len(possible_targets) > 0:
                    target_start = np.random.choice(possible_targets)

                    mutated_genome = genome.copy()
                    segment = genome[source_start:source_start + self.segment_size]
                    mutated_genome[target_start:target_start + self.segment_size] = segment
                    return mutated_genome

        return genome


class MultipleMutation(GEMutationStrategy):
    """
    Multiple mutation strategies combiner
    Args:
        strategies (list): List of mutation strategies
        probabilities (list, optional): Probabilities for each strategy
    """

    def __init__(self, strategies, probabilities=None):
        super().__init__()
        self.strategies = strategies

        if probabilities is None:
            self.probabilities = np.ones(len(strategies)) / len(strategies)
        else:
            self.probabilities = np.array(probabilities)
            self.probabilities = self.probabilities / self.probabilities.sum()

    def mutate(self, genome):
        """
        Apply a randomly selected mutation strategy.

        Args:
            genome (np.ndarray or list): Genome to mutate

        Returns:
            np.ndarray: Mutated genome
        """
        # Select strategy based on probabilities
        strategy_idx = np.random.choice(len(self.strategies), p=self.probabilities)
        selected_strategy = self.strategies[strategy_idx]

        return selected_strategy.mutate(genome)