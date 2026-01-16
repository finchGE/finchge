import numpy as np
from abc import ABC, abstractmethod
import random

class GECrossoverStrategy(ABC):
    """
    Abstract method for crossover implementation.

    Initialize crossover strategy.

    Args:
        crossover_proba (float): Probability of crossover occurring
    """

    def __init__(self, crossover_proba: float):
        if not 0 <= crossover_proba <= 1:
            raise ValueError("crossover_proba must be between 0 and 1")

        self.crossover_proba = crossover_proba

    @abstractmethod
    def cross(self, p_0, p_1, used_codons_0, used_codons_1, within_used=True):
        """
        Abstract method for crossover implementation.

        Args:
            p_0: First parent genome
            p_1: Second parent genome
            used_codons_0: Number of used codons in first parent
            used_codons_1: Number of used codons in second parent
            within_used: If True, crossover points will be within used section
        """
        pass


class OnePointCrossover(GECrossoverStrategy):
    """
    One-point crossover strategy.

    Args:
        codon_size (int): Size of each codon
        crossover_proba (float): Probability of crossover occurring
    """

    def __init__(self, codon_size: int, crossover_proba: float):
        super().__init__(crossover_proba=crossover_proba)
        self.codon_size = codon_size

    def cross(self, p_0, p_1, used_codons_0, used_codons_1, within_used=True):
        """
        Perform one-point crossover with robust handling of edge cases.

        Args:
            p_0: First parent genotype
            p_1: Second parent genotype
            used_codons_0: Number of used codons in first parent
            used_codons_1: Number of used codons in second parent
            within_used: If True, crossover points will be within used section

        Returns:
            tuple: Two child genomes
        """
        # Convert to numpy arrays if not already
        p_0 = np.asarray(p_0)
        p_1 = np.asarray(p_1)

        # Clone parents
        c_p_0, c_p_1 = p_0.copy(), p_1.copy()

        # Determine crossover points with safety checks
        if within_used:
            # Make sure used_codons values are integers
            if not isinstance(used_codons_0, int):
                used_codons_0 = len(c_p_0) if hasattr(c_p_0, '__len__') else 1
            if not isinstance(used_codons_1, int):
                used_codons_1 = len(c_p_1) if hasattr(c_p_1, '__len__') else 1

            # Ensure at least one codon is used (minimum value of 1)
            max_p_0 = max(1, min(used_codons_0, len(c_p_0) if hasattr(c_p_0, '__len__') else 1))
            max_p_1 = max(1, min(used_codons_1, len(c_p_1) if hasattr(c_p_1, '__len__') else 1))
        else:
            max_p_0 = max(1, len(c_p_0))
            max_p_1 = max(1, len(c_p_1))

        # Perform crossover based on probability
        if random.random() < self.crossover_proba and max_p_0 > 1 and max_p_1 > 1:
            try:
                # Randomly select crossover points
                pt_p_0 = random.randint(1, max_p_0)
                pt_p_1 = random.randint(1, max_p_1)

                # Create children by combining parent segments
                c_0 = np.concatenate([c_p_0[:pt_p_0], c_p_1[pt_p_1:]])
                c_1 = np.concatenate([c_p_1[:pt_p_1], c_p_0[pt_p_0:]])
            except (ValueError, IndexError) as e:
                # Fallback to parent copies if any error occurs
                print(f"Crossover failed with error: {e}. Using parent copies instead.")
                c_0, c_1 = c_p_0.copy(), c_p_1.copy()
        else:
            # If no crossover, return copies of parents
            c_0, c_1 = c_p_0.copy(), c_p_1.copy()

        return c_0, c_1


class TwoPointCrossover(GECrossoverStrategy):
    """
    Two-point crossover strategy
    Initialize two-point crossover strategy.

    Args:
        codon_size (int): Size of each codon
        crossover_proba (float): Probability of crossover occurring
    """

    def __init__(self, codon_size: int, crossover_proba: float):
        super().__init__(crossover_proba=crossover_proba)
        self.codon_size = codon_size

    def cross(self, p_0, p_1, used_codons_0, used_codons_1, within_used=True):
        """
        Perform two-point crossover.

        Args:
            p_0: First parent genome
            p_1: Second parent genome
            used_codons_0: Number of used codons in first parent
            used_codons_1: Number of used codons in second parent
            within_used: If True, crossover points will be within used section

        Returns:
            tuple: Two child genomes
        """
        # Convert to numpy arrays if not already
        p_0 = np.asarray(p_0)
        p_1 = np.asarray(p_1)

        # Clone parents
        c_p_0, c_p_1 = p_0.copy(), p_1.copy()

        # Determine crossover points range
        if within_used:
            max_p_0, max_p_1 = used_codons_0, used_codons_1
        else:
            max_p_0, max_p_1 = len(c_p_0), len(c_p_1)

        # Perform crossover based on probability
        if random.random() < self.crossover_proba:
            # Randomly select two crossover points
            pt_p_0_1 = random.randint(1, max_p_0 - 1)
            pt_p_0_2 = random.randint(pt_p_0_1 + 1, max_p_0)
            pt_p_1_1 = random.randint(1, max_p_1 - 1)
            pt_p_1_2 = random.randint(pt_p_1_1 + 1, max_p_1)

            # Create children by combining parent segments
            c_0 = np.concatenate([
                c_p_0[:pt_p_0_1], 
                c_p_1[pt_p_1_1:pt_p_1_2], 
                c_p_0[pt_p_0_2:]
            ])
            c_1 = np.concatenate([
                c_p_1[:pt_p_1_1], 
                c_p_0[pt_p_0_1:pt_p_0_2], 
                c_p_1[pt_p_1_2:]
            ])
        else:
            # If no crossover, return copies of parents
            c_0, c_1 = c_p_0.copy(), c_p_1.copy()

        return c_0, c_1


class UniformCrossover(GECrossoverStrategy):
    """
    Uniform crossover strategy
    Args:
        crossover_proba (float): Probability of crossover occurring
    """

    def __init__(self, crossover_proba: float):
        super().__init__(crossover_proba=crossover_proba)

    def cross(self, p_0, p_1, used_codons_0, used_codons_1, within_used=True):
        """
        Perform uniform crossover.

        Args:
            p_0: First parent genome
            p_1: Second parent genome
            used_codons_0: Number of used codons in first parent
            used_codons_1: Number of used codons in second parent
            within_used: If True, crossover points will be within used section

        Returns:
            tuple: Two child genomes
        """
        # Convert to numpy arrays if not already
        p_0 = np.asarray(p_0)
        p_1 = np.asarray(p_1)

        # Clone parents
        c_p_0, c_p_1 = p_0.copy(), p_1.copy()

        # Determine crossover range
        if within_used:
            max_p_0, max_p_1 = used_codons_0, used_codons_1
        else:
            max_p_0, max_p_1 = len(c_p_0), len(c_p_1)

        # Perform crossover based on probability
        if random.random() < self.crossover_proba:
            # Create child genomes
            c_0 = c_p_0.copy()
            c_1 = c_p_1.copy()

            # Determine shorter genome length
            min_length = min(max_p_0, max_p_1)

            # Randomly swap genes
            for i in range(min_length):
                if random.random() < 0.5:
                    c_0[i], c_1[i] = c_1[i], c_0[i]

        else:
            # If no crossover, return copies of parents
            c_0, c_1 = c_p_0.copy(), c_p_1.copy()

        return c_0, c_1