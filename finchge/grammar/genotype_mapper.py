from finchge.grammar.rule import Rule
from finchge.grammar.tree_node import TreeNode
from typing import List, Dict
from tabulate import tabulate
import html

class GenotypeMapper:
    """
    Maps a genotype (a list of integers) to a phenotype (a structured output string)
    using a user-defined grammar in the form of production rules.

    This class implements a depth- and wrap-aware mapping mechanism that builds
    a derivation tree from a list of codons, navigating the grammar rules accordingly.
    It supports optional repair strategies to post-process incomplete or malformed phenotypes.
    Args:
            rules (Dict[str, Rule]): A dictionary mapping non-terminal symbols to their grammar rules.
            terminals (Set[str]): A set of terminal symbols in the grammar.
            non_terminals (Set[str]): A set of non-terminal symbols in the grammar.
            max_recursion_depth (int, optional): Maximum depth for recursive expansion. Defaults to 10.
            max_wraps (int, optional): Maximum allowed genotype wraps (restart from beginning). Defaults to 3.
            repair_strategy (optional): Optional strategy object for repairing malformed phenotypes.
    """
    def __init__(self, rules: Dict[str, Rule], terminals: set, non_terminals: set,
                 max_recursion_depth=10, max_wraps=3, repair_strategy=None):
        self.rules = rules
        self.terminals = terminals
        self.non_terminals = non_terminals
        self.start_rule = next(iter(rules.keys()))
        self.max_recursion_depth = max_recursion_depth
        self.max_wraps = max_wraps
        self.repair_strategy = repair_strategy


    def __str__(self):
        """
        Returns string representation of grammar rules in BNF format.

        Returns:
            str: BNF grammar as a formatted string
        """
        mapper_str = (f"GenotypeMapper\n"
                      f"max_recursion_depth={self.max_recursion_depth}\n"
                      f"max_wraps={self.max_wraps}")
        return mapper_str

    def __repr__(self):
        """
        Returns representation shown in Jupyter when object is evaluated.
        """
        return self.__str__()

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
            info_str = "GenotypeMapper:<br />"
        else:
            info_str = "GenotypeMapper:\n"
        table_data = [
            ["Max Recursion Depth", self.max_recursion_depth],
            ["Max Wraps", self.max_wraps],
        ]
        headers = ["Attribute", "Value"]
        info_str += tabulate(table_data, headers=headers, tablefmt=tablefmt)
        return info_str

    def map(self, genotype: List[int]):
        """
        Maps a given genotype (list of integers) to a phenotype string using the grammar.

        This function builds a derivation tree by applying grammar rules based on the
        codons in the genotype. It handles recursive rule expansion, wraps, and depth limits.

        Args:
            genotype (List[int]): A list of integers representing the genotype.

        Returns:
            Tuple:
                phenotype (str): The resulting phenotype string.
                used_genotype (List[int]): List of codons actually used during the mapping.
                used_codons_count (int): Number of codons consumed.
                invalid (bool): Whether the mapping failed due to invalid symbols or limits.
                root (TreeNode): The root node of the derivation tree.
        """
        phenotype_ = []
        root = TreeNode(self.start_rule)
        stack = [(root, 0, 0)]  # root node, codon_index, depth

        genotype_len = len(genotype)
        current_index = 0
        used_codons_count = 0
        used_genotype = []
        invalid = False
        wraps = 0

        while stack:
            current_node, _, depth = stack.pop()
            current_symbol = current_node.symbol

            if depth > self.max_recursion_depth:
                invalid = True
                break


            if len(current_symbol) > 0:
                if current_symbol in self.terminals:
                    phenotype_.append(current_symbol)
                elif current_symbol in self.non_terminals:

                    if current_index >= genotype_len:
                        wraps += 1
                        current_index = 0

                    # Check if we've exceeded the maximum number of wraps
                    if wraps > self.max_wraps:
                        invalid = True
                        break

                    node = self.rules[current_symbol]
                    choices = node.choices
                    choice_index = genotype[current_index] % len(choices)


                    used_codons_count += 1
                    used_genotype.append(genotype[current_index])
                    current_index += 1
                    selected_choice = choices[choice_index]

                    for symbol in reversed(selected_choice):
                        if symbol.strip():
                            child_node = TreeNode(symbol)
                            current_node.add_child(child_node)
                            stack.append((child_node, current_index, depth + 1))
                else:
                    invalid = True
                    print(f"Symbol '{current_symbol}' is not recognized as terminal or non-terminal.")
                    break

        phenotype = "".join(phenotype_).replace('""', '"') # remove double quotes if any

        # Apply repair if needed
        if self.repair_strategy is not None:
            phenotype = self.repair_strategy.repair(phenotype)
        return phenotype, used_genotype, used_codons_count, invalid, root