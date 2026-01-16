from finchge.grammar.parser import GrammarParser
from finchge.grammar.genotype_mapper import GenotypeMapper
from finchge.utils.syntax import highlight
from tabulate import tabulate
import html

class BNFGrammar:
    """
    Represents a grammar in Backus-Naur Form (BNF) and provides methods for mapping genotypes
    to phenotypes using grammatical evolution.

    This class parses the BNF grammar string and delegates genotype-to-phenotype translation
    to an internal `GenotypeMapper` instance.

    Args:
        grammar_str (str): The grammar definition as a BNF-formatted string.
        max_recursion_depth (int, optional): Maximum depth allowed during recursive expansions. Defaults to 4.
        max_wraps (int, optional): Maximum number of allowed wraps when reading the genotype. Defaults to 6.
        repair_strategy (optional): Optional strategy to repair incomplete or invalid phenotypes.
        parser(Optional): GrammarParser will be used if not provided
        mapper(Optional): GenotypeMapper will be used if not provided
    """

    def __init__(self, grammar_str, max_recursion_depth=4, max_wraps=6, repair_strategy=None, parser=None, mapper=None):
        self.max_recursion_depth = max_recursion_depth
        parser = GrammarParser(grammar_str) if not parser else parser
        self.rules, self.rules_expanded, self.start_rule, self.terminals, self.non_terminals = parser.parse()

        self.mapper = GenotypeMapper(
            rules=self.rules_expanded,
            terminals=self.terminals,
            non_terminals=self.non_terminals,
            max_recursion_depth=self.max_recursion_depth,
            max_wraps=max_wraps,
            repair_strategy=repair_strategy
        ) if not mapper else mapper

    @classmethod
    def from_file(cls, filename, max_recursion_depth=4, max_wraps=6, repair_strategy=None, parser=None, mapper=None):
        """
            Create a BNFGrammar instance from a file containing BNF rules.
            This method reads the BNF grammar from a text file and initializes the grammar
            with the same parameters as the direct constructor.

            Args:
                filename (str): Path to the file containing BNF grammar rules
                max_recursion_depth (int, optional): Maximum depth allowed during recursive expansions. Defaults to 4.
                max_wraps (int, optional): Maximum number of allowed wraps when reading the genotype. Defaults to 6.
                repair_strategy (optional): Optional strategy to repair incomplete or invalid phenotypes.
                parser(Optional): GrammarParser will be used if not provided
                mapper(Optional): GenotypeMapper will be used if not provided
        """
        with open(filename, 'r') as f:
            grammar_str = f.read()
        return cls(
            grammar_str=grammar_str,
            max_recursion_depth=max_recursion_depth,
            max_wraps=max_wraps,
            repair_strategy=repair_strategy,
            parser=parser,
            mapper=mapper
        )

    def map_genotype_to_phenotype(self, genotype):
        """
        Maps a genotype (list of codons) to a phenotype using the grammar.

        Args:
            genotype (List[int]): A list of integers representing codons in the genotype.

        Returns:
            Tuple:
                phenotype (str): The mapped output string.
                used_genotype (List[int]): Codons used during mapping.
                used_codons_count (int): Total number of codons consumed.
                invalid (bool): Whether the mapping was invalid (due to wrap or depth).
                root (TreeNode): Root of the generated derivation tree.
        """
        return self.mapper.map(genotype)

    def __str__(self):
        """
        Returns string representation of grammar rules in BNF format.

        Returns:
            str: BNF grammar as a formatted string
        """
        return "\n".join([str(rule) for rule in self.rules.values()])

    def __repr__(self):
        """
        Returns representation shown in Jupyter when object is evaluated.
        """
        return "\n".join([str(rule) for rule in self.rules.values()])


    def _repr_html_(self):
        return highlight("\n".join([str(rule) for rule in self.rules.values()]))

    def describe(self, expanded=True):
        """
        Returns summary information about the grammar, including rule counts and structure.
        Set expanded=False to display original contracted versions like [a-z] for range if the grammar uses that syntax
        Default setting is to display expanded version.

        Args:
            expanded (bool): flag whether to show expanded version True by default.

        Returns:
           str: A formatted string containing grammar statistics and structure.
        """

        grammar_string = "\n".join([str(rule)
                                    for rule in self.rules_expanded.values()]) if expanded else "\n".join([str(rule)
                                                                                                           for rule in self.rules.values()])

        # handle display for jupyter notebook
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
            info_str = "Grammar:<br />"
            info_str += highlight(grammar_string)
            info_str += "<br /><br />"
        else:
            info_str = "Grammar:\n"
            info_str += "===========GRAMMAR============\n"
            info_str += grammar_string
            info_str += "\n\n"
        table_data = [
            ["Number of Rules", len(self.rules)],
            ["Start Rule", self.start_rule],
            ["Number of Terminals", len(self.terminals)],
            ["Number of Non-Terminals", len(self.non_terminals)],
        ]
        headers = ["Description", "Count"]
        info_str += tabulate(table_data, headers=headers, tablefmt=tablefmt) # rst, pretty,
        info_str += "\n\n"

        if in_jupyter:
            from IPython.display import display, HTML
            info_str = display(HTML(info_str))
            return info_str
        return info_str
