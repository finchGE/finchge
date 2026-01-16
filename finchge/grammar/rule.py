from finchge.utils.syntax import highlight

class Rule:
    """
    Represents a single production rule in a BNF grammar.

    Each rule contains a non-terminal symbol (e.g., "<expr>") and a list of
    possible expansions (choices) for that symbol.

    Args:
        symbol (str): The non-terminal symbol on the left-hand side (LHS) of the rule.
        choices (List[List[str]]): A list of possible right-hand side (RHS) expansions.
            Each choice is a list of symbols (either terminals or non-terminals).
    """

    def __init__(self, symbol, choices):
        self.symbol = symbol
        self.choices = choices

    def __repr__(self):
        """
        String representation of the node showing its symbol and choices.
        """
        choices_str = " | ".join([" ".join(choice) for choice in self.choices])
        return f"{self.symbol} ::= {choices_str}"

    def __str__(self):
        """
        String representation of the node showing its symbol and choices.
        """
        choices_str = " | ".join([" ".join(choice) for choice in self.choices])
        return f"{self.symbol} ::= {choices_str}"

    def _repr_html_(self):
        """
        String representation of the node showing its symbol and choices.
        """
        choices_str = " | ".join([" ".join(choice) for choice in self.choices])
        rule_string = f"{self.symbol} ::= {choices_str}"
        return highlight(rule_string)