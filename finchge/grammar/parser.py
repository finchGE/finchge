from abc import ABC, abstractmethod
from typing import Tuple, List, Set, Dict
from finchge.grammar.rule import Rule
import re


class BaseParser(ABC):
    """
    Base class for all grammar parsers.

    Any custom parser must implement the parse() method which returns
    a standardized tuple containing grammar components.
    """

    @abstractmethod
    def parse(self) -> Tuple[Dict[str, 'Rule'], str, List[str], List[str]]:
        """
        Parse input data and return grammar components.

        Returns:
            Tuple containing:
                rules (Dict[str, Rule]): Mapping of non-terminal symbols to Rule objects.
                start_rule (str): The first non-terminal rule, considered the start rule.
                terminals (List[str]): List of terminal symbols.
                non_terminals (List[str]): List of non-terminal symbols.
        """
        pass


class GrammarParser(BaseParser):
    """
    Parses a grammar string written in Backus-Naur Form (BNF) into structured rules.

    The parser identifies terminal and non-terminal symbols, splits productions into alternatives,
    and validates the structure of the grammar.

    Attributes:
        grammar_str (str): The raw BNF grammar string.
        rules (Dict[str, Rule]): A dictionary mapping non-terminals to their grammar rules.
        non_terminals (List[str]): List of all non-terminal symbols.
        terminals (List[str]): List of all terminal symbols.
        start_rule (str): The first non-terminal encountered, treated as the start symbol.

    Args:
        grammar_str (str): The grammar definition in BNF format.

    """
    def __init__(self, grammar_str):
        super().__init__()
        self.rules = {}
        self.rules_expanded = {} # we need expanded rules too
        self.non_terminals = []
        self.terminals = []
        self.start_rule = None
        self.grammar_str = grammar_str

        # Split LHS and RHS
        self.non_terminal_pattern = re.compile(r'^<\w+>$')
        self.symbol_pattern = re.compile(
            r"""
            \d+\.\.\d+(?:\s*step\s+\d+)?      |   # 10..70 or 10..70 step 10
            '[^']*'\.\.'[^']*'(?:\s*step\s+\d+)?   |   # 'a'..'z' or '10'..'70' step N
            \[[^\]]+\]                              |   # [a-zA-Z]    (char class)
            "[^"]*"                                 |   # "abc"       (double-quoted terminal)
            '[^']*'                                 |   # 'a'         (single-quoted terminal)
            <[^>]+>                                 |   # <nonterminal>
            \d+\.\d+                                |   # float numbers
            \w+                                     |   # identifiers
            [{}()\[\];|:=,+\-*/=]                       # symbols
            """,
            re.VERBOSE
        )

        self.charclass_pattern = re.compile(r"\[([^\]]+)\]")
        self.range_pattern = re.compile(r"(?:'([^']+)'|(\d+))\.\.(?:'([^']+)'|(\d+))(?:\s*step\s+(\d+))?")


    def split_choices(self, rhs_symbols, choice_marker='|'):
        """
        Splits a list of RHS symbols into multiple choices based on a marker.

        This function is used to break down the right-hand side (RHS) of a grammar rule
        into separate production choices whenever the specified `choice_marker` is encountered.

        Args:
            rhs_symbols (List[str]): List of symbols in the RHS of a grammar rule.
            choice_marker (str, optional): Symbol that indicates a new production choice. Defaults to '|'.

        Returns:
            List[List[str]]: A list of production alternatives, where each alternative is a list of symbols.
        """
        result = []
        current = []
        for item in rhs_symbols:
            if item == choice_marker:
                if current:
                    result.append(current)
                    current = []
            else:
                current.append(item)
        if current:
            result.append(current)
        return result

    def expand_range(self, token):
        """
        Expands both quoted and unquoted ranges with optional step.
        Returns a list of symbols INCLUDING '|' separators so that split_choices works.
        """
        m = self.range_pattern.fullmatch(token)
        if not m:
            return None

        start_char, start_num, end_char, end_num, step = m.groups()
        step = int(step) if step else 1

        # Detect character range: 'a'..'z'
        if start_char and end_char:
            items = [chr(c) for c in range(ord(start_char), ord(end_char) + 1, step)]
        else:
            # Numeric range: 10..70 or '10'..'70'
            start = int(start_num or start_char)
            end = int(end_num or end_char)
            items = [str(i) for i in range(start, end + 1, step)]

        # Insert '|' separators so split_choices works
        expanded = []
        for i, item in enumerate(items):
            expanded.append(f"{item}")
            if i < len(items) - 1:
                expanded.append('|')

        return expanded

    def expand_charclass(self, token):
        m = self.charclass_pattern.fullmatch(token)
        if not m:
            return None
        content = m.group(1)

        i = 0
        items = []
        while i < len(content):
            # Detect ranges like a-z
            if i + 2 < len(content) and content[i + 1] == '-':
                start = content[i]
                end = content[i + 2]
                letters = [chr(c) for c in range(ord(start), ord(end) + 1)]
                for ch in letters:
                    items.append(ch)
                i += 3
            else:
                items.append(content[i])
                i += 1

        # Insert '|' between all tokens
        expanded = []
        for idx, ch in enumerate(items):
            expanded.append(f"{ch}")
            if idx < len(items) - 1:
                expanded.append('|')
        return expanded

    def parse(self) -> Tuple[Dict[str, Rule], Set[str], Set[str]]:
        """
        Parses the BNF grammar string into rules, terminals, and non-terminals.

        Returns:
            Tuple:
                rules (Dict[str, Rule]): Mapping of non-terminal symbols to Rule objects.
                start_rule (str): The first non-terminal rule, considered the start rule.
                terminals (List[str]): List of terminal symbols.
                non_terminals (List[str]): List of non-terminal symbols.

        Raises:
            ValueError: If the grammar syntax is invalid or contains undefined symbols.
        """
        lines = self.grammar_str.strip().split('\n')
        filtered_lines = [
            line for line in lines
            if line.strip() and not line.strip().startswith('#')
        ]

        final_list = []
        for line in filtered_lines:
            if '::=' in line:
                final_list.append(line)
            else:
                if final_list:
                    final_list[-1] += line
                else:
                    final_list.append(line)  # just in case the first line is weird

        lhs_list = []
        rhs_tokens = []

        for line in final_list:
            if '::=' in line:
                lhs, rhs = line.split('::=', 1)  # split once at ::=
                lhs = lhs.strip()

                # LHS
                if self.non_terminal_pattern.match(lhs):
                    lhs_list.append(lhs)
                else:
                    raise ValueError(f"Value Error: Invalid non-terminal.' {lhs}'")

                # RHS
                open_count = rhs.count('<')
                close_count = rhs.count('>')
                if open_count != close_count:
                    raise ValueError(f"Syntax Error in RHS of '{lhs}': unmatched angle brackets in '{rhs}'")

                tokens = [token for token in self.symbol_pattern.findall(rhs)]
                processed_tokens = []
                for token_ in tokens:

                    # Try expanding `'a'..'z'`
                    expanded = self.expand_range(token_)
                    if expanded:
                        processed_tokens.extend(expanded)
                        continue

                    # Try expanding `[a-zA-Z]`
                    expanded = self.expand_charclass(token_)
                    if expanded:
                        processed_tokens.extend(expanded)
                        continue

                    # Default: keep existing token
                    processed_tokens.append(token_)

                rhs_tokens.append((lhs, processed_tokens))
                self.rules[lhs] = Rule(lhs, self.split_choices(tokens))
                self.rules_expanded[lhs] = Rule(lhs, self.split_choices(processed_tokens))

        for lhs, tokens in rhs_tokens:
            for token in tokens:
                if token not in lhs_list and self.non_terminal_pattern.match(token):
                    raise ValueError(f"Undefined Terminal symbol {token}, found in rule for '{lhs}'")
                if token not in lhs_list:
                    if token not in self.terminals:
                        self.terminals.append(token)

        self.start_rule = lhs_list[0]
        self.non_terminals = lhs_list
        # not very efficient to return both short and expanded forms.. but maybe i will optimize it later
        return self.rules, self.rules_expanded, self.start_rule, self.terminals, self.non_terminals