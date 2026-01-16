def test_grammar_creation():
    """Test basic grammar functionality."""
    from finchge.grammar import BNFGrammar
    grammar_str = """
    # Grammar for lower case letters
    <string> ::= <letter> | <letter> <string>
    <letter> ::= _ | [a-z]
    """
    grammar = BNFGrammar(grammar_str, max_recursion_depth=20)
    assert grammar is not None