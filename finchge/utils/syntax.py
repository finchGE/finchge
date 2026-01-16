def highlight(grammar_text):
    """ grammar highlighter"""
    lines = grammar_text.strip().split('\n') if isinstance(grammar_text, str) else grammar_text

    html_lines = []
    for line in lines:
        if '::=' in line:
            lhs, rhs = line.split(' ::= ', 1)
            # Format LHS (non-terminals in bold pink)
            lhs = lhs.replace('<', '<b style="color: #d33682;">&lt;').replace('>', '&gt;</b>')
            # Format RHS (non-terminals in bold pink, terminals in blue)
            rhs = rhs.replace('<', '<b style="color: #d33682;">&lt;').replace('>', '&gt;</b>')
            rhs = rhs.replace(' | ', ' <span style="color: #859900;">|</span> ')
            html_lines.append(
                f'{lhs} <span style="color: #666;">::=</span> <span style="color: #268bd2;">{rhs}</span>')
        else:
            html_lines.append(line)

    from IPython.display import HTML
    return f'{"<br>".join(html_lines)}'