def remove_outer_parentheses(code: str) -> str:
    code = code.strip()
    if code.startswith("(") and code.endswith(")"):
        return code[1:-1].strip()
    return code

def remove_comment_lines(code: str) -> str:
    return "\n".join(
        line for line in code.splitlines()
        if "#" not in line
    )

def collapse_to_one_line(code: str) -> str:
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    return "".join(lines)