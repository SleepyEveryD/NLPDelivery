"""A safe arithmetic tool -- for the maths questions, this is.

Evaluate numbers it does, but the dangerous power of eval(), grant it we never will.
Only an arithmetic AST we walk -- safe by construction, this keeps us.
"""
from __future__ import annotations

import ast
import operator

# The only operations permitted, these are.
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    # Walk the tree we do; only numbers and arithmetic, allow we will.
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers, allowed they are.")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Forbidden, this expression is.")


def calculate(expression: str) -> float:
    """Compute a pure-arithmetic expression, safely this does.

    Args:
        expression: e.g. "12 * (3 + 4) / 2".
    Returns:
        The numeric result, it does.
    """
    tree = ast.parse(expression, mode="eval")
    return _eval(tree.body)
