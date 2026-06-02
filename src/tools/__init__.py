"""tools package. A module of the PoliMillionaire system, this is.

The tool registry the pipeline consumes, here it lives: a name -> callable map, by which the
JSON-dispatch loop (D-013) finds and runs a tool the model asks for.
"""
from .calculator import calculate
from .math_solvers import solve_maths


def default_tools() -> dict:
    """The standard tool registry -- just the safe-AST calculator, for now it holds.

    Pass it to QAPipeline(tools=default_tools()); by the JSON "name" the loop dispatches.
    """
    return {"calculator": calculate}


__all__ = ["calculate", "default_tools", "solve_maths"]
