"""Shared bounded tool-loop validation; no transport, parsing or execution policy."""
import re


class ProtocolError(ValueError):
    pass


class LimitError(RuntimeError):
    pass


def integer(value, bound=1000000):
    if type(value) is not int or abs(value) > bound:
        raise ProtocolError("expected bounded integer")
    return value


def validate_operation(op):
    fields = {"op", "id", "budget", "max_calls", "max_generated", "wall_seconds", "answer"}
    if set(op) != fields:
        raise ValueError("invalid tool_loop fields")
    if not isinstance(op["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", op["id"]):
        raise ValueError("invalid tool_loop id")
    for name, upper in (("budget", 256), ("max_calls", 16), ("max_generated", 4096),
                        ("wall_seconds", 120)):
        if type(op[name]) is not int or not 1 <= op[name] <= upper:
            raise ValueError(f"invalid {name}")
    integer(op["answer"], 1000000000000)
