"""
Debugging utilities
=======
"""
# src/typingkit/core/_debug.py

from typing import Any, TypeVar, get_args, get_origin

from rich.tree import Tree


def diagnostic(obj: Any, pfx: str | None = None) -> Tree:
    _pfx = f"[bold cyan]{pfx}[/] " if pfx else ""
    tree = Tree(
        f"{_pfx}[yellow]obj[/]=[green]{obj!r}[/], "
        f"[yellow]type[/]=[magenta]{type(obj).__name__}[/]"
    )
    match obj:
        case tuple():
            for x in obj:  # type: ignore
                tree.add(diagnostic(x))

        case TypeVar():
            tree.add(diagnostic(obj.__bound__, "__bound__:"))
            tree.add(diagnostic(obj.__constraints__, "__constraints__:"))
            tree.add(diagnostic(obj.__default__, "__default__:"))
            tree.add(diagnostic(obj.__covariant__, "__covariant__:"))
            tree.add(diagnostic(obj.__contravariant__, "__contravariant__:"))

        # GenericAlias | Literal | UnionType
        case _ if (origin := get_origin(obj)) is not None:
            tree.add(diagnostic(origin, "origin:"))
            tree.add(diagnostic(get_args(obj), "args:"))

        case _:
            pass
    return tree
