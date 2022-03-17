from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable


def delete_contents(dirpath: Path) -> None:
    """Recursively deletes the content in a dir."""
    for item in dirpath.iterdir():
        if item.is_dir():
            delete_contents(item)
            item.rmdir()
        else:
            item.unlink()


def merge_list(ds: Iterable[dict]) -> dict:
    """Recursively aggregate dictionaries with the same keys.

    Given two dictionaries with the same structure
    """
    result = {}
    first = ds[0]

    # Check: all dictionaries should have the same keys.
    for other in ds[1:]:
        assert set(first.keys()) == set(other.keys())

    for k, v in first.items():
        if isinstance(v, dict):
            result[k] = merge_list([each[k] for each in ds])
        else:
            result[k] = [each[k] for each in ds]
    return result


def map_over_leaves(d: dict, c: Callable) -> dict:
    """Call `c` on each "leaf" of `d`, i.e. a value in `d` or a sub-dict of `d` that is not a dict itself.
    Return a new dict, leaving `d` intact.
    """
    r = deepcopy(d)
    for k, v in r.items():
        if isinstance(v, dict):
            r[k] = map_over_leaves(v, c)
        else:
            r[k] = c(v)
    return r


def filter_out(d: dict, condition: Callable) -> dict:
    """Return a copy of `d`, such that each leaf not passing `condition` is removed."""
    r = {}
    for k, v in d.items():
        if isinstance(v, dict):
            r[k] = filter_out(v, condition)
        else:
            if condition(v):
                r[k] = v
    return r


def check_span_overlap(ann1, ann2) -> str:
    """Returns the type of overlap between annotations.
    Can return any one of the following strings:
        'perfect': annotations overlap perfectly.
        'partial': annotations overlap, but not perfectly.
        'none': annotations do not overlap at all.
    """

    s1 = set(ann1["features"]["span"])
    s2 = set(ann2["features"]["span"])

    if s1 == s2:
        return "perfect"
    elif len(s1.intersection(s2)) == 0:
        return "none"
    else:
        return "partial"


def dice_coef(items1, items2) -> float:
    """Calculate the Sørensen-Dice coefficient over two sets.
    It follows the original version: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    """
    if len(items1) + len(items2) == 0:
        return 0
    intersect = set(items1).intersection(set(items2))
    return 2.0 * len(intersect) / (len(items1) + len(items2))
