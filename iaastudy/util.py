import random
import string
from collections import defaultdict
from typing import Callable
from copy import deepcopy


def recursive_delete(dirpath):
    # deleted everything in a dir, recursively.
    for item in dirpath.iterdir():
        if item.is_dir():
            recursive_delete(item)
            item.rmdir()
        else:
            item.unlink()


def four_char_code():
    candidates = string.ascii_lowercase + string.digits + string.digits
    code = [random.choice(candidates) for _ in range(4)]
    return "".join(code)


def merge(ds):  # , condition=None, callback=None):
    """Recursively aggregate dictionaries with the same keys.
    If `condition` is given, don't aggregate values failing to meet it.
    If `callback` is given, call this function on the aggregates values.
    """
    r = {}
    first_d = ds[0]
    for k, v in first_d.items():
        if isinstance(v, dict):
            r[k] = merge([each[k] for each in ds])
        else:
            # if condition:
            #     if not condition(v):
            #         continue
            r[k] = [each[k] for each in ds]
            # if callback:
            #     r[k] = callback(r[k])

    return r


def map_over_leaves(d: dict, c: Callable):
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


def filter_out(d: dict, condition: Callable):
    """Return a copy of `d`, such that each leaf not passing `condition` is removed."""
    r = {}
    for k, v in d.items():
        if isinstance(v, dict):
            r[k] = filter_out(v, condition)
        else:
            if condition(v):
                r[k] = v
    return r


def remove_punct(tokens):
    """Return a new list of tokens such that punctuation is removed.
    Specifically, discard any 'token' that is composed entirely of punctuation marks.
    """
    new_list = []
    for token in tokens:
        if not all((char in string.punctuation) for char in token):
            new_list.append(token)
    return new_list


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


def dice_coef(items1, items2):
    if len(items1) + len(items2) == 0:
        return 0
    intersect = set(items1).intersection(set(items2))
    return 2.0 * len(intersect) / (len(items1) + len(items2))
