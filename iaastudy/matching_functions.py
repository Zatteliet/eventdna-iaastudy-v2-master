"""Defines and provides access to different matching functions."""

import random
import string
from functools import partial, update_wrapper

from loguru import logger

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Aux
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


def _remove_punctuation(tokens):
    """Return a new list of tokens such that punctuation is removed.
    Specifically, discard any 'token' that is composed entirely of punctuation marks.
    """
    new_list = []
    for token in tokens:
        if not all((char in string.punctuation) for char in token):
            new_list.append(token)
    return new_list


def _check_span_overlap(ann1, ann2) -> str:
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


def _dice_coefficient(items1, items2):
    if len(items1) + len(items2) == 0:
        return 0
    intersect = set(items1).intersection(set(items2))
    return 2.0 * len(intersect) / (len(items1) + len(items2))


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Function factories
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


def _boolean_dice_mfn(head_or_token, threshold):
    """Return a Dice-based fuzzy matching function over two annotations,
    using the given threshold.
    """

    threshold = float(threshold)

    def head_match(ann1, ann2, threshold):
        return _dice_coefficient(ann1["head_set"], ann2["head_set"]) >= threshold

    def token_match(ann1, ann2, threshold):
        return (
            _dice_coefficient(ann1["features"]["span"], ann2["features"]["span"])
            >= threshold
        )

    if head_or_token == "head":
        r = partial(head_match, threshold=threshold)
    elif head_or_token == "token":
        r = partial(token_match, threshold=threshold)
    else:
        assert False, "Bad input."

    r.__name__ = f"dice {head_or_token} {threshold}"  # ? doesn't seem to work
    # r.__rep__ = f"dice {head_or_token} {threshold}"
    return r


def _yield_stepwise_dice_fns(head_or_token):
    """Dice fuzzy matching functions, with thresholds from 0 to 1 with step = 0.05 by default.
    """
    # range to 101 because otherwise it will not go to 100 (exclusive count)
    # threshold 100 is the same as exact set match --> compare for error check purposes
    for threshold in range(0, 101, 5):
        threshold = float(threshold / 100)
        yield _boolean_dice_mfn(head_or_token=head_or_token, threshold=threshold)


def _wrap_matching_fn(fuzzy_matching_fn):
    """Takes a fuzzy matching function with signature `(annotation1, annotation2) -> bool_match_or_not`
    and wraps it around the overlap checking function, returning a function with the signature
    `(annotation1, annotation2) -> (bool_match_or_not, overlap_type)` which handles perfect and no overlap cases correctly.
    """

    def f(ann1, ann2, fuzzy_matching_fn):
        overlap = _check_span_overlap(ann1, ann2)
        if overlap == "perfect":
            return True, overlap
        elif overlap == "none":
            return False, overlap
        else:
            return fuzzy_matching_fn(ann1, ann2), overlap

    r = partial(f, fuzzy_matching_fn=fuzzy_matching_fn)
    update_wrapper(r, fuzzy_matching_fn)
    # r.__name__ = f"{fuzzy_matching_fn.__name__} (wrapped)"
    return r


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Fuzzy matching function definitions
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


# def _tokens_without_punctuation_match_baseline(ann1, ann2):
#     return _remove_punctuation(ann1.tokens) == _remove_punctuation(ann2.tokens)


def _exact_string_match(ann1, ann2) -> bool:
    if _check_span_overlap(ann1, ann2) == "perfect":
        return True
    return False


def _head_subset_match(ann1, ann2) -> bool:
    """Return True if the heads in ann1 are a subset (or the same set) of the heads in ann2.
    NOTE: If there are no heads in either annotation, return false.
    This is because the empty set is a subset of all sets.
    """
    if len(ann1["head_set"]) == 0 or len(ann2["head_set"]) == 0:
        return False
    return set(ann1["head_set"]).issubset(set(ann2["head_set"]))


def _random_match(ann1, ann2):
    """Randomly return True or False."""
    return random.choice([True, False])


def _head_exact_match(ann1, ann2) -> bool:
    """Return True if the heads in ann1 and ann2 match exactly.
    NOTE: If there are no heads in either annotation, return false.
    """
    if len(ann1["head_set"]) == 0 or len(ann2["head_set"]) == 0:
        return False
    return ann1["head_set"] == ann2["head_set"]


def _token_set_match(ann1, ann2):
    assert len(ann1["features"]["span"]) > 0
    assert len(ann2["features"]["span"]) > 0
    return ann1["features"]["span"] == ann2["features"]["span"]


def _fallback_match(ann1, ann2):
    """Try to match heads first. If either annotation doesn't have any heads, fall back to fuzzy matching on tokens.
    """
    head_fn = _boolean_dice_mfn("head", 0.8)
    token_fn = _boolean_dice_mfn("token", 0.8)
    if head_fn(ann1, ann2):
        return True
    elif token_fn(ann1, ann2):
        return True
    return False


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Exports
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


## Collect all UNWRAPPED functions, i.e. those that don't check for perfect or no span overlap
_UNWRAPPED_MATCHING_FNS = [
    _random_match,
    _exact_string_match,
    _head_exact_match,  # expectation that it is the same as exact string match, otherwise there is an error
    _token_set_match,
    _head_subset_match,
    _fallback_match
    # _tokens_without_punctuation_match_baseline,
]
_HEAD_FNS = list(_yield_stepwise_dice_fns("head"))
_UNWRAPPED_MATCHING_FNS.extend(_HEAD_FNS)
_TOKEN_FNS = list(_yield_stepwise_dice_fns("token"))
_UNWRAPPED_MATCHING_FNS.extend(_TOKEN_FNS)

## Collect WRAPPED functions: these are meant to be exported.
# Wrapped functions also check for perfect or no span overlap.
_MATCHING_FNS_LIST = [_wrap_matching_fn(f) for f in _UNWRAPPED_MATCHING_FNS]
MFN_DICT = {mfn.__name__: mfn for mfn in _MATCHING_FNS_LIST}

# (CC 22/11/2019) this dict, to give an idea of how functions names are formatted:
# {'_exact_string_match': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D296BC80>, fuzzy_matching_fn=<function _exact_string_match at 0x000001B4D29631E0>),
#  '_head_exact_match': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D296BD08>, fuzzy_matching_fn=<function _head_exact_match at 0x000001B4D2963378>),
#  '_head_subset_match': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D296BE18>, fuzzy_matching_fn=<function _head_subset_match at 0x000001B4D2963268>),
#  '_random_match': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D296BBF8>, fuzzy_matching_fn=<function _random_match at 0x000001B4D29632F0>),
#  '_token_set_match': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D296BD90>, fuzzy_matching_fn=<function _token_set_match at 0x000001B4D2963400>),
#  '_tokens_without_punctuation_match_baseline': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D296BEA0>, fuzzy_matching_fn=<function _tokens_without_punctuation_match_baseline at 0x000001B4D2963158>),
#  'dice head 0.0': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D296BF28>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963488>, threshold=0.0)),
#  'dice head 0.05': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970048>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963510>, threshold=0.05)),
#  'dice head 0.1': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D29700D0>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963598>, threshold=0.1)),
#  'dice head 0.15': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970158>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963620>, threshold=0.15)),
#  'dice head 0.2': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D29701E0>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D29636A8>, threshold=0.2)),
#  'dice head 0.25': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970268>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963730>, threshold=0.25)),
#  'dice head 0.3': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D29702F0>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D29637B8>, threshold=0.3)),
#  'dice head 0.35': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970378>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963840>, threshold=0.35)),
#  'dice head 0.4': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970400>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D29638C8>, threshold=0.4)),
#  'dice head 0.45': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970488>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963950>, threshold=0.45)),
#  'dice head 0.5': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970510>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D29639D8>, threshold=0.5)),
#  'dice head 0.55': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970598>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963A60>, threshold=0.55)),
#  'dice head 0.6': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970620>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963AE8>, threshold=0.6)),
#  'dice head 0.65': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D29706A8>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963B70>, threshold=0.65)),
#  'dice head 0.7': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970730>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963BF8>, threshold=0.7)),
#  'dice head 0.75': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D29707B8>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963C80>, threshold=0.75)),
#  'dice head 0.8': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970840>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963D08>, threshold=0.8)),
#  'dice head 0.85': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D29708C8>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963D90>, threshold=0.85)),
#  'dice head 0.9': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970950>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963E18>, threshold=0.9)),
#  'dice head 0.95': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D29709D8>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963EA0>, threshold=0.95)),
#  'dice head 1.0': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970A60>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.head_match at 0x000001B4D2963F28>, threshold=1.0)),
#  'dice token 0.0': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970AE8>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B0D0>, threshold=0.0)),
#  'dice token 0.05': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970B70>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B158>, threshold=0.05)),
#  'dice token 0.1': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970BF8>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B1E0>, threshold=0.1)),
#  'dice token 0.15': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970C80>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B268>, threshold=0.15)),
#  'dice token 0.2': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970D08>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B2F0>, threshold=0.2)),
#  'dice token 0.25': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970D90>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B378>, threshold=0.25)),
#  'dice token 0.3': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970E18>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B400>, threshold=0.3)),
#  'dice token 0.35': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970EA0>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B488>, threshold=0.35)),
#  'dice token 0.4': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2970F28>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B510>, threshold=0.4)),
#  'dice token 0.45': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2976048>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B598>, threshold=0.45)),
#  'dice token 0.5': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D29760D0>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B620>, threshold=0.5)),
#  'dice token 0.55': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2976158>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B6A8>, threshold=0.55)),
#  'dice token 0.6': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D29761E0>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B730>, threshold=0.6)),
#  'dice token 0.65': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2976268>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B7B8>, threshold=0.65)),
#  'dice token 0.7': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D29762F0>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B840>, threshold=0.7)),
#  'dice token 0.75': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2976378>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B8C8>, threshold=0.75)),
#  'dice token 0.8': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2976400>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B950>, threshold=0.8)),
#  'dice token 0.85': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2976488>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296B9D8>, threshold=0.85)),
#  'dice token 0.9': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2976510>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296BA60>, threshold=0.9)),
#  'dice token 0.95': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2976598>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296BAE8>, threshold=0.95)),
#  'dice token 1.0': functools.partial(<function _wrap_matching_fn.<locals>.f at 0x000001B4D2976620>, fuzzy_matching_fn=functools.partial(<function _get_headOrToken_booleanDice_function.<locals>.token_match at 0x000001B4D296BB70>, threshold=1.0))}
