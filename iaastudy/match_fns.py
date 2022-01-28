from typing import Callable
from dataclasses import dataclass
from iaastudy.util import dice_coef, check_span_overlap


DICE_THRESHOLD = 0.8


def fallback_match(ann1, ann2):

    overlap = check_span_overlap(ann1, ann2)
    if overlap == "perfect":
        return True
    elif overlap == "none":
        return False
    elif head_dice_match(ann1, ann2, DICE_THRESHOLD):
        return True
    return token_dice_match(ann1, ann2, DICE_THRESHOLD)


def head_dice_match(ann1, ann2, threshold):
    return dice_coef(ann1["head_set"], ann2["head_set"]) >= threshold


def token_dice_match(ann1, ann2, threshold):
    return (
        dice_coef(ann1["features"]["span"], ann2["features"]["span"])
        >= threshold
    )
