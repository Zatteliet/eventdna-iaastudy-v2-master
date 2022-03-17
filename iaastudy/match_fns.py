from iaastudy.util import dice_coef, check_span_overlap

DICE_THRESHOLD = 0.8


def fallback_match(ann1: dict, ann2: dict) -> bool:
    """Return True if the given annotations match."""
    overlap = check_span_overlap(ann1, ann2)
    if overlap == "perfect":
        return True
    elif overlap == "none":
        return False
    elif _head_dice_match(ann1, ann2, DICE_THRESHOLD):
        return True
    return _token_dice_match(ann1, ann2, DICE_THRESHOLD)


def _head_dice_match(ann1, ann2, threshold) -> bool:
    return dice_coef(ann1["head_set"], ann2["head_set"]) >= threshold


def _token_dice_match(ann1, ann2, threshold) -> bool:
    dice_score = dice_coef(ann1["features"]["span"], ann2["features"]["span"])
    return dice_score >= threshold


match_fns = {"fallback": fallback_match}
