"""Check that the Dice coefficient fn as defined in iaastudy/matching_functions works as intended."""

from iaastudy.matching_functions import _dice_coefficient

if __name__ == "__main__":
    a = {1, 2, 3, 4}
    b = {5, 6, 7, 4}
    print(_dice_coefficient(a, b))
    assert _dice_coefficient(a, b) == 0.25

    a = {1, 2, 3, 4}
    b = {5, 6, 7, 8}
    print(_dice_coefficient(a, b))
    assert _dice_coefficient(a, b) == 0.0
