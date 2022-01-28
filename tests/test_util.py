from iaastudy.util import dice_coef, filter_out, map_over_leaves, merge_list


def test_merge():
    cases = [
        {"a": 1, "b": 2, "c": {"ca": 30, "cb": 50}},
        {"a": 17, "b": 18, "c": {"ca": 130, "cb": 150}},
    ]
    expected = {
        "a": [1, 17],
        "b": [2, 18],
        "c": {"ca": [30, 130], "cb": [50, 150]},
    }

    assert merge_list(cases) == expected


def test_map_over_leaves():
    case = {
        "a": [1, 17],
        "b": [2, 18],
        "c": {"ca": [30, 130], "cb": [50, 150]},
    }

    expected = {
        "a": 18,
        "b": 20,
        "c": {"ca": 160, "cb": 200},
    }

    assert map_over_leaves(case, sum) == expected


def test_filter_out():
    cases = [
        {"a": 1, "b": 2, "c": {"ca": 30, "cb": "Hello!"}},
        {"a": 17, "b": 18, "c": {"ca": 130, "cb": "Hi!"}},
    ]
    expected = [
        {"a": 1, "b": 2, "c": {"ca": 30}},
        {"a": 17, "b": 18, "c": {"ca": 130}},
    ]

    for case, exp in zip(cases, expected):
        assert filter_out(case, condition=lambda v: isinstance(v, int)) == exp


def test_dice():
    a = {1, 2, 3, 4}
    b = {5, 6, 7, 4}
    assert dice_coef(a, b) == 0.25

    a = {1, 2, 3, 4}
    b = {5, 6, 7, 8}
    assert dice_coef(a, b) == 0.0
