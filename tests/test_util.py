from iaastudy.util import map_over_leaves, merge, filter_out


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

    assert merge(cases) == expected


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

# def test_merge_with_callback():
#     cases = [
#         {"a": 1, "b": 2, "c": {"ca": 30, "cb": 50}},
#         {"a": 17, "b": 18, "c": {"ca": 130, "cb": 150}},
#     ]
#     expected = {
#         "a": 18,
#         "b": 20,
#         "c": {"ca": 160, "cb": 200},
#     }

#     assert merge(cases, callback=sum) == expected
