from sklearn.metrics import cohen_kappa_score
from iaastudy.iptc_cms import (
    get_krippendorf_alpha,
    get_cohens_kappa,
    get_accuracy,
)


def test_get_krippendorf_alpha():
    gold_topics = ["medtop:20000644", "medtop:20000610"]
    pred_topics = ["medtop:20000644", "medtop:20000610", "medtop:20000639"]

    print(get_krippendorf_alpha(gold_topics, pred_topics))


def test_get_cohens_kappa():
    gold_topics = ["medtop:20000644", "medtop:20000610"]
    pred_topics = ["medtop:20000644", "medtop:20000610", "medtop:20000639"]

    print(get_cohens_kappa(gold_topics, pred_topics))


def test_get_accuracy():
    gold_topics = ["medtop:20000644", "medtop:20000610"]
    pred_topics = ["medtop:20000644", "medtop:20000610", "medtop:20000639"]

    print(get_accuracy(gold_topics, pred_topics))


def test_cohen():
    """Check workings of the SKLearn cohen's kappa implementation."""
    g, p = [1, 0], [1, 0]
    print(g, p, cohen_kappa_score(g, p))

    g, p = [1, 0, 1], [1, 1, 1]
    print(g, p, cohen_kappa_score(g, p))

    g, p = [1, 0, 0, 1], [1, 0, 1, 1]
    print(g, p, cohen_kappa_score(g, p))

    g, p = [1], [0]
    print(g, p, cohen_kappa_score(g, p))
