from itertools import combinations
from statistics import mean
from typing import Iterable

from pycm import ConfusionMatrix

from iaastudy.defs import ANNOTATORS, DocumentSet
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import cohen_kappa_score

MLB = MultiLabelBinarizer()


def get_iptc_codes(dnaf, most_specific: bool):
    """Yield IPTC annotations on the given dnaf document.
    Only yield the certain annotations, not the uncertain ones.
    If `most_specific` is True, return the most specific topic in the chain; else, return the most general one.

    A single topic is encoded as a chain of topics in the IPTC hierarchy, starting with the most specific and ending with the most general.
    e.g.
    ```
    [
        [
            "medtop:20000644",
            "international organisation"
        ],
        [
            "medtop:20000638",
            "international relations"
        ],
        [
            "medtop:11000000",
            "politics"
        ]
    ]
    ```
    """
    topics = []
    for code_chain in dnaf["doc"]["annotations"]["iptc_codes"]["certain"]:
        if most_specific:
            code, _ = code_chain[0]
            topics.append(code)
        else:
            code, _ = code_chain[-1]
            topics.append(code)
    return set(topics)


def get_krippendorf_alpha(gold_codes, pred_codes):
    g, p = MLB.fit_transform([gold_codes, pred_codes])
    cm = ConfusionMatrix(g, p)
    return cm.overall_stat["Krippendorff Alpha"]


def get_cohens_kappa(gold_codes, pred_codes):
    """Note that Cohen's Kappa is symmetric."""
    g, p = MLB.fit_transform([gold_codes, pred_codes])
    print(g, p)
    return cohen_kappa_score(g, p)


def get_accuracy(gold_codes, pred_codes):
    g, p = MLB.fit_transform([gold_codes, pred_codes])
    # print(g, p)
    cm = ConfusionMatrix(g, p)
    print(cm.ACC, g, p)
    if not 1 in cm.ACC:
        return cm.ACC["1"]
    return cm.ACC[1]

    # assert 1 in cm.ACC or "1" in cm.ACC

    # # assert cm.ACC.get(1), cm.ACC
    # try:
    #     return cm.ACC["1"]
    # except KeyError as e:
    #     print(cm.ACC)
    #     raise e
    # # return cm.Overall_ACC


def collect_scores_over_pairs(doc_sets: Iterable[DocumentSet], most_specific: bool):

    scorer = get_accuracy

    def macro_avg_score(gold_dnafs, pred_dnafs):
        scores = []
        for gold_dnaf, pred_dnaf in zip(gold_dnafs, pred_dnafs):
            g = get_iptc_codes(gold_dnaf, most_specific)
            p = get_iptc_codes(pred_dnaf, most_specific)
            scores.append(scorer(g, p))
        return mean(scores)

    for gold_annr, pred_annr in combinations(ANNOTATORS, 2):
        g = (doc_set.dnafs[gold_annr] for doc_set in doc_sets)
        p = (doc_set.dnafs[pred_annr] for doc_set in doc_sets)

        yield gold_annr, pred_annr, macro_avg_score(g, p)
