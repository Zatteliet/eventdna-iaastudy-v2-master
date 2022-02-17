from itertools import combinations
from statistics import mean
from typing import Iterable

from pycm import ConfusionMatrix

from iaastudy.defs import ANNOTATORS, DocumentSet
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import cohen_kappa_score


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


# def get_krippendorf_alpha(gold_codes, pred_codes):
#     g, p = MLB.fit_transform([gold_codes, pred_codes])
#     cm = ConfusionMatrix(g, p)
#     return cm.overall_stat["Krippendorff Alpha"]


# def get_cohens_kappa(gold_codes, pred_codes):
#     """Note that Cohen's Kappa is symmetric."""
#     g, p = MLB.fit_transform([gold_codes, pred_codes])
#     print(g, p)
#     return cohen_kappa_score(g, p)


def get_accuracy(gold_codes, pred_codes):
    mlb = MultiLabelBinarizer()
    g, p = mlb.fit_transform([gold_codes, pred_codes])
    cm = ConfusionMatrix(g, p)
    # print(cm.ACC, g, p)
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


def f1_score(prec, rec):
    return (2 * (prec * rec)) / (prec + rec)


# def get_prf1(gold_codes, pred_codes):
#     """Get precision, recall and F1 score for the positive class."""
#     g, p = MLB.fit_transform([gold_codes, pred_codes])
#     # print(g, p)
#     cm = ConfusionMatrix(g, p)
#     # print(cm.precision[1], g, p)

#     label = 1 if 1 in cm.ACC else "1"

#     # if not 1 in cm.ACC:
#     #     return cm.ACC["1"]

#     # rec = cm.TPR[label]
#     # prec = cm.PPV[label]
#     # f1 = f1_score(prec, rec)

#     return cm.ACC[label]


def micro_avg_score(gold_dnafs, pred_dnafs, most_specific: bool):
    """Yield gold and pred vectors over the documents."""
    mlb = MultiLabelBinarizer()
    flat_gold_vector = []
    flat_pred_vector = []

    for gold_dnaf, pred_dnaf in zip(gold_dnafs, pred_dnafs):

        gold_codes = get_iptc_codes(gold_dnaf, most_specific)
        pred_codes = get_iptc_codes(pred_dnaf, most_specific)

        gold_vector, pred_vector = mlb.fit_transform([gold_codes, pred_codes])

        print(gold_codes, pred_codes, gold_vector, pred_vector)

        flat_gold_vector.extend(gold_vector)
        flat_pred_vector.extend(pred_vector)

    cm = ConfusionMatrix(flat_gold_vector, flat_pred_vector)
    scores = {"precision": cm.PPV[1], "recall": cm.TPR[1], "f1": cm.F1[1]}
    return scores


def macro_avg_score(gold_dnafs, pred_dnafs, most_specific: bool):
    scores = []
    for gold_dnaf, pred_dnaf in zip(gold_dnafs, pred_dnafs):
        g = get_iptc_codes(gold_dnaf, most_specific)
        p = get_iptc_codes(pred_dnaf, most_specific)
        scores.append(get_accuracy(g, p))
        scores.append(get_accuracy(g, p))
    return mean(scores)
    # return mean(recs), mean(precs), mean(f1s)


def collect_scores_over_pairs(
    doc_sets: Iterable[DocumentSet], most_specific: bool, macro: bool
):
    for gold_annr, pred_annr in combinations(ANNOTATORS, 2):
        gold_dnafs = (doc_set.dnafs[gold_annr] for doc_set in doc_sets)
        pred_dnafs = (doc_set.dnafs[pred_annr] for doc_set in doc_sets)

        if macro:
            scores = macro_avg_score(
                gold_dnafs,
                pred_dnafs,
                most_specific=most_specific,
            )
        else:
            scores = micro_avg_score(
                gold_dnafs, pred_dnafs, most_specific=most_specific
            )

        yield gold_annr, pred_annr, scores
