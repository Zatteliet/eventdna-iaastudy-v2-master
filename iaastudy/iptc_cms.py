from itertools import combinations
from statistics import mean
from typing import Iterable, List

from pycm import ConfusionMatrix

from iaastudy.defs import ANNOTATORS, DocumentSet
from sklearn.preprocessing import MultiLabelBinarizer


def get_iptc_codes(dnaf: dict, most_specific: bool):
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


def get_accuracy(gold_codes, pred_codes):
    """Return the accuracy score comparing `gold_codes` to `pred_codes`."""
    mlb = MultiLabelBinarizer()
    g, p = mlb.fit_transform([gold_codes, pred_codes])
    cm = ConfusionMatrix(g, p)

    # The cm sometimes encodes the positive class as a string rather than an int.
    if not 1 in cm.ACC:
        return cm.ACC["1"]
    return cm.ACC[1]


def micro_avg_prf_score(
    gold_dnafs: List[dict], pred_dnafs: List[dict], most_specific: bool
) -> dict:
    """Return micro-average PRF scores comparing gold and pred `dnaf` documents."""
    mlb = MultiLabelBinarizer()
    flat_gold_vector = []
    flat_pred_vector = []

    for gold_dnaf, pred_dnaf in zip(gold_dnafs, pred_dnafs):

        gold_codes = get_iptc_codes(gold_dnaf, most_specific)
        pred_codes = get_iptc_codes(pred_dnaf, most_specific)

        gold_vector, pred_vector = mlb.fit_transform([gold_codes, pred_codes])

        flat_gold_vector.extend(gold_vector)
        flat_pred_vector.extend(pred_vector)

    cm = ConfusionMatrix(flat_gold_vector, flat_pred_vector)
    scores = {"precision": cm.PPV[1], "recall": cm.TPR[1], "f1": cm.F1[1]}
    return scores


def macro_avg_acc_score(gold_dnafs: List[dict], pred_dnafs: List[dict], most_specific: bool) -> float:
    """Return macro-averaged accuracy scores comparing gold and pred `dnaf` documents."""
    scores = []
    for gold_dnaf, pred_dnaf in zip(gold_dnafs, pred_dnafs):
        g = get_iptc_codes(gold_dnaf, most_specific)
        p = get_iptc_codes(pred_dnaf, most_specific)
        scores.append(get_accuracy(g, p))
    return mean(scores)


def collect_scores_over_pairs(
    doc_sets: Iterable[DocumentSet], most_specific: bool, macro: bool
):
    """Yield scores for each annotator pairs attested in `doc_sets`.

    Args:
        doc_sets (Iterable[DocumentSet]): document set containing 4 annotated `dnaf` documents.
        most_specific (bool): if True, the most specific IPTC codes will be compared. If False, the most general ones.
        macro (bool): return macro-averaged ACC scores or micro-averaged PRF scores.
    """

    for gold_annr, pred_annr in combinations(ANNOTATORS, 2):
        gold_dnafs = (doc_set.dnafs[gold_annr] for doc_set in doc_sets)
        pred_dnafs = (doc_set.dnafs[pred_annr] for doc_set in doc_sets)

        if macro:
            scores = macro_avg_acc_score(
                gold_dnafs,
                pred_dnafs,
                most_specific=most_specific,
            )
        else:
            scores = micro_avg_prf_score(
                gold_dnafs, pred_dnafs, most_specific=most_specific
            )

        yield gold_annr, pred_annr, scores
