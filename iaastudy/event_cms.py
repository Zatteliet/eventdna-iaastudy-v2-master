import itertools
import logging
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable

from pycm import ConfusionMatrix
from pycm.pycm_obj import pycmVectorError

from iaastudy.defs import ANNOTATORS, DocumentSet, Layer
from iaastudy.util import filter_out, map_over_leaves, merge_list

logger = logging.getLogger(__name__)

FOUND = "Found"
NOT_FOUND = "Not found"


def collect_cms(
    doc_sets: Iterable[DocumentSet], layer: Layer, match_fn: Callable
) -> Iterable[ConfusionMatrix]:
    """Yield confusion matrices built for each pair of annotators attested in `doc_sets`.
    The CM inputs are built by comparing the events found by annotators in the pairs, using `match_fn` to compare event spans.

    `layer` determines whether comparisons will be made over events or entities.
    """

    layer = layer.value

    def docs_by(annotator, docs):
        for doc in docs:
            yield doc.dnafs[annotator]

    def annotator_pairs():
        pairs = list(itertools.combinations(ANNOTATORS, 2))
        for a1, a2 in pairs:
            yield a1, a2

    for gold_annr, pred_annr in annotator_pairs():
        try:
            cm = make_cm(
                docs_by(gold_annr, doc_sets),
                docs_by(pred_annr, doc_sets),
                layer,
                match_fn,
            )
        except pycmVectorError as e:
            m = f"Error on pair: {gold_annr}, {pred_annr}: {e}"
            logger.error(m)

        cm.name = f"{gold_annr} - {pred_annr}"

        logger.info("Constructed cm: " + cm.name)
        yield cm


def make_cm(
    gold_docs: Iterable[DocumentSet],
    pred_docs: Iterable[DocumentSet],
    target_layer: Layer,
    match_fn: Callable,
):
    """Given two collections of documents, one by annotator A and one by B, return a confusion matrix.

    It is constructed by matching either event or entity annotations (indicated by `target_layer`) using `match_fn`.

    The straightforward method is to have one vector position for each gold annotation and record 0 or 1 for found or not found in the pred vector. However, we populate the gold vector with each *match* found from pred ann to the gold anns, rather that each gold ann. This ensures that the vectors are not dependent on which direction the matching happens (annotator A to annotator B or B to A), since the number of annotations by A and B is not necessary equal, but the number of matchings found in either direction is.
    """
    # Collect annotations.
    def get_anns(dnafs, layer):
        for dnaf in dnafs:
            anns = dnaf["doc"]["annotations"][layer].values()
            for ann in anns:
                ann["home_doc"] = dnaf["meta"]["id"]
                yield ann

    gold_anns = list(get_anns(gold_docs, target_layer))
    pred_anns = list(get_anns(pred_docs, target_layer))

    gold_vector = []
    pred_vector = []

    # Keep a list of pred annotations that were matched to a gold annotation.
    # This is necessary for calculating false positives later.
    did_match = []

    for g in gold_anns:
        # Gather candidates that match the gold and that occur in the same sentence.
        matches = [
            p for p in pred_anns if same_sentence(p, g) and match_fn(p, g)
        ]
        did_match.extend(matches)

        # If a match is found among the candidates, add 1 to the prediction vector also, else add 0.
        if len(matches) > 0:
            for _ in matches:
                gold_vector.append(FOUND)
                pred_vector.append(FOUND)  # true positives
        else:
            gold_vector.append(FOUND)
            pred_vector.append(NOT_FOUND)  # false negatives

    # Calculate false positives.
    # Every pred_ann that has NOT been matched to a gold_ann is counted as a false positive.
    for pred_ann in pred_anns:
        if pred_ann not in did_match:
            gold_vector.append(NOT_FOUND)
            pred_vector.append(FOUND)  # false positives

    cm = ConfusionMatrix(gold_vector, pred_vector)

    return cm


def same_sentence(ann1, ann2):
    """Return True if the two annotation share the same document and home sentence."""
    if (ann1["home_doc"] == ann2["home_doc"]) and (
        ann1["home_sentence"] == ann2["home_sentence"]
    ):
        return True
    return False


def avg_cm_scores(cms: Iterable[ConfusionMatrix]):
    """A `pycm.ConfusionMatrix` object holds an `cm.overall_stat` dict.

    This takes multiple CMs and returns one dict where the overall stat scores are collected in lists, and another where these scores are averaged.
    """

    def can_be_averaged(v):
        return isinstance(v, int) or isinstance(v, float)

    ds = [cm.overall_stat for cm in cms]
    ds = [filter_out(d, can_be_averaged) for d in ds]
    listed = merge_list(ds)
    averaged = map_over_leaves(listed, mean)
    return {"listed_scores": listed, "averaged_scores": averaged}


def write_report(cms: Iterable[ConfusionMatrix], outpath: Path):
    """Write F1, precision and recall scores of each cm (looking at the positive (FOUND) class) to outpath.

    PPV corresponds to precision, and TPR to recall.
    """

    def get_stat(cm, stat):
        return cm.class_stat[stat][FOUND]

    def stat_message(cm):
        m = [cm.name]
        for stat in ["F1", "PPV", "TPR"]:
            m.append("{}\t{}".format(stat, get_stat(cm, stat)))
        return "\n".join(m)

    with open(outpath, "w") as f:
        txt = (stat_message(cm) for cm in cms)
        f.write("\n\n".join(txt))
