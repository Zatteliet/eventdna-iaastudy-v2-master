import itertools
from collections import defaultdict
from pprint import pprint
from statistics import mean

import pandas as pd
import pycm
from loguru import logger
from enum import Enum
from iaastudy.annotators import ANNOTATORS
from iaastudy.util import filter_out, map_over_leaves, merge


FOUND = "Found"
NOT_FOUND = "Not found"


def collect_cms(docs, layer, match_fn):
    for gold_annr, pred_annr in annotator_pairs():
        try:
            cm = make_cm(
                gold_docs=docs_by(gold_annr, docs),
                pred_docs=docs_by(pred_annr, docs),
                target_layer=layer,
                match_fn=match_fn,
            )
            cm.name = f"{gold_annr} - {pred_annr}"
            logger.success("Constructed cm: " + cm.name)
            cm.print_matrix()
            yield cm
        except pycm.pycm_obj.pycmVectorError as e:
            logger.error(
                "Error on pair: {}, {}: {}".format(gold_annr, pred_annr, e)
            )


def docs_by(annotator, docs):
    for doc in docs:
        yield doc.dnafs[annotator]


def annotator_pairs():
    pairs = list(itertools.combinations(ANNOTATORS, 2))
    for a1, a2 in pairs:
        yield a1, a2


def make_cm(gold_docs, pred_docs, target_layer, match_fn):
    """Given two collections of documents, one by annotator A and one by B, return an appropriately-named confusion matrix.

    `target_layer` = e.g. `"events"`.
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

    # Build found/not-found vectors.

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

            # debug
            logger.debug(
                "\nOK\tFound matches: \n\tGOLD\t [{}]\n\tPREDS\t {}\n".format(
                    g["string"],
                    [match["string"] for match in matches],
                )
            )

            ## CC 21/01/2020
            # The task here is to create two vectors, one corresponding to gold annotations and the other (pred vector) to the matches found to the gold annotations.
            # The gold vector is all positive. The pred vector indicates for each gold annotation, whether a match has been found or not.
            # The most straightforward method is to have one vector position for each gold annotation and record 0 or 1 for found or not found in the pred vector.
            # However, we go over each *matching* found to the gold anns, rather that each gold ann.
            # This ensures that the vectors are not dependent on which direction the matching happens (annotator A to annotator B or B to A),
            # since the number of annotations by A and B is not necessary equal, but the number of matchings found in either direction is.
            for _ in matches:
                gold_vector.append(FOUND)
                pred_vector.append(FOUND)  # true positives
        else:

            # debug
            logger.debug(
                "\n:(\tNo match found: \n\tGOLD\t [{}]\n\tEVENTS IN DOC\n\t\t{}\n".format(
                    g["string"],
                    "\n\t\t".join([c["string"] for c in matches]),
                )
            )

            gold_vector.append(FOUND)
            pred_vector.append(NOT_FOUND)  # false negatives

    # Calculate false positives.
    # Every pred_ann that has NOT been matched to a gold_ann is counted as a false positive.
    for pred_ann in pred_anns:
        if pred_ann not in did_match:
            gold_vector.append(NOT_FOUND)
            pred_vector.append(FOUND)  # false positives

    cm = pycm.ConfusionMatrix(gold_vector, pred_vector)

    return cm


def same_sentence(ann1, ann2):
    """Return True if the two annotation share the same document and home sentence."""
    if (ann1["home_doc"] == ann2["home_doc"]) and (
        ann1["home_sentence"] == ann2["home_sentence"]
    ):
        return True
    return False


def avg_cm_scores(cms):
    """A `pycm.ConfusionMatrix` object holds two dicts: `cm.overall_stat` and `cm.class_stat`.
    This takes multiple CMs and returns the average of the overall stats.
    """

    def can_be_averaged(v):
        return isinstance(v, int) or isinstance(v, float)

    ds = [cm.overall_stat for cm in cms]
    ds = [filter_out(d, can_be_averaged) for d in ds]
    listed = merge(ds)
    averaged = map_over_leaves(listed, mean)
    return {"listed_scores": listed, "averaged_scores": averaged}


def write_report(cms, outpath):
    """Write F1, PREC, REC scores of each cm to outpath."""

    def get_stat(cm, stat):
        return cm.class_stat[stat][FOUND]

    def stat_message(cm):
        stats = [
            "{}\t{}".format(stat, get_stat(cm, stat))
            for stat in ["F1", "PPV", "TPR"]
        ]
        return "\n".join([cm.name] + stats)

    with open(outpath, "w") as f:
        txt = (stat_message(cm) for cm in cms)
        f.write("\n\n".join(txt))
