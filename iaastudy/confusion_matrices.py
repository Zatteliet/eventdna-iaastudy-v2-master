import itertools
from collections import defaultdict
from pprint import pprint
from statistics import mean

import pandas as pd
import pycm
from loguru import logger


def cms_over_annotator_pairs(docs, target_layer, match_fn):
    """Given a collection of documents, create confusion matrices for each possible gold-pred pairing of annotators.
    
    Args:
        docs (list of DNAF dicts): Unstructured list of EventDNA document-dicts (where each article is represented 4 times, once per annotator).
        target_layer (str): "events" or "entities".
        match_fn (Callable): a matching function as defined in the dict exported from `matching_functions.py`.
    
    Returns:
        dict: `pycm` confusion matrix objects, by annotator pair. E.g.
            ```
            {
                Actual anno_1 - Pred anno_2: cm_object1,
                ...
            }
            ```
    """

    logger.info(
        f"Getting confusion matrices on layer {target_layer}, on function {match_fn.__name__}"
    )

    # Sort docs by their annotator.
    annotator_to_docList = defaultdict(list)
    for doc in docs:
        annotator_to_docList[doc["meta"]["author"]].append(doc)

    # Sanity check: check that every annotator annotated the same number of docs.
    _lens = [len(l) for l in annotator_to_docList.values()]
    assert _lens[0] == _lens[1] == _lens[2] == _lens[3]

    # Get all pairs of gold-pred annotators.
    annotators = set(annotator_to_docList.keys())
    annotator_pairs = list(itertools.combinations(annotators, 2))

    # For each pair, get the right documents and create the confusion matrix.
    # Store results in a dict so that k = pair tuple and v = confusion matrix.
    annotatorPair_to_cm = {}
    for gold_atr, pred_atr in annotator_pairs:
        # logger.debug(f"Getting confusion matrix for pair ({gold_atr}, {pred_atr})")
        gold_docs = annotator_to_docList[gold_atr]
        pred_docs = annotator_to_docList[pred_atr]

        # logger.debug("Pair: {}, n gold anns: {}, n pred anns: {}", (gold_atr, pred_atr))

        try:
            cm = _confusion_matrix(gold_docs, pred_docs, target_layer, match_fn)
            cm_name = "Actual {} - Pred {}".format(gold_atr, pred_atr)
            annotatorPair_to_cm[cm_name] = cm
        except pycm.pycm_obj.pycmVectorError as e:
            logger.error("Error on pair: {}, {}: {}".format(gold_atr, pred_atr, e))

    return annotatorPair_to_cm


def _confusion_matrix(gold_docs, pred_docs, target_layer, match_fn):
    """Given two collections of documents, one by annotator A and one by B, return an appropriately-named confusion matrix.

    `target_layer` = e.g. `"events"`.
    """

    g_atr, p_atr = gold_docs[0]["meta"]["author"], pred_docs[0]["meta"]["author"]
    logger.debug(f"Comparing: gold {g_atr}, pred {p_atr}")

    # Labels as they will appear in the output CM.
    found = "Found"
    not_found = "Not found"

    ## Check input validity.
    # For each gold doc, there is exactly one pred doc with the same id.
    for gd in gold_docs:
        matching_predDocs = [
            pd for pd in pred_docs if pd["meta"]["id"] == gd["meta"]["id"]
        ]
        assert len(matching_predDocs) == 1

    ## Collect annotations.

    def get_annotations(doc_list, target_layer):
        """Collect all annotations from a list of docs.
        
        ! ADDS INFORMATION ON THE EVENT'S HOME DOCUMENT TO THE EVENT OBJECT!
        """
        anns = []
        for doc in doc_list:
            doc_anns = doc["doc"]["annotations"][target_layer].values()
            for a in doc_anns:
                a["home_doc"] = doc["meta"]["id"]
            anns.extend(doc_anns)
        return anns

    gold_anns = get_annotations(gold_docs, target_layer)
    pred_anns = get_annotations(pred_docs, target_layer)

    ## Some auxiliary functions.

    def find_matches(gold_ann, candidates):
        """Return the annotations in candidates which match with the given gold annotation.
        """
        matching_predAnns = []
        for c in candidates:
            match_flag, _ = match_fn(gold_ann, c)
            if match_flag == True:
                matching_predAnns.append(c)
        return matching_predAnns

    def same_sentence(ann1, ann2):
        """Return True if the two annotation share the same document and home sentence."""
        if (ann1["home_doc"] == ann2["home_doc"]) and (
            ann1["home_sentence"] == ann2["home_sentence"]
        ):
            return True
        return False

    ## Build found/not-found vectors.

    gold_vector = []
    pred_vector = []

    # Keep a list of pred annotations that were matched to a gold annotation.
    # This is necessary for calculating false positives later.
    pred_anns_withGoldMatch = []

    for gold_ann in gold_anns:

        # Constrain candidate pred annotations: they have to appear in the same sentence.
        candidates = [pa for pa in pred_anns if same_sentence(gold_ann, pa)]

        # Search among the constrained candidates for matches.
        matching_predAnns = find_matches(gold_ann, candidates)
        pred_anns_withGoldMatch.extend(matching_predAnns)

        # If a match is found among the candidates, add 1 to the prediction vector also, else add 0.
        has_match = len(matching_predAnns) > 0
        if has_match:

            # debug
            logger.debug(
                "\nOK\tFound matches: \n\tGOLD\t [{}]\n\tPREDS\t {}\n".format(
                    gold_ann["string"], [ma["string"] for ma in matching_predAnns]
                )
            )

            ## CC 21/01/202
            # The task here is to create two vectors, one corresponding to gold annotations and the other (pred vector) to the matches found to the gold annotations.
            # The gold vector is all positive. The pred vector indicates for each gold annotation, whether a match has been found or not.
            # The most straightforward method is to have one vector position for each gold annotation and record 0 or 1 for found or not found in the pred vector.
            # However, we go over each *matching* found to the gold anns, rather that each gold ann.
            # This ensures that the vectors are not dependent on which direction the matching happens (annotator A to annotator B or B to A),
            # since the number of annotations by A and B is not necessary equal, but the number of matchings found in either direction is.
            for _ in matching_predAnns:
                gold_vector.append(found)
                pred_vector.append(found)  # true positives
        else:

            # debug
            logger.debug(
                "\n:(\tNo match found: \n\tGOLD\t [{}]\n\tEVENTS IN DOC\n\t\t{}\n".format(
                    gold_ann["string"], "\n\t\t".join([c["string"] for c in candidates])
                )
            )

            gold_vector.append(found)
            pred_vector.append(not_found)  # false negatives

    # Calculate false positives.
    # Every pred_ann that has NOT been matched to a gold_ann is counted as a false positive.
    for pred_ann in pred_anns:
        if pred_ann not in pred_anns_withGoldMatch:
            gold_vector.append(not_found)
            pred_vector.append(found)  # false positives

    cm = pycm.ConfusionMatrix(gold_vector, pred_vector)

    return cm


def write_kappas(cm_name_to_cm, outpath):
    """Write the kappa scores of each cm to outpath."""

    r = {"kappa": {}}
    for cm_name, cm in cm_name_to_cm.items():
        r["kappa"][cm_name] = cm.Kappa

    df = pd.DataFrame(r)
    df.to_excel(outpath)
    logger.info(f"Written stats to {outpath}.")


def average_cm_scores(cm_list):
    """A `pycm.ConfusionMatrix` object holds two dicts: `cm.overall_stat` and `cm.class_stat`.
    This takes multiple CMs and returns the average of these dicts.
    """

    assert isinstance(cm_list, list)
    assert all(isinstance(cm, pycm.ConfusionMatrix) for cm in cm_list)

    # collect stat dicts
    overall_stat_dicts = [cm.overall_stat for cm in cm_list]  # {stat name: value}
    class_stat_dicts = [
        cm.class_stat for cm in cm_list
    ]  # {stat name: {class name: value}}

    ## collect values over all CMs

    overall_stats_asLists = defaultdict(list)
    for d in overall_stat_dicts:
        for stat_name, val in d.items():
            if isinstance(val, int) or isinstance(
                val, float
            ):  # ignore values we can't take the average of
                overall_stats_asLists[stat_name].append(val)

    class_stats_asLists = defaultdict(dict)  # cannot nest defaultdict calls!
    for d in class_stat_dicts:
        for stat_name, class_d in d.items():
            for class_name, val in class_d.items():
                if class_name not in class_stats_asLists[stat_name]:
                    if isinstance(val, int) or isinstance(val, float):
                        class_stats_asLists[stat_name][class_name] = []
                if isinstance(val, int) or isinstance(val, float):
                    class_stats_asLists[stat_name][class_name].append(val)

    ## average the collected values

    overall_stats_Avg = {
        stat_name: mean(vals) for stat_name, vals in overall_stats_asLists.items()
    }

    class_stats_Avg = defaultdict(dict)
    for stat_name, class_d in class_stats_asLists.items():
        for class_name, val_list in class_d.items():
            class_stats_Avg[stat_name][class_name] = mean(val_list)

    return (
        dict(overall_stats_asLists),
        dict(class_stats_asLists),
        dict(overall_stats_Avg),
        dict(class_stats_Avg),
    )
