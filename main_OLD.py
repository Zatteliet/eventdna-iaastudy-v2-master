"""This is the entry point to the script. This file's `main` function runs when this module is called.

You should define all parameters for the script in the `CONFIG` dict. This dict is passed to the main function.
However, the search for heads in annotations is defined and parametrized in `alpino_heads.py`. 
"""

import csv
import json
import sys
import zipfile
from pathlib import Path

from loguru import logger

from iaastudy import alpino_heads, confusion_matrices, matching_functions, util

CONFIG = {
    "exp_name": "test",
    "in_data_zip": Path(
        "data/iaa_set_dnaf.zip"
        # "/mnt/c/Users/ccolruyt/repos/eventdna_repos/eventdna-webanno-reformatter/data/iaa_set/iaa_set_dnaf.zip"
    ),
    "mother_out_dir": Path("out"),
    # "match_fn": matching_functions.MFN_DICT["dice head 0.8"],
    # "match_fn": matching_functions.MFN_DICT["_token_set_match"],
    "match_fn": matching_functions.MFN_DICT["_fallback_match"],
    "log_level": "INFO",
}


def main(in_data_zip, mother_out_dir, match_fn, log_level, exp_name=None):

    ## Prepare output dir.

    if not exp_name:
        exp_name = "iaa_run_{}".format(
            util.four_char_code
        )  # generate a codename for this experiment
    out_dir = mother_out_dir / exp_name
    if out_dir.exists():
        logger.warning("Out dir already exists. Cleaning it...")
        util.recursive_delete(out_dir)
    out_dir.mkdir(exist_ok=True)

    # the log output of this script is written to a txt file in the output dir
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add(out_dir / "log_output.txt")

    # also write out the config dict used for this experiment
    config_to_print = {k: str(v) for k, v in CONFIG.items()}
    with open(out_dir / "config.json", "w") as json_out:
        json.dump(config_to_print, json_out)

    ## Read in data.
    # Collect all .dnaf.json files in the input dir into a list,
    # and pair them with their respective alpino dirs.

    logger.info("Preparing data...")
    data_zip = zipfile.ZipFile(in_data_zip)

    # Extract the zip to a temporary directory which will be deleted afterwards.
    temp_data_dir = out_dir / "temp"
    temp_data_dir.mkdir()
    data_zip.extractall(temp_data_dir)

    # Process each article dir.
    dnafs_with_alpino_dirs = []
    for d in temp_data_dir.iterdir():
        for anno_doc in list(d.glob("*.dnaf.json")):
            with open(anno_doc) as json_in:
                dnaf = json.load(json_in)
                dnafs_with_alpino_dirs.append((dnaf, d / "alpino"))

    # Sanity check.
    _n_articles = len(dnafs_with_alpino_dirs)
    assert _n_articles > 0
    logger.info("Number of articles in IAA set: {}", _n_articles)

    ## Augment events in data with head seats.

    for dnaf_file, alpino_dir in dnafs_with_alpino_dirs:
        alpino_heads.add_heads(dnaf_file, alpino_dir)

    ## Get confusion matrices and average scores, then write out the results.

    logger.info("Creating confusion matrices...")
    annotator_pair_to_cm_map = confusion_matrices.cms_over_annotator_pairs(
        docs=[dnaf for dnaf, _ in dnafs_with_alpino_dirs],
        target_layer="events",
        match_fn=match_fn,
        log_outpath=out_dir / "log_output.txt",
    )

    # Write out each confusion matrix.
    logger.info("Writing results...")
    for annotator_pair, cm in annotator_pair_to_cm_map.items():
        cm.save_html(str(out_dir / str(annotator_pair)))

    # Get average scores over annotator pairs.
    (
        list_overall_stats,
        list_class_stats,
        avg_overall_stats,
        avg_class_stats,
    ) = confusion_matrices.average_cm_scores(list(annotator_pair_to_cm_map.values()))

    # Write out results.
    with open(out_dir / "overall_stats.json", "w") as json_o:
        json.dump(list_overall_stats, json_o)
    with open(out_dir / "class_stats.json", "w") as json_o:
        json.dump(list_class_stats, json_o)
    with open(out_dir / "average_overall_stats.json", "w") as json_o:
        json.dump(avg_overall_stats, json_o)
    with open(out_dir / "average_class_stats.json", "w") as json_o:
        json.dump(avg_class_stats, json_o)
    with open(out_dir / "average_stats.csv", "w", newline="") as csv_out:
        w = csv.writer(csv_out, dialect="excel")
        for stat_name, val in avg_overall_stats.items():
            w.writerow([stat_name, val])
        for stat_name, val in avg_class_stats.items():
            w.writerow([stat_name, val])
    confusion_matrices.write_kappas(annotator_pair_to_cm_map, out_dir / "kappas.xlsx")

    # Delete the temporary data dir.
    util.recursive_delete(temp_data_dir)
    temp_data_dir.rmdir()

    logger.info("Written to dir {}.", out_dir)


if __name__ == "__main__":
    main(**CONFIG)
