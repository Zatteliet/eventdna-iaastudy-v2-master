"""This module is for data verification purposes. `main` outputs all events and their heads in the IAA corpus.
"""

# TODO Must be rewritten to use zipfiles as input data, as in `main.py`.

import json
from pathlib import Path

from loguru import logger

from iaastudy import alpino_heads

CONFIG = {
    # "exp_name": "run2 - exact token match + no Contact or UNK",
    "in_data_zip": Path(
        "/mnt/c/Users/ccolruyt/repos/eventdna_repos/eventdna-working-corpus/iaa study indexed dnaf"
    ),
    "write_out_path": Path(
        "/mnt/c/Users/ccolruyt/Desktop/IAASTUDY_OUT/heads_no_AP_ADVP.txt"
    )
    # "out_dir": Path("/mnt/c/Users/ccolruyt/Desktop/IAASTUDY_OUT"),
    # "match_fn": matching_functions.MFN_DICT["dice head 0.8"],
    # "match_fn": matching_functions.MFN_DICT["_token_set_match"],
    # "test": False,
    # "write_out": False
}


def main(in_data_zip, write_out_path):

    counts = {"Events with heads": 0, "Events without heads": 0}

    ## read in data
    logger.info("Preparing data...")

    dnafs_with_alpino_dirs = []
    for d in in_data_zip.iterdir():
        for anno_doc in list(d.glob("*.dnaf.json")):
            with open(anno_doc) as json_in:
                dnaf = json.load(json_in)
                dnafs_with_alpino_dirs.append((dnaf, d / "alpino"))

    ## augment events in data with head seats

    with open(write_out_path, "w") as write_o:

        for dnaf_file, alpino_dir in dnafs_with_alpino_dirs:
            alpino_heads.augment_with_heads(dnaf_file, alpino_dir)

            write_o.write(
                "\tDoc: {}, author: {}\n\n".format(
                    dnaf_file["meta"]["id"], dnaf_file["meta"]["author"],
                )
            )

            ## print head sets out nicely

            # print(_pretty_string(dnaf_file["doc"]["string"].split()))

            events = dnaf_file["doc"]["annotations"]["events"].values()
            # print(len(events))
            for event in events:

                write_o.write(_pretty_event(dnaf_file, event, counts))
                write_o.write("\n\n")

        write_o.write(str(counts))


def _pretty_string(words, vector):
    s = []
    for token, vector_tag in zip(words, vector):
        v = "".join([("H" if vector_tag == 1 else ".") for char in token])
        s.append(v)
    return " ".join(s)


def _pretty_event(dnaf, event, counts):

    sentence_idcs = dnaf["doc"]["sentences"][event["home_sentence"]]["token_ids"]
    sentence_tokens = [
        tok
        for i, tok in enumerate(dnaf["doc"]["token_string"].split())
        if i in sentence_idcs
    ]

    sentence_tokidc_to_doc_tokidc_to_tok = list(
        zip([i for i, _ in enumerate(sentence_idcs)], sentence_idcs, sentence_tokens)
    )
    # print(sentence_tokidc_to_doc_tokidc_to_tok)

    s = []
    s.append(" ".join(sentence_tokens))

    sent_s = []
    for s_id, doc_id, token in sentence_tokidc_to_doc_tokidc_to_tok:
        if s_id in event["head_set"]:
            sent_s.append("".join(["H" for char in token]))
        elif doc_id in event["features"]["span"]:
            sent_s.append("".join(["_" for char in token]))
        else:
            sent_s.append("".join(["." for char in token]))
    sent_s.append(
        " [{}.{}]".format(event["features"]["type"], event["features"]["subtype"])
    )
    sent_s = " ".join(sent_s)
    s.append(sent_s)

    s = "\n".join(s)

    has_event = "H" in s
    if has_event:
        counts["Events with heads"] += 1
    else:
        counts["Events without heads"] += 1

    return s


if __name__ == "__main__":
    main(**CONFIG)
