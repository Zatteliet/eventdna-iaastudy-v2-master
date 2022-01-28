import dataclasses
from loguru import logger
from pathlib import Path
from zipfile import ZipFile

import json
from dataclasses import dataclass
from iaastudy.alpino_heads import add_heads
from iaastudy.cm import (
    make_cm,
    annotator_pairs,
    docs_by,
    collect_cms,
    avg_cm_scores,
    write_report,
)
from iaastudy.annotators import ANNOTATORS
from iaastudy.util import recursive_delete


@dataclass
class DocumentSet:
    dnafs: dict
    alpino: Path


def run_iaa_study(data_zip: Path, out_dir: Path, match_fn):

    # Prepare the output dir.
    prepare(out_dir)

    # Read in the annotated data.
    unzipped_dir = data_zip.with_suffix("")
    check_extract(data_zip, unzipped_dir)
    data = list(get_data(d) for d in unzipped_dir.iterdir())

    assert len(data) > 0

    # Augment events in the data with their head sets.
    for doc in data:
        for dnaf in doc.dnafs.values():
            add_heads(dnaf, doc.alpino)

    # Get confusion matrices and scores.
    cms = list(collect_cms(docs=data, layer="events", match_fn=match_fn))

    # Write out the cms.
    for cm in cms:
        p = (out_dir / cm.name).with_suffix(".html")
        cm.save_html(str(p))

    # Average the scores over all annotator pairs. Write out.
    scores = avg_cm_scores(cms)
    for score_name, vals in scores.items():
        with open((out_dir / score_name).with_suffix(".json"), "w") as f:
            json.dump(vals, f)

    # Write a more straightforward score report.
    write_report(cms, out_dir / "f1_prec_rec.txt")

    logger.success(f"All done. Wrote to {out_dir}")


def prepare(out_dir):
    if out_dir.exists():
        logger.warning("Out dir already exists. Cleaning it...")
        recursive_delete(out_dir)
    out_dir.mkdir(exist_ok=True)


def get_data(doc_dir: Path):
    def read(json_file):
        with open(json_file) as f:
            return json.load(f)

    def get_dnaf(annotator):
        path = (doc_dir / annotator).with_suffix(".dnaf.json")
        return read(path)

    return DocumentSet(
        dnafs={annr: get_dnaf(annr) for annr in ANNOTATORS},
        alpino=doc_dir / "alpino",
    )


def check_extract(zip: Path, target: Path):
    """Extract `zip_p` to `target` if this has not been done already."""
    if not target.exists():
        target.mkdir()
        logger.info(f"Extracting zip to {target}")
        with ZipFile(zip) as z:
            z.extractall(target)
    else:
        logger.info(f"Using existing data dir: {target}")
