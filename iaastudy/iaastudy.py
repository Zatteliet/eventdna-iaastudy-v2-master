import json
from pathlib import Path
from zipfile import ZipFile

from loguru import logger

from iaastudy.alpino_heads import add_heads
from iaastudy.cm import avg_cm_scores, collect_cms, write_report
from iaastudy.defs import ANNOTATORS, DocumentSet, Layer
from iaastudy.util import recursive_delete


def run_iaa_study(data_zip: Path, out_dir: Path, match_fn) -> None:
    prepare(out_dir)

    # Read in the annotated data.
    unzipped_dir = data_zip.with_suffix("")
    check_extract(data_zip, unzipped_dir)
    doc_sets = list(get_doc_set(d) for d in unzipped_dir.iterdir())
    logger.info(f"Read in {len(doc_sets)} sets of 4 annotated docs each.")

    # Augment events in the data with their head sets.
    for doc_set in doc_sets:
        for dnaf in doc_set.dnafs.values():
            add_heads(dnaf, doc_set.alpino)

    # Get confusion matrices and scores.
    cms = list(
        collect_cms(doc_sets=doc_sets, layer=Layer.EVENTS, match_fn=match_fn)
    )

    # Write out the cms.
    for cm in cms:
        p = (out_dir / cm.name).with_suffix(".html")
        cm.save_html(str(p))

    # Average the scores over all annotator pairs. Write out.
    for score_name, vals in avg_cm_scores(cms).items():
        with open((out_dir / score_name).with_suffix(".json"), "w") as f:
            json.dump(vals, f, indent=4, sort_keys=True)

    # Write a more straightforward score report.
    write_report(cms, out_dir / "f1_prec_rec.txt")

    logger.success(f"All done. Wrote to {out_dir}")


def prepare(out_dir: Path) -> None:
    if out_dir.exists():
        logger.warning("Out dir already exists. Cleaning it...")
        recursive_delete(out_dir)
    out_dir.mkdir(exist_ok=True)


def get_doc_set(doc_dir: Path) -> DocumentSet:
    """Parse a directory consisting of 4 annotated documents (one per annotator) and a directory with Alpino data."""

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


def check_extract(zip: Path, target: Path) -> None:
    """Extract `zip_p` to `target` if this has not been done already."""
    if not target.exists():
        target.mkdir()
        logger.info(f"Extracting zip to {target}")
        with ZipFile(zip) as z:
            z.extractall(target)
    else:
        logger.info(f"Found existing data dir: {target}")
