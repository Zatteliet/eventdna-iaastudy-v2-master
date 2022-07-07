"""Run the IAA study over data provided in `assets/iaa_set_dnaf.zip`.

Args:
    out_dir (Path): All study results will be written to this dir.
    restricted_mode (bool): If True, heads belonging to modifiers will not be considered during head matching.
    most_specific (bool): If True, the most specific annotated IPTC codes will be used for IPTC match evaluation. If False, the codes will be generalized to their top-level equivalents.
"""
from datetime import datetime
import logging
from pathlib import Path

from iaastudy.iaastudy import run_iaa_study
from iaastudy.match_fns import match_fns


data_zip = Path("assets/iaa_set_dnaf.zip")
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out_dir = Path("output") / timestamp
out_dir.mkdir()

logging.basicConfig(filename=out_dir/"log.log", level=logging.DEBUG)

# Run with restricted heads.
d = out_dir / "restricted_heads"
d.mkdir(parents=True)
run_iaa_study(
    data_zip,
    d,
    match_fns["fallback"],
    restricted_mode=True,
    most_specific=False,
)

# Run with all heads.
d = out_dir / "all_heads"
d.mkdir(parents=True)
run_iaa_study(
    data_zip,
    d,
    match_fns["fallback"],
    restricted_mode=False,
    most_specific=False,
)
