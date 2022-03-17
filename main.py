from pathlib import Path
from iaastudy.match_fns import match_fns
from iaastudy.iaastudy import run_iaa_study


def main(
    out_dir: Path,
    restricted: bool = False,
    specific_topics: bool = False,
):
    """Run the IAA study over data provided in `assets/iaa_set_dnaf.zip`.

    Args:
        out_dir (Path): All study results will be written to this dir.
        restricted (bool): If True, heads belonging to modifiers will not be considered during head matching.
        specific_topics (bool): If True, the most specific annotated IPTC codes will be used for IPTC match evaluation. If False, the codes will be generalized to their top-level equivalents.
    """

    data_zip = Path("assets/iaa_set_dnaf.zip")

    run_iaa_study(
        data_zip, out_dir, match_fns["fallback"], restricted, specific_topics
    )


if __name__ == "__main__":
    out_dir = Path("data/results")
    main(out_dir / "restricted_heads", restricted=True)
    main(out_dir / "all_heads", restricted=False)
