from pathlib import Path
from iaastudy.match_fns import match_fns
from iaastudy.iaastudy import run_iaa_study


def main(
    out_dir: str,
    restricted: bool = False,
    specific_topics: bool = False,
):

    data_zip = Path("data/iaa_set_dnaf.zip")
    out_dir = Path(out_dir)

    run_iaa_study(
        data_zip, out_dir, match_fns["fallback"], restricted, specific_topics
    )


if __name__ == "__main__":
    main("data/results/restricted_heads", restricted=True)
    main("data/results/all_heads", restricted=False)
