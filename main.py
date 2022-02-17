from pathlib import Path
from iaastudy.match_fns import match_fns
from iaastudy.iaastudy import run_iaa_study


def main(
    out_dir: str,
    restricted: bool = False,
    match_fn: str = "fallback",
    specific_topics: bool = False,
):

    data_zip = Path("data/iaa_set_dnaf.zip")
    out_dir = Path(out_dir)
    match_fn = match_fns[match_fn]

    run_iaa_study(data_zip, out_dir, match_fn, restricted, specific_topics)


if __name__ == "__main__":
    main("data/results/restricted_heads", restricted=True)
    main("data/results/all_heads", restricted=False)
