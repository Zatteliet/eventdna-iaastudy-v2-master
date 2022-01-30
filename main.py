import typer
from pathlib import Path
from iaastudy.match_fns import match_fns
from iaastudy.iaastudy import run_iaa_study
from loguru import logger

DATA_ZIP = Path("data/iaa_set_dnaf.zip")

# OUT_DIR = Path("data/out")
OUT_DIR = Path("/mnt/c/Users/camie/Downloads/IAA_OUT/")


def main(
    data_zip: str,
    out_dir: str,
    match_fn: str,
    restricted: bool = False,
    specific_topics: bool = False,
):

    data_zip = Path(data_zip)
    out_dir = Path(out_dir)
    match_fn = match_fns[match_fn]

    config_logger()
    run_iaa_study(data_zip, out_dir, match_fn, restricted, specific_topics)


def config_logger():
    # handlers = [dict(sink=sys.stdout, level="INFO")]
    # logger.configure(handlers=handlers)
    pass


if __name__ == "__main__":
    typer.run(main)
