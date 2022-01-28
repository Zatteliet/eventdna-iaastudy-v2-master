from nis import match
from unicodedata import name
import typer
from pathlib import Path
from iaastudy.match_fns import fallback_match
from iaastudy.iaastudy import run_iaa_study
from loguru import logger
import sys

DATA_ZIP = Path("data/iaa_set_dnaf.zip")

# OUT_DIR = Path("data/out")
OUT_DIR = Path("/mnt/c/Users/camie/Downloads/IAA_OUT/")


MATCH_FN = fallback_match


def main():
    config_logger()
    run_iaa_study(DATA_ZIP, OUT_DIR, MATCH_FN)


def config_logger():
    handlers = [dict(sink=sys.stdout, level="INFO")]
    logger.configure(handlers=handlers)


if __name__ == "__main__":
    main()
