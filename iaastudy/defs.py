from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# ID codes for annotators used throughout processing.
ANNOTATORS = {
    "eventdna_anno_1",
    "eventdna_anno_2",
    "eventdna_anno_3",
    "eventdna_anno_4",
}


class Layer(Enum):
    EVENTS = "events"
    ENTITIES = "entities"


@dataclass
class DocumentSet:
    """Represents a single document annotated by 4 different annotators.

    `dnafs` is a dict of the form {annotator_id: dnaf_jsonlike}
    """

    dnafs: dict
    alpino: Path
