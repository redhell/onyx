from enum import auto
from enum import StrEnum


class SourceType(StrEnum):
    GITHUB = auto()
    BITBUCKET = auto()
    CONFLUENCE = auto()
    GOOGLE_DRIVE = auto()
    SLACK = auto()
