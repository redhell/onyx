from typing import List
from typing import Optional

from pydantic import BaseModel

from onyx.connectors.interfaces import ConnectorCheckpoint


class DrupalWikiSpace(BaseModel):
    """Model for a Drupal Wiki space"""

    id: int
    name: str
    type: str
    description: Optional[str] = None
    accessStatus: Optional[str] = None
    color: Optional[str] = None


class DrupalWikiPage(BaseModel):
    """Model for a Drupal Wiki page"""

    id: int
    title: str
    homeSpace: int
    lastModified: int
    type: str


class DrupalWikiPageContent(BaseModel):
    """Model for the content of a Drupal Wiki page"""

    id: int
    title: str
    body: str
    homeSpace: int
    lastModified: int
    type: str


class DrupalWikiSpaceResponse(BaseModel):
    """Model for the response from the Drupal Wiki spaces API"""

    totalPages: int
    totalElements: int
    size: int
    content: List[DrupalWikiSpace]
    number: int
    first: bool
    last: bool
    numberOfElements: int
    empty: bool


class DrupalWikiPageResponse(BaseModel):
    """Model for the response from the Drupal Wiki pages API"""

    totalPages: int
    totalElements: int
    size: int
    content: List[DrupalWikiPage]
    number: int
    first: bool
    last: bool
    numberOfElements: int
    empty: bool


class DrupalWikiCheckpoint(ConnectorCheckpoint):
    """Checkpoint for the Drupal Wiki connector"""

    current_space_index: int = 0
    current_page_index: int = 0
    current_page_id_index: int = 0
    spaces: List[int] = []
    page_ids: List[int] = []
    is_processing_specific_pages: bool = False
