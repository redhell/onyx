import random
import string
from uuid import uuid4

from scratch.qdrant.schemas.chunk import QdrantChunk
from scratch.qdrant.schemas.source_type import SourceType


def fake_email() -> str:
    name = "".join(random.choices(string.ascii_lowercase, k=6))
    domain = random.choice(["example.com", "test.com", "onyx.ai"])
    return f"{name}@{domain}"


def fake_content() -> str:
    words = [
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
        "sed",
        "do",
        "eiusmod",
        "tempor",
    ]
    return " ".join(random.choices(words, k=random.randint(10, 30)))


def fake_source_type() -> SourceType:
    return random.choice(list(SourceType))


def fake_acl() -> list[str]:
    return [fake_email() for _ in range(random.randint(1, 3))]


def generate_fake_qdrant_chunks(n: int, content: str | None = None):
    """
    Generator that yields n fake QdrantChunk objects.

    Args:
        n: Number of chunks to generate
        content: Optional fixed content for all chunks. If None, generates random content.
    """
    for _ in range(n):
        yield QdrantChunk(
            id=uuid4(),
            document_id=uuid4(),
            source_type=fake_source_type(),
            access_control_list=fake_acl(),
            content=content if content is not None else fake_content(),
        )
