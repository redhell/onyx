from pydantic import BaseModel


class GeneratedImage(BaseModel):
    file_id: str
    url: str
    revised_prompt: str
    width: int | None = None
    height: int | None = None
    shape: str | None = None


# Needed for PydanticType
class GeneratedImageFullResult(BaseModel):
    images: list[GeneratedImage]
