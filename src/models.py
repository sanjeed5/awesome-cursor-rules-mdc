from typing import Any, Dict, List

from pydantic import BaseModel, Field


class MDCRule(BaseModel):
    """Model for MDC rule data."""

    name: str = Field(
        ..., description="A meaningful name for the rule that reflects its purpose"
    )
    glob_pattern: str = Field(
        ...,
        description="Glob pattern to specify which files/folders the rule applies to",
    )
    description: str = Field(
        ...,
        description="A clear description of what the rule does and when it should be applied",
    )
    content: str = Field(..., description="The actual rule content")


class LibraryInfo(BaseModel):
    """Model for library information."""

    name: str
    tags: List[str]
    best_practices: str = ""
    citations: List[Dict[str, Any]] = []
