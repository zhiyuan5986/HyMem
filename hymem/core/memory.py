"""
Memory data structures for HyMem.

This module defines the basic memory units used in the hybrid memory system.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
from pydantic import BaseModel, Field, ConfigDict


class MemoryNote(BaseModel):
    """Basic memory unit with metadata."""

    model_config = ConfigDict(extra='allow')

    content: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    links: List[Any] = Field(default_factory=list)
    importance_score: float = 1.0
    retrieval_count: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M"))
    last_accessed: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M"))
    context: Optional[str] = None
    evolution_history: List[Any] = Field(default_factory=list)
    category: str = "Uncategorized"
    tags: List[str] = Field(default_factory=list)

    def update_access_time(self) -> None:
        self.last_accessed = datetime.now().strftime("%Y%m%d%H%M")
        self.retrieval_count += 1

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryNote":
        return cls.model_validate(data)


class MemorySummary(BaseModel):
    """Summary of memory content for efficient retrieval."""

    content: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    link: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "MemorySummary":
        return cls.model_validate(data)


class LLMSpan(BaseModel):
    """Fine-grained LLM span linked to a MemorySummary id."""

    content: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "LLMSpan":
        return cls.model_validate(data)
