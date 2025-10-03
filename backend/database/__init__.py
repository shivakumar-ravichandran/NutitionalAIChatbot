"""
Database initialization module
"""

from .database import Base, engine, get_db, init_db
from .models import (
    UserProfile,
    UserAllergy,
    UserHealthCondition,
    ChatSession,
    ChatMessage,
    Document,
    ExtractedEntity,
    EntityRelationship,
    VectorEmbedding,
)

__all__ = [
    "Base",
    "engine",
    "get_db",
    "init_db",
    "UserProfile",
    "UserAllergy",
    "UserHealthCondition",
    "ChatSession",
    "ChatMessage",
    "Document",
    "ExtractedEntity",
    "EntityRelationship",
    "VectorEmbedding",
]
