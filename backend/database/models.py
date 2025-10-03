"""
Database models for Nutritional AI Chatbot
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Float,
    Boolean,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.database import Base
import uuid


class UserProfile(Base):
    """User profile model without authentication"""

    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    profile_uuid = Column(
        String(36), unique=True, index=True, default=lambda: str(uuid.uuid4())
    )

    # Personal information
    age = Column(Integer, nullable=False)
    gender = Column(String(20))
    location = Column(String(100))
    culture = Column(String(50))

    # Dietary preferences
    dietary_preferences = Column(String(20))  # veg/non-veg/vegan
    activity_level = Column(String(20))  # sedentary/moderate/active

    # Response preferences
    response_style = Column(
        String(20), default="balanced"
    )  # simple/detailed/motivational
    language = Column(String(10), default="en")

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    allergies = relationship(
        "UserAllergy", back_populates="profile", cascade="all, delete-orphan"
    )
    health_conditions = relationship(
        "UserHealthCondition", back_populates="profile", cascade="all, delete-orphan"
    )
    chat_sessions = relationship(
        "ChatSession", back_populates="profile", cascade="all, delete-orphan"
    )


class UserAllergy(Base):
    """User allergies model"""

    __tablename__ = "user_allergies"

    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(Integer, ForeignKey("user_profiles.id"))
    allergy = Column(String(100), nullable=False)
    severity = Column(String(20))  # mild/moderate/severe

    # Relationships
    profile = relationship("UserProfile", back_populates="allergies")


class UserHealthCondition(Base):
    """User health conditions model"""

    __tablename__ = "user_health_conditions"

    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(Integer, ForeignKey("user_profiles.id"))
    condition_name = Column(String(100), nullable=False)
    severity = Column(String(20))
    notes = Column(Text)

    # Relationships
    profile = relationship("UserProfile", back_populates="health_conditions")


class ChatSession(Base):
    """Chat session model"""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_uuid = Column(
        String(36), unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
    profile_id = Column(Integer, ForeignKey("user_profiles.id"))

    # Session metadata
    title = Column(String(200))
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)

    # Relationships
    profile = relationship("UserProfile", back_populates="chat_sessions")
    messages = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    """Chat message model"""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))

    # Message content
    message_type = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)

    # Metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    tokens_used = Column(Integer)
    response_time_ms = Column(Float)
    confidence_score = Column(Float)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class Document(Base):
    """Uploaded document model"""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    document_uuid = Column(
        String(36), unique=True, index=True, default=lambda: str(uuid.uuid4())
    )

    # Document metadata
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(10))  # pdf/docx/txt
    file_size = Column(Integer)
    file_path = Column(String(500))

    # Document classification
    document_type = Column(
        String(50)
    )  # age_guidance/food_availability/cultural/nutrition_data
    category = Column(String(50))

    # Processing status
    status = Column(
        String(20), default="uploaded"
    )  # uploaded/processing/processed/error
    processed_at = Column(DateTime(timezone=True))
    processing_errors = Column(Text)

    # Timestamps
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    entities = relationship(
        "ExtractedEntity", back_populates="document", cascade="all, delete-orphan"
    )


class ExtractedEntity(Base):
    """Extracted entities from documents"""

    __tablename__ = "extracted_entities"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))

    # Entity information
    entity_text = Column(String(255), nullable=False)
    entity_type = Column(
        String(50), nullable=False
    )  # FOOD_ITEM/NUTRIENT/HEALTH_CONDITION/etc
    start_pos = Column(Integer)
    end_pos = Column(Integer)
    confidence_score = Column(Float)

    # Context
    context = Column(Text)
    sentence = Column(Text)

    # Relationships
    document = relationship("Document", back_populates="entities")


class EntityRelationship(Base):
    """Relationships between entities"""

    __tablename__ = "entity_relationships"

    id = Column(Integer, primary_key=True, index=True)

    # Source and target entities
    source_entity = Column(String(255), nullable=False)
    target_entity = Column(String(255), nullable=False)
    relationship_type = Column(
        String(50), nullable=False
    )  # CONTAINS/BENEFITS/SUITABLE_FOR/etc

    # Metadata
    confidence_score = Column(Float)
    document_source = Column(String(255))
    context = Column(Text)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class VectorEmbedding(Base):
    """Vector embeddings for documents and entities"""

    __tablename__ = "vector_embeddings"

    id = Column(Integer, primary_key=True, index=True)

    # Reference information
    reference_type = Column(String(20), nullable=False)  # document/entity/chunk
    reference_id = Column(String(100), nullable=False)

    # Content and embedding
    content = Column(Text, nullable=False)
    embedding_vector = Column(Text)  # JSON string of vector

    # Metadata
    model_name = Column(String(100))
    vector_dimension = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
