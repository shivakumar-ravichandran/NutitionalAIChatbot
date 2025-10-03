"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# Enums for validation
class GenderEnum(str, Enum):
    male = "male"
    female = "female"
    other = "other"
    prefer_not_to_say = "prefer_not_to_say"


class DietaryPreferenceEnum(str, Enum):
    vegetarian = "vegetarian"
    non_vegetarian = "non_vegetarian"
    vegan = "vegan"
    pescatarian = "pescatarian"


class ActivityLevelEnum(str, Enum):
    sedentary = "sedentary"
    lightly_active = "lightly_active"
    moderately_active = "moderately_active"
    very_active = "very_active"


class ResponseStyleEnum(str, Enum):
    simple = "simple"
    detailed = "detailed"
    motivational = "motivational"
    balanced = "balanced"


class SeverityEnum(str, Enum):
    mild = "mild"
    moderate = "moderate"
    severe = "severe"


# Profile Schemas
class AllergyCreate(BaseModel):
    allergy: str = Field(..., min_length=1, max_length=100)
    severity: Optional[SeverityEnum] = "moderate"


class AllergyResponse(AllergyCreate):
    id: int

    class Config:
        from_attributes = True


class HealthConditionCreate(BaseModel):
    condition_name: str = Field(..., min_length=1, max_length=100)
    severity: Optional[SeverityEnum] = "moderate"
    notes: Optional[str] = Field(None, max_length=500)


class HealthConditionResponse(HealthConditionCreate):
    id: int

    class Config:
        from_attributes = True


class ProfileCreate(BaseModel):
    # Personal information
    age: int = Field(..., ge=1, le=120, description="Age in years")
    gender: Optional[GenderEnum] = None
    location: Optional[str] = Field(None, max_length=100)
    culture: Optional[str] = Field(None, max_length=50)

    # Dietary preferences
    dietary_preferences: Optional[DietaryPreferenceEnum] = "non_vegetarian"
    activity_level: Optional[ActivityLevelEnum] = "moderately_active"

    # Response preferences
    response_style: Optional[ResponseStyleEnum] = "balanced"
    language: Optional[str] = Field("en", max_length=10)

    # Health information
    allergies: Optional[List[AllergyCreate]] = []
    health_conditions: Optional[List[HealthConditionCreate]] = []

    @validator("age")
    def validate_age(cls, v):
        if v < 1 or v > 120:
            raise ValueError("Age must be between 1 and 120")
        return v


class ProfileUpdate(BaseModel):
    age: Optional[int] = Field(None, ge=1, le=120)
    gender: Optional[GenderEnum] = None
    location: Optional[str] = Field(None, max_length=100)
    culture: Optional[str] = Field(None, max_length=50)
    dietary_preferences: Optional[DietaryPreferenceEnum] = None
    activity_level: Optional[ActivityLevelEnum] = None
    response_style: Optional[ResponseStyleEnum] = None
    language: Optional[str] = Field(None, max_length=10)


class ProfileResponse(BaseModel):
    id: int
    profile_uuid: str
    age: int
    gender: Optional[str]
    location: Optional[str]
    culture: Optional[str]
    dietary_preferences: Optional[str]
    activity_level: Optional[str]
    response_style: str
    language: str
    created_at: datetime
    updated_at: Optional[datetime]
    allergies: List[AllergyResponse] = []
    health_conditions: List[HealthConditionResponse] = []

    class Config:
        from_attributes = True


# Chat Schemas
class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)
    profile_uuid: Optional[str] = None


class MessageResponse(BaseModel):
    id: int
    message_type: str
    content: str
    timestamp: datetime
    confidence_score: Optional[float]
    response_time_ms: Optional[float]

    class Config:
        from_attributes = True


class ChatSessionResponse(BaseModel):
    id: int
    session_uuid: str
    title: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    is_active: bool
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    session_uuid: str
    user_message: MessageResponse
    assistant_message: MessageResponse
    suggestions: Optional[List[str]] = []


# Admin Schemas
class DocumentUploadResponse(BaseModel):
    id: int
    document_uuid: str
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    document_type: Optional[str]
    status: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


class EntityResponse(BaseModel):
    id: int
    entity_text: str
    entity_type: str
    confidence_score: Optional[float]
    context: Optional[str]

    class Config:
        from_attributes = True


class RelationshipResponse(BaseModel):
    id: int
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence_score: Optional[float]
    context: Optional[str]

    class Config:
        from_attributes = True


class DocumentProcessingResponse(BaseModel):
    document_uuid: str
    status: str
    entities_extracted: int
    relationships_found: int
    processing_time_ms: Optional[float]
    errors: Optional[List[str]] = []


# Knowledge Base Schemas
class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    filters: Optional[Dict[str, Any]] = {}
    limit: Optional[int] = Field(10, ge=1, le=100)


class SearchResult(BaseModel):
    content: str
    score: float
    source: str
    entity_type: Optional[str]
    metadata: Optional[Dict[str, Any]] = {}


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float


# System Schemas
class HealthCheckResponse(BaseModel):
    status: str
    service: str
    version: str
    message: str


class StatusResponse(BaseModel):
    api: str
    database: str
    nlp_service: str
    components: Dict[str, str]
