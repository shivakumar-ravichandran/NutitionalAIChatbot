"""
Admin API endpoints for document management
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    Request,
)
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import os
import shutil
from pathlib import Path
import time
from datetime import datetime

from database.database import get_db
from database.models import Document, ExtractedEntity, EntityRelationship
from schemas import (
    DocumentUploadResponse,
    EntityResponse,
    RelationshipResponse,
    DocumentProcessingResponse,
)

router = APIRouter()

# Create uploads directory
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Upload a document for processing"""

    # Validate file type
    allowed_extensions = [".pdf", ".docx", ".txt"]
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Allowed types: {', '.join(allowed_extensions)}",
        )

    try:
        # Generate unique filename
        document_uuid = str(uuid.uuid4())
        filename = f"{document_uuid}{file_extension}"
        file_path = UPLOAD_DIR / filename

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(file_path)

        # Create document record
        db_document = Document(
            document_uuid=document_uuid,
            filename=filename,
            original_filename=file.filename,
            file_type=file_extension[1:],  # Remove the dot
            file_size=file_size,
            file_path=str(file_path),
            document_type=document_type,
            category=category,
            status="uploaded",
        )

        db.add(db_document)
        db.commit()
        db.refresh(db_document)

        return db_document

    except Exception as e:
        # Clean up file if database operation fails
        if file_path.exists():
            file_path.unlink()

        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}",
        )


@router.post("/process/{document_uuid}", response_model=DocumentProcessingResponse)
async def process_document(
    document_uuid: str, request: Request, db: Session = Depends(get_db)
):
    """Process uploaded document for entity extraction"""

    # Get document
    document = (
        db.query(Document).filter(Document.document_uuid == document_uuid).first()
    )

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    if document.status == "processing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is already being processed",
        )

    try:
        start_time = time.time()

        # Update status to processing
        document.status = "processing"
        db.commit()

        # Get NLP service
        nlp_service = getattr(request.app.state, "nlp_service", None)
        if not nlp_service:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="NLP service not available",
            )

        # Process document
        processing_result = await process_document_content(document, nlp_service, db)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Update document status
        document.status = "processed"
        document.processed_at = datetime.utcnow()
        db.commit()

        return DocumentProcessingResponse(
            document_uuid=document_uuid,
            status="processed",
            entities_extracted=processing_result["entities_count"],
            relationships_found=processing_result["relationships_count"],
            processing_time_ms=processing_time_ms,
            errors=processing_result.get("errors", []),
        )

    except Exception as e:
        # Update document status to error
        document.status = "error"
        document.processing_errors = str(e)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}",
        )


@router.get("/documents", response_model=List[DocumentUploadResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List uploaded documents"""
    query = db.query(Document)

    if status_filter:
        query = query.filter(Document.status == status_filter)

    documents = query.offset(skip).limit(limit).all()
    return documents


@router.get("/documents/{document_uuid}", response_model=DocumentUploadResponse)
async def get_document(document_uuid: str, db: Session = Depends(get_db)):
    """Get document details"""
    document = (
        db.query(Document).filter(Document.document_uuid == document_uuid).first()
    )

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    return document


@router.get("/documents/{document_uuid}/entities", response_model=List[EntityResponse])
async def get_document_entities(
    document_uuid: str, entity_type: Optional[str] = None, db: Session = Depends(get_db)
):
    """Get entities extracted from a document"""
    document = (
        db.query(Document).filter(Document.document_uuid == document_uuid).first()
    )

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    query = db.query(ExtractedEntity).filter(ExtractedEntity.document_id == document.id)

    if entity_type:
        query = query.filter(ExtractedEntity.entity_type == entity_type)

    entities = query.all()
    return entities


@router.get("/entities", response_model=List[EntityResponse])
async def list_entities(
    entity_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List all extracted entities"""
    query = db.query(ExtractedEntity)

    if entity_type:
        query = query.filter(ExtractedEntity.entity_type == entity_type)

    entities = query.offset(skip).limit(limit).all()
    return entities


@router.get("/relationships", response_model=List[RelationshipResponse])
async def list_relationships(
    relationship_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List entity relationships"""
    query = db.query(EntityRelationship)

    if relationship_type:
        query = query.filter(EntityRelationship.relationship_type == relationship_type)

    relationships = query.offset(skip).limit(limit).all()
    return relationships


@router.delete("/documents/{document_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_uuid: str, db: Session = Depends(get_db)):
    """Delete a document and its associated data"""
    document = (
        db.query(Document).filter(Document.document_uuid == document_uuid).first()
    )

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    try:
        # Delete file from filesystem
        file_path = Path(document.file_path)
        if file_path.exists():
            file_path.unlink()

        # Delete from database (cascades to entities)
        db.delete(document)
        db.commit()

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}",
        )


async def process_document_content(
    document: Document, nlp_service, db: Session
) -> dict:
    """Process document content and extract entities"""

    try:
        # Read document content based on file type
        content = ""
        file_path = Path(document.file_path)

        if document.file_type == "txt":
            content = file_path.read_text(encoding="utf-8")
        elif document.file_type == "pdf":
            # Placeholder for PDF processing
            content = f"PDF content from {document.original_filename} - Processing not implemented yet"
        elif document.file_type == "docx":
            # Placeholder for DOCX processing
            content = f"DOCX content from {document.original_filename} - Processing not implemented yet"

        # Extract entities using NLP service (placeholder implementation)
        entities = await extract_entities_placeholder(content, nlp_service)

        # Save entities to database
        entities_count = 0
        for entity_info in entities:
            db_entity = ExtractedEntity(
                document_id=document.id,
                entity_text=entity_info["text"],
                entity_type=entity_info["type"],
                confidence_score=entity_info.get("confidence", 0.8),
                context=entity_info.get("context", ""),
            )
            db.add(db_entity)
            entities_count += 1

        # Extract relationships (placeholder)
        relationships = await extract_relationships_placeholder(entities)

        # Save relationships to database
        relationships_count = 0
        for rel_info in relationships:
            db_relationship = EntityRelationship(
                source_entity=rel_info["source"],
                target_entity=rel_info["target"],
                relationship_type=rel_info["type"],
                confidence_score=rel_info.get("confidence", 0.7),
                document_source=document.original_filename,
            )
            db.add(db_relationship)
            relationships_count += 1

        db.commit()

        return {
            "entities_count": entities_count,
            "relationships_count": relationships_count,
            "errors": [],
        }

    except Exception as e:
        db.rollback()
        return {"entities_count": 0, "relationships_count": 0, "errors": [str(e)]}


async def extract_entities_placeholder(content: str, nlp_service) -> List[dict]:
    """Placeholder entity extraction - will be replaced with actual NLP processing"""

    # Simple keyword-based entity extraction for demo
    entities = []

    # Food items
    food_keywords = [
        "apple",
        "banana",
        "chicken",
        "rice",
        "bread",
        "milk",
        "egg",
        "fish",
        "vegetable",
        "fruit",
    ]
    for food in food_keywords:
        if food.lower() in content.lower():
            entities.append(
                {
                    "text": food,
                    "type": "FOOD_ITEM",
                    "confidence": 0.9,
                    "context": f"Found in document context",
                }
            )

    # Nutrients
    nutrient_keywords = [
        "protein",
        "carbohydrate",
        "fat",
        "vitamin",
        "mineral",
        "calcium",
        "iron",
        "fiber",
    ]
    for nutrient in nutrient_keywords:
        if nutrient.lower() in content.lower():
            entities.append(
                {
                    "text": nutrient,
                    "type": "NUTRIENT",
                    "confidence": 0.85,
                    "context": f"Nutritional information",
                }
            )

    # Health conditions
    health_keywords = ["diabetes", "hypertension", "allergy", "obesity", "anemia"]
    for condition in health_keywords:
        if condition.lower() in content.lower():
            entities.append(
                {
                    "text": condition,
                    "type": "HEALTH_CONDITION",
                    "confidence": 0.8,
                    "context": f"Health-related content",
                }
            )

    return entities


async def extract_relationships_placeholder(entities: List[dict]) -> List[dict]:
    """Placeholder relationship extraction"""

    relationships = []

    # Simple rule-based relationships
    food_entities = [e for e in entities if e["type"] == "FOOD_ITEM"]
    nutrient_entities = [e for e in entities if e["type"] == "NUTRIENT"]

    # Create CONTAINS relationships between foods and nutrients
    for food in food_entities[:3]:  # Limit for demo
        for nutrient in nutrient_entities[:2]:
            relationships.append(
                {
                    "source": food["text"],
                    "target": nutrient["text"],
                    "type": "CONTAINS",
                    "confidence": 0.75,
                }
            )

    return relationships
