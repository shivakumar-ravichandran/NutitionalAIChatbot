"""
Knowledge Base API endpoints for search, entities, and graph operations
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any, Union
import asyncio
from datetime import datetime

from database.database import get_db
from database.models import UserProfile, ExtractedEntity, EntityRelationship, Document
from schemas import (
    SearchQuery,
    SearchResult,
    SearchResponse,
    EntityResponse,
    RelationshipResponse,
)

# Import services
from services.hybrid_retrieval_service import (
    get_hybrid_retrieval_service,
    UserContext,
    RetrievalMode,
)
from services.vector_service import get_vector_service
from services.graph_service import get_graph_service
from services.nlp_service import get_nlp_service

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_knowledge_base(
    search_query: SearchQuery,
    profile_uuid: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Search the knowledge base using hybrid retrieval"""

    start_time = datetime.now()

    try:
        # Get user profile for personalized search
        profile = None
        if profile_uuid:
            profile = (
                db.query(UserProfile)
                .filter(UserProfile.profile_uuid == profile_uuid)
                .first()
            )

        # Build user context
        user_context = UserContext(
            age=profile.age if profile else None,
            culture=profile.culture if profile else None,
            dietary_preferences=profile.dietary_preferences if profile else None,
            allergies=(
                [allergy.allergy for allergy in profile.allergies]
                if profile and profile.allergies
                else []
            ),
            health_conditions=(
                [condition.condition_name for condition in profile.health_conditions]
                if profile and profile.health_conditions
                else []
            ),
            activity_level=profile.activity_level if profile else None,
            response_style=profile.response_style if profile else None,
        )

        # Perform hybrid search
        hybrid_service = get_hybrid_retrieval_service()
        retrieval_results = await hybrid_service.hybrid_retrieve(
            query=search_query.query,
            user_context=user_context,
            k=search_query.limit,
            mode=RetrievalMode.HYBRID,
            filters=search_query.filters,
        )

        # Convert to search results format
        search_results = []
        for result in retrieval_results:
            search_results.append(
                SearchResult(
                    content=result.content,
                    score=result.score,
                    source=result.source_type,
                    entity_type=result.metadata.get("entity_type"),
                    metadata=result.metadata,
                )
            )

        # Calculate search time
        search_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return SearchResponse(
            query=search_query.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time_ms,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching knowledge base: {str(e)}",
        )


@router.get("/search/vector", response_model=List[SearchResult])
async def vector_search(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Number of results"),
    threshold: float = Query(
        0.0, ge=0.0, le=1.0, description="Minimum similarity threshold"
    ),
):
    """Perform vector similarity search"""

    try:
        vector_service = get_vector_service()
        similar_docs = await vector_service.search_similar(query, limit, threshold)

        results = []
        for doc in similar_docs:
            results.append(
                SearchResult(
                    content=doc["text"],
                    score=doc["score"],
                    source="vector",
                    metadata=doc.get("metadata", {}),
                )
            )

        return results

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in vector search: {str(e)}",
        )


@router.get("/search/graph", response_model=List[SearchResult])
async def graph_search(
    entity: str = Query(..., description="Entity to search for"),
    entity_type: str = Query("FOOD_ITEM", description="Type of entity"),
    relationship: Optional[str] = Query(None, description="Relationship type"),
    limit: int = Query(10, ge=1, le=100),
):
    """Search the knowledge graph for entities and relationships"""

    try:
        graph_service = get_graph_service()
        results = []

        if entity_type.upper() == "FOOD_ITEM" and not relationship:
            # Find nutrients in food
            nutrients = await graph_service.find_nutrients_in_food(entity)
            for nutrient in nutrients:
                results.append(
                    SearchResult(
                        content=f"{entity} contains {nutrient['amount']}{nutrient['unit']} of {nutrient['nutrient_name']}",
                        score=nutrient.get("confidence", 0.8),
                        source="graph",
                        entity_type="NUTRIENT_CONTENT",
                        metadata={
                            "food": entity,
                            "nutrient": nutrient["nutrient_name"],
                            "amount": nutrient["amount"],
                            "unit": nutrient["unit"],
                        },
                    )
                )

        elif entity_type.upper() == "NUTRIENT" and not relationship:
            # Find foods with nutrient
            foods = await graph_service.find_foods_with_nutrient(entity)
            for food in foods:
                results.append(
                    SearchResult(
                        content=f"{food['food_name']} is a good source of {entity} with {food['amount']}{food['unit']}",
                        score=food.get("confidence", 0.8),
                        source="graph",
                        entity_type="FOOD_NUTRIENT",
                        metadata={
                            "food": food["food_name"],
                            "nutrient": entity,
                            "amount": food["amount"],
                            "unit": food["unit"],
                        },
                    )
                )

        elif entity_type.upper() == "HEALTH_CONDITION":
            # Find food recommendations for health condition
            recommendations = await graph_service.get_food_recommendations(entity)
            for rec in recommendations:
                results.append(
                    SearchResult(
                        content=f"{rec['food_name']} is beneficial for {entity}",
                        score=rec.get("confidence", 0.7),
                        source="graph",
                        entity_type="HEALTH_RECOMMENDATION",
                        metadata={
                            "food": rec["food_name"],
                            "health_condition": entity,
                            "effectiveness": rec.get("effectiveness", 0.7),
                        },
                    )
                )

        return results[:limit]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in graph search: {str(e)}",
        )


@router.get("/entities", response_model=List[EntityResponse])
async def list_entities(
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """List extracted entities from documents"""

    query = db.query(ExtractedEntity)

    if entity_type:
        query = query.filter(ExtractedEntity.entity_type == entity_type.upper())

    entities = query.offset(skip).limit(limit).all()
    return entities


@router.get("/entities/types")
async def get_entity_types(db: Session = Depends(get_db)):
    """Get all available entity types"""

    types = db.query(ExtractedEntity.entity_type).distinct().all()

    return [t[0] for t in types]


@router.get("/entities/{entity_text}", response_model=List[EntityResponse])
async def get_entity_details(entity_text: str, db: Session = Depends(get_db)):
    """Get detailed information about a specific entity"""

    entities = (
        db.query(ExtractedEntity)
        .filter(ExtractedEntity.entity_text.ilike(f"%{entity_text}%"))
        .all()
    )

    if not entities:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Entity not found"
        )

    return entities


@router.get("/relationships", response_model=List[RelationshipResponse])
async def list_relationships(
    relationship_type: Optional[str] = Query(
        None, description="Filter by relationship type"
    ),
    source_entity: Optional[str] = Query(None, description="Filter by source entity"),
    target_entity: Optional[str] = Query(None, description="Filter by target entity"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """List entity relationships"""

    query = db.query(EntityRelationship)

    if relationship_type:
        query = query.filter(
            EntityRelationship.relationship_type == relationship_type.upper()
        )

    if source_entity:
        query = query.filter(
            EntityRelationship.source_entity.ilike(f"%{source_entity}%")
        )

    if target_entity:
        query = query.filter(
            EntityRelationship.target_entity.ilike(f"%{target_entity}%")
        )

    relationships = query.offset(skip).limit(limit).all()
    return relationships


@router.get("/relationships/types")
async def get_relationship_types(db: Session = Depends(get_db)):
    """Get all available relationship types"""

    types = db.query(EntityRelationship.relationship_type).distinct().all()

    return [t[0] for t in types]


@router.get("/graph/visualize/{entity}")
async def get_graph_visualization_data(
    entity: str,
    depth: int = Query(2, ge=1, le=3, description="Graph traversal depth"),
    max_nodes: int = Query(50, ge=10, le=200, description="Maximum nodes to return"),
):
    """Get graph visualization data for an entity"""

    try:
        graph_service = get_graph_service()

        # This would typically query Neo4j for graph structure
        # For now, return a simplified structure
        nodes = []
        edges = []

        # Add central node
        nodes.append({"id": entity, "label": entity, "type": "central", "size": 30})

        # Find related entities based on type
        if any(
            food in entity.lower() for food in ["apple", "banana", "chicken", "rice"]
        ):
            # Food item - find nutrients
            nutrients = await graph_service.find_nutrients_in_food(entity)
            for nutrient in nutrients[:10]:  # Limit to prevent overcrowding
                nutrient_name = nutrient["nutrient_name"]
                nodes.append(
                    {
                        "id": nutrient_name,
                        "label": nutrient_name,
                        "type": "nutrient",
                        "size": 20,
                    }
                )
                edges.append(
                    {
                        "source": entity,
                        "target": nutrient_name,
                        "relationship": "CONTAINS",
                        "weight": nutrient.get("confidence", 0.8),
                    }
                )

        elif any(
            nutrient in entity.lower() for nutrient in ["protein", "vitamin", "calcium"]
        ):
            # Nutrient - find foods
            foods = await graph_service.find_foods_with_nutrient(entity)
            for food in foods[:10]:
                food_name = food["food_name"]
                nodes.append(
                    {"id": food_name, "label": food_name, "type": "food", "size": 20}
                )
                edges.append(
                    {
                        "source": food_name,
                        "target": entity,
                        "relationship": "CONTAINS",
                        "weight": food.get("confidence", 0.8),
                    }
                )

        return {
            "nodes": nodes,
            "edges": edges,
            "center_entity": entity,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating graph visualization: {str(e)}",
        )


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_text(
    text: str,
    extract_entities: bool = Query(True, description="Extract entities from text"),
    find_relationships: bool = Query(
        True, description="Find relationships between entities"
    ),
    generate_insights: bool = Query(True, description="Generate nutritional insights"),
):
    """Analyze text for nutritional content and insights"""

    try:
        nlp_service = get_nlp_service()

        if not nlp_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NLP service not available",
            )

        analysis_result = await nlp_service.analyze_nutritional_content(text)

        # Filter results based on request parameters
        result = {"text": text}

        if extract_entities:
            result["entities"] = analysis_result["entities"]
            result["categories"] = analysis_result["categories"]

        if find_relationships:
            result["relationships"] = analysis_result["relationships"]

        if generate_insights:
            result["insights"] = analysis_result["insights"]
            result["summary"] = analysis_result["summary"]

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing text: {str(e)}",
        )


@router.get("/stats")
async def get_knowledge_base_stats(db: Session = Depends(get_db)):
    """Get statistics about the knowledge base"""

    try:
        # Database stats
        total_documents = db.query(Document).count()
        total_entities = db.query(ExtractedEntity).count()
        total_relationships = db.query(EntityRelationship).count()

        entity_type_counts = {}
        entity_types = (
            db.query(
                ExtractedEntity.entity_type, db.func.count(ExtractedEntity.entity_type)
            )
            .group_by(ExtractedEntity.entity_type)
            .all()
        )
        for entity_type, count in entity_types:
            entity_type_counts[entity_type] = count

        relationship_type_counts = {}
        relationship_types = (
            db.query(
                EntityRelationship.relationship_type,
                db.func.count(EntityRelationship.relationship_type),
            )
            .group_by(EntityRelationship.relationship_type)
            .all()
        )
        for rel_type, count in relationship_types:
            relationship_type_counts[rel_type] = count

        # Vector service stats
        vector_service = get_vector_service()
        vector_stats = vector_service.get_index_stats()

        # Graph service stats
        graph_service = get_graph_service()
        graph_connected = graph_service.is_connected()

        # Hybrid retrieval stats
        hybrid_service = get_hybrid_retrieval_service()
        cache_stats = hybrid_service.get_cache_stats()

        return {
            "database": {
                "total_documents": total_documents,
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "entity_types": entity_type_counts,
                "relationship_types": relationship_type_counts,
            },
            "vector_index": vector_stats,
            "graph_database": {
                "connected": graph_connected,
                "service_available": graph_service is not None,
            },
            "hybrid_retrieval": cache_stats,
            "last_updated": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting knowledge base stats: {str(e)}",
        )


@router.post("/rebuild")
async def rebuild_knowledge_base():
    """Rebuild knowledge base indices and caches"""

    try:
        tasks = []
        results = {}

        # Rebuild vector index
        vector_service = get_vector_service()
        if vector_service:
            tasks.append(("vector_index", vector_service.rebuild_index()))

        # Clear hybrid retrieval cache
        hybrid_service = get_hybrid_retrieval_service()
        if hybrid_service:
            tasks.append(("hybrid_cache", hybrid_service.clear_cache()))

        # Execute all rebuild tasks
        for task_name, task in tasks:
            try:
                await task
                results[task_name] = "success"
            except Exception as e:
                results[task_name] = f"error: {str(e)}"

        return {
            "message": "Knowledge base rebuild initiated",
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error rebuilding knowledge base: {str(e)}",
        )
