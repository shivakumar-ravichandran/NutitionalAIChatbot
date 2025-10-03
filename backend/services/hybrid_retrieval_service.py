"""
Hybrid Retrieval Service combining graph-based filtering with vector similarity search
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum
import hashlib

from services.graph_service import get_graph_service
from services.vector_service import get_vector_service

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """Different retrieval modes for hybrid search"""

    GRAPH_ONLY = "graph_only"
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval"""

    id: str
    content: str
    score: float
    source_type: str  # "graph", "vector", or "hybrid"
    metadata: Dict[str, Any]
    graph_score: Optional[float] = None
    vector_score: Optional[float] = None
    relevance_score: Optional[float] = None


@dataclass
class UserContext:
    """User context for personalized retrieval"""

    age: Optional[int] = None
    culture: Optional[str] = None
    dietary_preferences: Optional[str] = None
    allergies: List[str] = None
    health_conditions: List[str] = None
    activity_level: Optional[str] = None
    response_style: Optional[str] = None


class HybridRetrievalService:
    """Service for hybrid retrieval combining graph and vector search"""

    def __init__(self):
        self.graph_service = None
        self.vector_service = None
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)  # Cache results for 1 hour

        # Retrieval weights and parameters
        self.default_weights = {
            "graph_weight": 0.4,
            "vector_weight": 0.6,
            "relevance_boost": 0.1,
            "recency_boost": 0.05,
        }

        # Age-specific retrieval parameters
        self.age_parameters = {
            "children": {
                "graph_weight": 0.5,  # More structured info for children
                "vector_weight": 0.5,
                "simplicity_boost": 0.2,
                "safety_filter": True,
            },
            "adults": {
                "graph_weight": 0.4,
                "vector_weight": 0.6,
                "detail_boost": 0.1,
                "safety_filter": False,
            },
            "elderly": {
                "graph_weight": 0.6,  # More reliable structured info
                "vector_weight": 0.4,
                "clarity_boost": 0.15,
                "safety_filter": True,
            },
        }

        self._initialize_services()

    def _initialize_services(self):
        """Initialize graph and vector services"""
        try:
            self.graph_service = get_graph_service()
            self.vector_service = get_vector_service()
            logger.info("Hybrid retrieval service initialized")
        except Exception as e:
            logger.error(f"Error initializing hybrid retrieval service: {e}")

    def _get_cache_key(
        self, query: str, user_context: UserContext, k: int, mode: RetrievalMode
    ) -> str:
        """Generate cache key for retrieval request"""
        context_str = json.dumps(
            {
                "age": user_context.age,
                "culture": user_context.culture,
                "dietary_preferences": user_context.dietary_preferences,
                "allergies": user_context.allergies or [],
                "health_conditions": user_context.health_conditions or [],
            },
            sort_keys=True,
        )

        cache_input = f"{query}:{context_str}:{k}:{mode.value}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        if "timestamp" not in cache_entry:
            return False

        cached_time = datetime.fromisoformat(cache_entry["timestamp"])
        return datetime.now() - cached_time < self.cache_ttl

    async def hybrid_retrieve(
        self,
        query: str,
        user_context: UserContext,
        k: int = 10,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Main hybrid retrieval function"""

        # Check cache first
        cache_key = self._get_cache_key(query, user_context, k, mode)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            logger.info("Returning cached retrieval results")
            return [
                RetrievalResult(**result) for result in self.cache[cache_key]["results"]
            ]

        try:
            # Get age group for context
            age_group = self._get_age_group(user_context.age)
            retrieval_params = self.age_parameters.get(age_group, self.default_weights)

            # Determine retrieval mode
            if mode == RetrievalMode.ADAPTIVE:
                mode = self._select_adaptive_mode(query, user_context)

            # Execute retrieval based on mode
            if mode == RetrievalMode.GRAPH_ONLY:
                results = await self._graph_only_retrieve(
                    query, user_context, k, filters
                )
            elif mode == RetrievalMode.VECTOR_ONLY:
                results = await self._vector_only_retrieve(
                    query, user_context, k, filters
                )
            else:  # HYBRID
                results = await self._hybrid_retrieve_internal(
                    query, user_context, k, retrieval_params, filters
                )

            # Apply post-processing
            results = await self._post_process_results(results, query, user_context)

            # Cache results
            self.cache[cache_key] = {
                "results": [result.__dict__ for result in results],
                "timestamp": datetime.now().isoformat(),
            }

            return results

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return []

    def _get_age_group(self, age: Optional[int]) -> str:
        """Determine age group from age"""
        if not age:
            return "adults"

        if age < 18:
            return "children"
        elif age >= 65:
            return "elderly"
        else:
            return "adults"

    def _select_adaptive_mode(
        self, query: str, user_context: UserContext
    ) -> RetrievalMode:
        """Intelligently select retrieval mode based on query and context"""
        query_lower = query.lower()

        # Graph-heavy queries (specific foods, nutrients, conditions)
        graph_keywords = [
            "food",
            "nutrient",
            "vitamin",
            "mineral",
            "protein",
            "carbohydrate",
            "diabetes",
            "hypertension",
            "allergy",
            "contains",
            "benefits",
        ]

        # Vector-heavy queries (general advice, explanations, comparisons)
        vector_keywords = [
            "how",
            "why",
            "what",
            "explain",
            "compare",
            "difference",
            "advice",
            "tips",
            "recommend",
            "suggest",
            "best",
        ]

        graph_score = sum(1 for keyword in graph_keywords if keyword in query_lower)
        vector_score = sum(1 for keyword in vector_keywords if keyword in query_lower)

        if graph_score > vector_score * 1.5:
            return RetrievalMode.GRAPH_ONLY
        elif vector_score > graph_score * 1.5:
            return RetrievalMode.VECTOR_ONLY
        else:
            return RetrievalMode.HYBRID

    async def _graph_only_retrieve(
        self,
        query: str,
        user_context: UserContext,
        k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """Retrieve using only graph-based search"""

        if not self.graph_service or not self.graph_service.is_connected():
            logger.warning("Graph service not available, falling back to vector search")
            return await self._vector_only_retrieve(query, user_context, k, filters)

        results = []

        try:
            # Extract entities from query for graph filtering
            query_entities = await self._extract_query_entities(query)

            # Apply user context filters
            graph_filters = self._build_graph_filters(user_context, filters)

            # Search for foods if food entities detected
            if any(entity["type"] == "FOOD_ITEM" for entity in query_entities):
                food_names = [
                    e["text"] for e in query_entities if e["type"] == "FOOD_ITEM"
                ]
                for food_name in food_names:
                    nutrients = await self.graph_service.find_nutrients_in_food(
                        food_name
                    )
                    for nutrient in nutrients:
                        results.append(
                            RetrievalResult(
                                id=f"food_{food_name}_{nutrient['nutrient_name']}",
                                content=f"{food_name} contains {nutrient['amount']}{nutrient['unit']} of {nutrient['nutrient_name']}",
                                score=nutrient.get("confidence", 0.8),
                                source_type="graph",
                                metadata={
                                    "food": food_name,
                                    "nutrient": nutrient["nutrient_name"],
                                    "amount": nutrient["amount"],
                                    "unit": nutrient["unit"],
                                },
                                graph_score=nutrient.get("confidence", 0.8),
                            )
                        )

            # Search for nutrients if nutrient entities detected
            if any(entity["type"] == "NUTRIENT" for entity in query_entities):
                nutrient_names = [
                    e["text"] for e in query_entities if e["type"] == "NUTRIENT"
                ]
                for nutrient_name in nutrient_names:
                    foods = await self.graph_service.find_foods_with_nutrient(
                        nutrient_name
                    )
                    for food in foods:
                        results.append(
                            RetrievalResult(
                                id=f"nutrient_{nutrient_name}_{food['food_name']}",
                                content=f"{food['food_name']} is a good source of {nutrient_name} with {food['amount']}{food['unit']}",
                                score=food.get("confidence", 0.8),
                                source_type="graph",
                                metadata={
                                    "food": food["food_name"],
                                    "nutrient": nutrient_name,
                                    "amount": food["amount"],
                                    "unit": food["unit"],
                                },
                                graph_score=food.get("confidence", 0.8),
                            )
                        )

            # Search for health condition recommendations
            if user_context.health_conditions:
                for condition in user_context.health_conditions:
                    recommendations = await self.graph_service.get_food_recommendations(
                        condition
                    )
                    for rec in recommendations:
                        results.append(
                            RetrievalResult(
                                id=f"health_{condition}_{rec['food_name']}",
                                content=f"{rec['food_name']} is beneficial for {condition}",
                                score=rec.get("confidence", 0.7),
                                source_type="graph",
                                metadata={
                                    "food": rec["food_name"],
                                    "health_condition": condition,
                                    "effectiveness": rec.get("effectiveness", 0.7),
                                },
                                graph_score=rec.get("confidence", 0.7),
                            )
                        )

            # Sort by score and limit results
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:k]

        except Exception as e:
            logger.error(f"Error in graph-only retrieval: {e}")
            return []

    async def _vector_only_retrieve(
        self,
        query: str,
        user_context: UserContext,
        k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """Retrieve using only vector-based search"""

        if not self.vector_service:
            logger.warning("Vector service not available")
            return []

        try:
            # Enhance query with user context
            enhanced_query = self._enhance_query_with_context(query, user_context)

            # Perform vector search
            similar_docs = await self.vector_service.search_similar(
                enhanced_query, k, threshold=0.1
            )

            # Convert to RetrievalResult format
            results = []
            for doc in similar_docs:
                results.append(
                    RetrievalResult(
                        id=doc["id"],
                        content=doc["text"],
                        score=doc["score"],
                        source_type="vector",
                        metadata=doc.get("metadata", {}),
                        vector_score=doc["score"],
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Error in vector-only retrieval: {e}")
            return []

    async def _hybrid_retrieve_internal(
        self,
        query: str,
        user_context: UserContext,
        k: int,
        retrieval_params: Dict[str, float],
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """Internal hybrid retrieval combining graph and vector results"""

        try:
            # Run graph and vector retrieval in parallel
            graph_task = self._graph_only_retrieve(query, user_context, k * 2, filters)
            vector_task = self._vector_only_retrieve(
                query, user_context, k * 2, filters
            )

            graph_results, vector_results = await asyncio.gather(
                graph_task, vector_task
            )

            # Combine and rank results
            combined_results = self._combine_and_rank_results(
                graph_results, vector_results, retrieval_params, k
            )

            return combined_results

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return []

    def _combine_and_rank_results(
        self,
        graph_results: List[RetrievalResult],
        vector_results: List[RetrievalResult],
        params: Dict[str, float],
        k: int,
    ) -> List[RetrievalResult]:
        """Combine and rank graph and vector results"""

        # Create lookup for duplicate detection
        combined_results = {}

        # Add graph results
        for result in graph_results:
            combined_results[result.id] = result
            result.source_type = "graph"

        # Add vector results (merge if duplicate)
        for result in vector_results:
            if result.id in combined_results:
                # Merge scores for hybrid result
                existing = combined_results[result.id]
                existing.vector_score = result.score
                existing.source_type = "hybrid"

                # Calculate combined score
                existing.score = (
                    params["graph_weight"] * (existing.graph_score or 0)
                    + params["vector_weight"] * result.score
                )
            else:
                # Add as vector-only result
                combined_results[result.id] = result
                result.source_type = "vector"

        # Convert to list and sort by combined score
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)

        return final_results[:k]

    async def _extract_query_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities from query for graph filtering"""
        try:
            # Try to use NLP service if available
            from services.nlp_service import get_nlp_service

            nlp_service = get_nlp_service()

            if nlp_service:
                return await nlp_service.extract_entities(query)
            else:
                # Fallback entity extraction
                return self._fallback_entity_extraction(query)

        except Exception as e:
            logger.error(f"Error extracting query entities: {e}")
            return self._fallback_entity_extraction(query)

    def _fallback_entity_extraction(self, query: str) -> List[Dict[str, Any]]:
        """Simple fallback entity extraction"""
        entities = []
        query_lower = query.lower()

        # Simple keyword matching
        food_keywords = ["apple", "banana", "chicken", "rice", "milk", "egg", "fish"]
        nutrient_keywords = ["protein", "vitamin", "calcium", "iron", "fiber"]
        health_keywords = ["diabetes", "hypertension", "obesity", "allergy"]

        for food in food_keywords:
            if food in query_lower:
                entities.append({"text": food, "type": "FOOD_ITEM", "confidence": 0.8})

        for nutrient in nutrient_keywords:
            if nutrient in query_lower:
                entities.append(
                    {"text": nutrient, "type": "NUTRIENT", "confidence": 0.8}
                )

        for condition in health_keywords:
            if condition in query_lower:
                entities.append(
                    {"text": condition, "type": "HEALTH_CONDITION", "confidence": 0.8}
                )

        return entities

    def _build_graph_filters(
        self, user_context: UserContext, additional_filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build graph filters based on user context"""
        filters = {}

        if user_context.dietary_preferences:
            filters["dietary_preference"] = user_context.dietary_preferences

        if user_context.allergies:
            filters["exclude_allergens"] = user_context.allergies

        if user_context.culture:
            filters["culture"] = user_context.culture

        if additional_filters:
            filters.update(additional_filters)

        return filters

    def _enhance_query_with_context(self, query: str, user_context: UserContext) -> str:
        """Enhance query with user context for better vector search"""
        enhancements = []

        if user_context.dietary_preferences:
            enhancements.append(
                f"dietary preference: {user_context.dietary_preferences}"
            )

        if user_context.culture:
            enhancements.append(f"culture: {user_context.culture}")

        if user_context.health_conditions:
            enhancements.append(
                f"health conditions: {', '.join(user_context.health_conditions)}"
            )

        if user_context.allergies:
            enhancements.append(f"allergies: {', '.join(user_context.allergies)}")

        if enhancements:
            enhanced_query = f"{query} Context: {' '.join(enhancements)}"
            return enhanced_query

        return query

    async def _post_process_results(
        self, results: List[RetrievalResult], query: str, user_context: UserContext
    ) -> List[RetrievalResult]:
        """Post-process results for relevance and safety"""

        processed_results = []

        for result in results:
            # Calculate relevance score
            relevance_score = await self._calculate_relevance_score(
                result, query, user_context
            )
            result.relevance_score = relevance_score

            # Apply safety filtering if needed
            if self._should_apply_safety_filter(user_context):
                if not self._passes_safety_filter(result, user_context):
                    continue

            # Apply age-appropriate filtering
            if not self._is_age_appropriate(result, user_context):
                continue

            processed_results.append(result)

        # Re-rank based on relevance scores
        processed_results.sort(key=lambda x: x.relevance_score or x.score, reverse=True)

        return processed_results

    async def _calculate_relevance_score(
        self, result: RetrievalResult, query: str, user_context: UserContext
    ) -> float:
        """Calculate relevance score considering user context"""
        base_score = result.score

        # Boost score for user-specific matches
        relevance_boost = 0.0
        content_lower = result.content.lower()

        # Dietary preference boost
        if user_context.dietary_preferences:
            if user_context.dietary_preferences.lower() in content_lower:
                relevance_boost += 0.1

        # Culture-specific boost
        if user_context.culture:
            if user_context.culture.lower() in content_lower:
                relevance_boost += 0.1

        # Health condition relevance
        if user_context.health_conditions:
            for condition in user_context.health_conditions:
                if condition.lower() in content_lower:
                    relevance_boost += 0.15
                    break

        # Allergy safety boost (negative if contains allergens)
        if user_context.allergies:
            for allergy in user_context.allergies:
                if allergy.lower() in content_lower:
                    relevance_boost -= 0.2  # Penalize results containing allergens
                    break

        return min(1.0, base_score + relevance_boost)

    def _should_apply_safety_filter(self, user_context: UserContext) -> bool:
        """Determine if safety filtering should be applied"""
        age_group = self._get_age_group(user_context.age)
        return age_group in ["children", "elderly"] or bool(
            user_context.health_conditions
        )

    def _passes_safety_filter(
        self, result: RetrievalResult, user_context: UserContext
    ) -> bool:
        """Check if result passes safety filters"""
        content_lower = result.content.lower()

        # Check for allergen mentions
        if user_context.allergies:
            for allergy in user_context.allergies:
                if allergy.lower() in content_lower:
                    return False

        # Add more safety checks as needed
        unsafe_keywords = ["raw", "unpasteurized", "excessive", "overdose"]
        if any(keyword in content_lower for keyword in unsafe_keywords):
            return False

        return True

    def _is_age_appropriate(
        self, result: RetrievalResult, user_context: UserContext
    ) -> bool:
        """Check if result is appropriate for user's age"""
        if not user_context.age:
            return True

        content_lower = result.content.lower()
        age_group = self._get_age_group(user_context.age)

        # Age-specific filtering
        if age_group == "children":
            # Avoid complex medical terms for children
            complex_terms = ["pharmacokinetics", "bioavailability", "metabolism"]
            if any(term in content_lower for term in complex_terms):
                return False

        return True

    async def clear_cache(self):
        """Clear the retrieval cache"""
        self.cache.clear()
        logger.info("Hybrid retrieval cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        valid_entries = sum(
            1 for entry in self.cache.values() if self._is_cache_valid(entry)
        )

        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "cache_hit_ratio": valid_entries / max(1, len(self.cache)),
            "cache_ttl_hours": self.cache_ttl.total_seconds() / 3600,
        }


# Global hybrid retrieval service instance
hybrid_retrieval_service = None


def get_hybrid_retrieval_service() -> HybridRetrievalService:
    """Get or create hybrid retrieval service instance"""
    global hybrid_retrieval_service
    if hybrid_retrieval_service is None:
        hybrid_retrieval_service = HybridRetrievalService()
    return hybrid_retrieval_service


async def initialize_hybrid_retrieval_service() -> HybridRetrievalService:
    """Initialize hybrid retrieval service on startup"""
    global hybrid_retrieval_service
    try:
        hybrid_retrieval_service = HybridRetrievalService()
        logger.info("Hybrid retrieval service initialized")
        return hybrid_retrieval_service
    except Exception as e:
        logger.error(f"Failed to initialize hybrid retrieval service: {e}")
        # Return service anyway for fallback functionality
        hybrid_retrieval_service = HybridRetrievalService()
        return hybrid_retrieval_service
