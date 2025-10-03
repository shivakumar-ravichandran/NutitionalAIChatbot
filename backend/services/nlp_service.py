"""
NLP Service for entity extraction and text processing
"""

import spacy
from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NLPService:
    """Natural Language Processing service for nutritional text analysis"""

    def __init__(self):
        self.nlp = None
        self.custom_patterns = {}
        self.nutrition_entities = {}
        self._initialize_models()
        self._load_nutrition_knowledge()

    def _initialize_models(self):
        """Initialize spaCy models"""
        try:
            # Try to load the English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model successfully")
        except OSError:
            logger.warning("spaCy English model not found. Using blank model.")
            # Create a blank English model if the trained model isn't available
            self.nlp = spacy.blank("en")

        # Add custom pipeline components
        self._add_custom_components()

    def _add_custom_components(self):
        """Add custom NLP pipeline components"""

        # Add entity ruler for nutrition-specific entities
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")

            # Define nutrition patterns
            patterns = [
                # Nutrients
                {"label": "NUTRIENT", "pattern": [{"LOWER": "protein"}]},
                {"label": "NUTRIENT", "pattern": [{"LOWER": "carbohydrate"}]},
                {"label": "NUTRIENT", "pattern": [{"LOWER": "fat"}]},
                {"label": "NUTRIENT", "pattern": [{"LOWER": "fiber"}]},
                {
                    "label": "NUTRIENT",
                    "pattern": [
                        {"LOWER": "vitamin"},
                        {"TEXT": {"REGEX": r"[A-K]|[0-9]+"}},
                    ],
                },
                {"label": "NUTRIENT", "pattern": [{"LOWER": "calcium"}]},
                {"label": "NUTRIENT", "pattern": [{"LOWER": "iron"}]},
                {"label": "NUTRIENT", "pattern": [{"LOWER": "sodium"}]},
                {"label": "NUTRIENT", "pattern": [{"LOWER": "potassium"}]},
                # Food items
                {"label": "FOOD_ITEM", "pattern": [{"LOWER": "apple"}]},
                {"label": "FOOD_ITEM", "pattern": [{"LOWER": "banana"}]},
                {"label": "FOOD_ITEM", "pattern": [{"LOWER": "chicken"}]},
                {"label": "FOOD_ITEM", "pattern": [{"LOWER": "rice"}]},
                {"label": "FOOD_ITEM", "pattern": [{"LOWER": "bread"}]},
                {"label": "FOOD_ITEM", "pattern": [{"LOWER": "milk"}]},
                {"label": "FOOD_ITEM", "pattern": [{"LOWER": "egg"}]},
                # Health conditions
                {"label": "HEALTH_CONDITION", "pattern": [{"LOWER": "diabetes"}]},
                {"label": "HEALTH_CONDITION", "pattern": [{"LOWER": "hypertension"}]},
                {"label": "HEALTH_CONDITION", "pattern": [{"LOWER": "obesity"}]},
                {"label": "HEALTH_CONDITION", "pattern": [{"LOWER": "allergy"}]},
                # Measurements
                {
                    "label": "MEASUREMENT",
                    "pattern": [
                        {"LIKE_NUM": True},
                        {
                            "LOWER": {
                                "IN": ["g", "mg", "kg", "ml", "l", "cup", "tbsp", "tsp"]
                            }
                        },
                    ],
                },
                {
                    "label": "MEASUREMENT",
                    "pattern": [{"LIKE_NUM": True}, {"LOWER": "calories"}],
                },
            ]

            ruler.add_patterns(patterns)

    def _load_nutrition_knowledge(self):
        """Load nutrition knowledge base"""
        self.nutrition_entities = {
            "foods": {
                "fruits": [
                    "apple",
                    "banana",
                    "orange",
                    "grape",
                    "strawberry",
                    "mango",
                    "pineapple",
                ],
                "vegetables": [
                    "carrot",
                    "broccoli",
                    "spinach",
                    "tomato",
                    "onion",
                    "potato",
                    "bell pepper",
                ],
                "proteins": [
                    "chicken",
                    "fish",
                    "beef",
                    "egg",
                    "tofu",
                    "beans",
                    "lentils",
                ],
                "grains": [
                    "rice",
                    "bread",
                    "pasta",
                    "oats",
                    "quinoa",
                    "wheat",
                    "barley",
                ],
                "dairy": ["milk", "cheese", "yogurt", "butter", "cream"],
            },
            "nutrients": {
                "macronutrients": ["protein", "carbohydrate", "fat", "fiber"],
                "vitamins": [
                    "vitamin a",
                    "vitamin b",
                    "vitamin c",
                    "vitamin d",
                    "vitamin e",
                    "vitamin k",
                ],
                "minerals": [
                    "calcium",
                    "iron",
                    "sodium",
                    "potassium",
                    "magnesium",
                    "zinc",
                ],
            },
            "health_conditions": [
                "diabetes",
                "hypertension",
                "obesity",
                "anemia",
                "osteoporosis",
                "heart disease",
                "kidney disease",
                "liver disease",
            ],
        }

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract nutrition-related entities from text"""

        if not self.nlp:
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entity_info = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": self._calculate_confidence(ent),
            }
            entities.append(entity_info)

        # Add regex-based extraction for specific patterns
        regex_entities = self._extract_with_regex(text)
        entities.extend(regex_entities)

        return entities

    def _extract_with_regex(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns"""
        entities = []

        # Calorie patterns
        calorie_pattern = r"(\d+(?:\.\d+)?)\s*(?:cal|calories|kcal)"
        calorie_matches = re.finditer(calorie_pattern, text, re.IGNORECASE)
        for match in calorie_matches:
            entities.append(
                {
                    "text": match.group(),
                    "label": "CALORIE_VALUE",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9,
                }
            )

        # Serving size patterns
        serving_pattern = r"(\d+(?:\.\d+)?)\s*(cup|cups|tbsp|tsp|oz|lb|g|mg|ml|l)\b"
        serving_matches = re.finditer(serving_pattern, text, re.IGNORECASE)
        for match in serving_matches:
            entities.append(
                {
                    "text": match.group(),
                    "label": "SERVING_SIZE",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.85,
                }
            )

        # Percentage patterns (like "20% daily value")
        percentage_pattern = r"(\d+(?:\.\d+)?)\s*%\s*(?:daily|dv|recommended)"
        percentage_matches = re.finditer(percentage_pattern, text, re.IGNORECASE)
        for match in percentage_matches:
            entities.append(
                {
                    "text": match.group(),
                    "label": "DAILY_VALUE",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8,
                }
            )

        return entities

    def _calculate_confidence(self, entity) -> float:
        """Calculate confidence score for an entity"""
        # Simple confidence calculation based on entity type and length
        base_confidence = 0.7

        if entity.label_ in ["FOOD_ITEM", "NUTRIENT"]:
            base_confidence = 0.85
        elif entity.label_ in ["HEALTH_CONDITION"]:
            base_confidence = 0.8
        elif entity.label_ in ["MEASUREMENT"]:
            base_confidence = 0.9

        # Adjust based on entity length (longer entities might be more reliable)
        length_bonus = min(0.1, len(entity.text) * 0.01)

        return min(0.99, base_confidence + length_bonus)

    async def extract_relationships(
        self, entities: List[Dict[str, Any]], text: str
    ) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []

        # Create spaCy doc for dependency parsing
        doc = self.nlp(text)

        # Simple rule-based relationship extraction
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i >= j:  # Avoid duplicate pairs
                    continue

                relationship = self._find_relationship(entity1, entity2, doc)
                if relationship:
                    relationships.append(relationship)

        return relationships

    def _find_relationship(
        self, entity1: Dict[str, Any], entity2: Dict[str, Any], doc
    ) -> Optional[Dict[str, Any]]:
        """Find relationship between two entities"""

        # Food contains nutrient
        if entity1["label"] == "FOOD_ITEM" and entity2["label"] == "NUTRIENT":
            return {
                "source": entity1["text"],
                "target": entity2["text"],
                "relationship": "CONTAINS",
                "confidence": 0.75,
            }

        # Nutrient affects health condition
        if entity1["label"] == "NUTRIENT" and entity2["label"] == "HEALTH_CONDITION":
            return {
                "source": entity1["text"],
                "target": entity2["text"],
                "relationship": "AFFECTS",
                "confidence": 0.7,
            }

        # Food helps with health condition
        if entity1["label"] == "FOOD_ITEM" and entity2["label"] == "HEALTH_CONDITION":
            return {
                "source": entity1["text"],
                "target": entity2["text"],
                "relationship": "HELPS_WITH",
                "confidence": 0.65,
            }

        # Measurement describes food or nutrient
        if entity1["label"] in [
            "MEASUREMENT",
            "SERVING_SIZE",
            "CALORIE_VALUE",
        ] and entity2["label"] in ["FOOD_ITEM", "NUTRIENT"]:
            return {
                "source": entity2["text"],
                "target": entity1["text"],
                "relationship": "HAS_MEASUREMENT",
                "confidence": 0.8,
            }

        return None

    async def analyze_nutritional_content(self, text: str) -> Dict[str, Any]:
        """Analyze nutritional content of text and provide insights"""

        entities = await self.extract_entities(text)
        relationships = await self.extract_relationships(entities, text)

        # Categorize entities
        foods = [e for e in entities if e["label"] == "FOOD_ITEM"]
        nutrients = [e for e in entities if e["label"] == "NUTRIENT"]
        health_conditions = [e for e in entities if e["label"] == "HEALTH_CONDITION"]
        measurements = [
            e
            for e in entities
            if e["label"] in ["MEASUREMENT", "SERVING_SIZE", "CALORIE_VALUE"]
        ]

        # Generate insights
        insights = self._generate_insights(
            foods, nutrients, health_conditions, measurements
        )

        return {
            "entities": entities,
            "relationships": relationships,
            "categories": {
                "foods": foods,
                "nutrients": nutrients,
                "health_conditions": health_conditions,
                "measurements": measurements,
            },
            "insights": insights,
            "summary": {
                "total_entities": len(entities),
                "total_relationships": len(relationships),
                "food_items_found": len(foods),
                "nutrients_found": len(nutrients),
            },
        }

    def _generate_insights(
        self,
        foods: List[Dict],
        nutrients: List[Dict],
        health_conditions: List[Dict],
        measurements: List[Dict],
    ) -> List[str]:
        """Generate nutritional insights based on extracted entities"""
        insights = []

        if foods:
            food_names = [f["text"] for f in foods]
            insights.append(
                f"Identified {len(foods)} food items: {', '.join(food_names[:5])}"
            )

        if nutrients:
            nutrient_names = [n["text"] for n in nutrients]
            insights.append(
                f"Found {len(nutrients)} nutrients mentioned: {', '.join(nutrient_names[:5])}"
            )

        if health_conditions:
            condition_names = [h["text"] for h in health_conditions]
            insights.append(
                f"Health conditions referenced: {', '.join(condition_names)}"
            )

        if measurements:
            insights.append(
                f"Found {len(measurements)} nutritional measurements or serving sizes"
            )

        # Provide specific recommendations based on combinations
        if any("protein" in n["text"].lower() for n in nutrients) and foods:
            insights.append("Consider protein balance with the mentioned food items")

        if any("vitamin" in n["text"].lower() for n in nutrients):
            insights.append(
                "Vitamin content information detected - ensure adequate intake"
            )

        return insights

    def get_nutrition_suggestions(self, query: str) -> List[str]:
        """Get nutrition suggestions based on user query"""
        suggestions = []
        query_lower = query.lower()

        # Food-based suggestions
        for category, items in self.nutrition_entities["foods"].items():
            for item in items:
                if item in query_lower:
                    suggestions.append(f"Learn more about {item} nutrition facts")
                    suggestions.append(f"Find recipes with {item}")
                    break

        # Nutrient-based suggestions
        for category, items in self.nutrition_entities["nutrients"].items():
            for item in items:
                if item in query_lower:
                    suggestions.append(f"Foods rich in {item}")
                    suggestions.append(f"Daily requirements for {item}")
                    break

        # Health condition suggestions
        for condition in self.nutrition_entities["health_conditions"]:
            if condition in query_lower:
                suggestions.append(f"Diet recommendations for {condition}")
                suggestions.append(f"Foods to avoid with {condition}")
                break

        return suggestions[:5]  # Limit to 5 suggestions


# Global NLP service instance
nlp_service = None


def get_nlp_service() -> NLPService:
    """Get or create NLP service instance"""
    global nlp_service
    if nlp_service is None:
        nlp_service = NLPService()
    return nlp_service


async def initialize_nlp_service():
    """Initialize NLP service on startup"""
    global nlp_service
    try:
        nlp_service = NLPService()
        logger.info("NLP service initialized successfully")
        return nlp_service
    except Exception as e:
        logger.error(f"Failed to initialize NLP service: {e}")
        raise
