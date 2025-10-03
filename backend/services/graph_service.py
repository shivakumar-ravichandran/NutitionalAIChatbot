"""
Graph database service for knowledge graph management
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json

# Import neo4j when available
try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

logger = logging.getLogger(__name__)


class GraphService:
    """Service for managing nutrition knowledge graph with Neo4j"""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._connected = False

        if NEO4J_AVAILABLE:
            self._connect()
        else:
            logger.warning(
                "Neo4j driver not available. Graph functionality will be limited."
            )

    def _connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            self._connected = True
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to Neo4j"""
        return self._connected and self.driver is not None

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self._connected = False

    async def create_food_node(
        self, food_name: str, properties: Dict[str, Any] = None
    ) -> bool:
        """Create a food node in the knowledge graph"""
        if not self.is_connected():
            logger.warning("Neo4j not connected. Using fallback storage.")
            return await self._fallback_create_node("FOOD", food_name, properties)

        try:
            with self.driver.session() as session:
                query = """
                MERGE (f:Food {name: $name})
                ON CREATE SET f.created_at = datetime()
                ON MATCH SET f.updated_at = datetime()
                """

                if properties:
                    # Dynamically add properties
                    set_clause = ", ".join(
                        [f"f.{key} = ${key}" for key in properties.keys()]
                    )
                    query += f", {set_clause}"

                params = {"name": food_name}
                if properties:
                    params.update(properties)

                session.run(query, **params)
                logger.info(f"Created/updated food node: {food_name}")
                return True

        except Exception as e:
            logger.error(f"Error creating food node: {e}")
            return False

    async def create_nutrient_node(
        self, nutrient_name: str, properties: Dict[str, Any] = None
    ) -> bool:
        """Create a nutrient node in the knowledge graph"""
        if not self.is_connected():
            return await self._fallback_create_node(
                "NUTRIENT", nutrient_name, properties
            )

        try:
            with self.driver.session() as session:
                query = """
                MERGE (n:Nutrient {name: $name})
                ON CREATE SET n.created_at = datetime()
                ON MATCH SET n.updated_at = datetime()
                """

                if properties:
                    set_clause = ", ".join(
                        [f"n.{key} = ${key}" for key in properties.keys()]
                    )
                    query += f", {set_clause}"

                params = {"name": nutrient_name}
                if properties:
                    params.update(properties)

                session.run(query, **params)
                logger.info(f"Created/updated nutrient node: {nutrient_name}")
                return True

        except Exception as e:
            logger.error(f"Error creating nutrient node: {e}")
            return False

    async def create_relationship(
        self,
        source_name: str,
        source_type: str,
        target_name: str,
        target_type: str,
        relationship_type: str,
        properties: Dict[str, Any] = None,
    ) -> bool:
        """Create a relationship between two nodes"""
        if not self.is_connected():
            return await self._fallback_create_relationship(
                source_name,
                source_type,
                target_name,
                target_type,
                relationship_type,
                properties,
            )

        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (s:{source_type} {{name: $source_name}})
                MATCH (t:{target_type} {{name: $target_name}})
                MERGE (s)-[r:{relationship_type}]->(t)
                ON CREATE SET r.created_at = datetime()
                ON MATCH SET r.updated_at = datetime()
                """

                if properties:
                    set_clause = ", ".join(
                        [f"r.{key} = ${key}" for key in properties.keys()]
                    )
                    query += f", {set_clause}"

                params = {"source_name": source_name, "target_name": target_name}
                if properties:
                    params.update(properties)

                session.run(query, **params)
                logger.info(
                    f"Created relationship: {source_name} -{relationship_type}-> {target_name}"
                )
                return True

        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            return False

    async def find_foods_with_nutrient(
        self, nutrient_name: str
    ) -> List[Dict[str, Any]]:
        """Find all foods that contain a specific nutrient"""
        if not self.is_connected():
            return await self._fallback_find_foods_with_nutrient(nutrient_name)

        try:
            with self.driver.session() as session:
                query = """
                MATCH (f:Food)-[r:CONTAINS]->(n:Nutrient {name: $nutrient_name})
                RETURN f.name as food_name, r.amount as amount, r.unit as unit, r.confidence as confidence
                ORDER BY r.amount DESC
                """

                result = session.run(query, nutrient_name=nutrient_name)
                foods = []

                for record in result:
                    foods.append(
                        {
                            "food_name": record["food_name"],
                            "amount": record.get("amount"),
                            "unit": record.get("unit"),
                            "confidence": record.get("confidence", 0.8),
                        }
                    )

                return foods

        except Exception as e:
            logger.error(f"Error finding foods with nutrient: {e}")
            return []

    async def find_nutrients_in_food(self, food_name: str) -> List[Dict[str, Any]]:
        """Find all nutrients in a specific food"""
        if not self.is_connected():
            return await self._fallback_find_nutrients_in_food(food_name)

        try:
            with self.driver.session() as session:
                query = """
                MATCH (f:Food {name: $food_name})-[r:CONTAINS]->(n:Nutrient)
                RETURN n.name as nutrient_name, r.amount as amount, r.unit as unit, r.confidence as confidence
                ORDER BY r.amount DESC
                """

                result = session.run(query, food_name=food_name)
                nutrients = []

                for record in result:
                    nutrients.append(
                        {
                            "nutrient_name": record["nutrient_name"],
                            "amount": record.get("amount"),
                            "unit": record.get("unit"),
                            "confidence": record.get("confidence", 0.8),
                        }
                    )

                return nutrients

        except Exception as e:
            logger.error(f"Error finding nutrients in food: {e}")
            return []

    async def get_food_recommendations(
        self, health_condition: str
    ) -> List[Dict[str, Any]]:
        """Get food recommendations for a health condition"""
        if not self.is_connected():
            return await self._fallback_get_recommendations(health_condition)

        try:
            with self.driver.session() as session:
                query = """
                MATCH (h:HealthCondition {name: $condition})<-[r:HELPS_WITH]-(f:Food)
                RETURN f.name as food_name, r.effectiveness as effectiveness, r.confidence as confidence
                ORDER BY r.effectiveness DESC
                LIMIT 10
                """

                result = session.run(query, condition=health_condition)
                recommendations = []

                for record in result:
                    recommendations.append(
                        {
                            "food_name": record["food_name"],
                            "effectiveness": record.get("effectiveness", 0.7),
                            "confidence": record.get("confidence", 0.8),
                        }
                    )

                return recommendations

        except Exception as e:
            logger.error(f"Error getting food recommendations: {e}")
            return []

    async def store_document_entities(
        self,
        document_id: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> bool:
        """Store extracted entities and relationships from a document"""
        try:
            # Create nodes for entities
            for entity in entities:
                if entity["label"] == "FOOD_ITEM":
                    await self.create_food_node(
                        entity["text"],
                        {
                            "document_id": document_id,
                            "confidence": entity.get("confidence", 0.8),
                        },
                    )
                elif entity["label"] == "NUTRIENT":
                    await self.create_nutrient_node(
                        entity["text"],
                        {
                            "document_id": document_id,
                            "confidence": entity.get("confidence", 0.8),
                        },
                    )

            # Create relationships
            for rel in relationships:
                source_type = (
                    "Food" if rel.get("source_type") == "FOOD_ITEM" else "Nutrient"
                )
                target_type = (
                    "Food" if rel.get("target_type") == "FOOD_ITEM" else "Nutrient"
                )

                await self.create_relationship(
                    rel["source"],
                    source_type,
                    rel["target"],
                    target_type,
                    rel["relationship"],
                    {
                        "document_id": document_id,
                        "confidence": rel.get("confidence", 0.7),
                    },
                )

            return True

        except Exception as e:
            logger.error(f"Error storing document entities: {e}")
            return False

    # Fallback methods for when Neo4j is not available

    async def _fallback_create_node(
        self, node_type: str, name: str, properties: Dict[str, Any] = None
    ) -> bool:
        """Fallback node creation (in-memory or file-based)"""
        logger.info(
            f"Fallback: Would create {node_type} node '{name}' with properties: {properties}"
        )
        return True

    async def _fallback_create_relationship(
        self,
        source_name: str,
        source_type: str,
        target_name: str,
        target_type: str,
        relationship_type: str,
        properties: Dict[str, Any] = None,
    ) -> bool:
        """Fallback relationship creation"""
        logger.info(
            f"Fallback: Would create relationship {source_name} -{relationship_type}-> {target_name}"
        )
        return True

    async def _fallback_find_foods_with_nutrient(
        self, nutrient_name: str
    ) -> List[Dict[str, Any]]:
        """Fallback method to find foods with nutrient"""
        # Simple hardcoded knowledge for demo
        nutrition_data = {
            "protein": [
                {"food_name": "chicken", "amount": 31, "unit": "g", "confidence": 0.9},
                {"food_name": "egg", "amount": 13, "unit": "g", "confidence": 0.9},
                {"food_name": "fish", "amount": 25, "unit": "g", "confidence": 0.85},
            ],
            "vitamin c": [
                {"food_name": "orange", "amount": 70, "unit": "mg", "confidence": 0.9},
                {
                    "food_name": "strawberry",
                    "amount": 85,
                    "unit": "mg",
                    "confidence": 0.85,
                },
                {
                    "food_name": "broccoli",
                    "amount": 90,
                    "unit": "mg",
                    "confidence": 0.8,
                },
            ],
            "calcium": [
                {"food_name": "milk", "amount": 300, "unit": "mg", "confidence": 0.95},
                {"food_name": "cheese", "amount": 200, "unit": "mg", "confidence": 0.9},
                {
                    "food_name": "yogurt",
                    "amount": 150,
                    "unit": "mg",
                    "confidence": 0.85,
                },
            ],
        }

        return nutrition_data.get(nutrient_name.lower(), [])

    async def _fallback_find_nutrients_in_food(
        self, food_name: str
    ) -> List[Dict[str, Any]]:
        """Fallback method to find nutrients in food"""
        food_data = {
            "apple": [
                {
                    "nutrient_name": "vitamin c",
                    "amount": 14,
                    "unit": "mg",
                    "confidence": 0.9,
                },
                {
                    "nutrient_name": "fiber",
                    "amount": 4,
                    "unit": "g",
                    "confidence": 0.85,
                },
            ],
            "chicken": [
                {
                    "nutrient_name": "protein",
                    "amount": 31,
                    "unit": "g",
                    "confidence": 0.95,
                },
                {"nutrient_name": "iron", "amount": 1, "unit": "mg", "confidence": 0.8},
            ],
            "milk": [
                {
                    "nutrient_name": "calcium",
                    "amount": 300,
                    "unit": "mg",
                    "confidence": 0.95,
                },
                {
                    "nutrient_name": "protein",
                    "amount": 8,
                    "unit": "g",
                    "confidence": 0.9,
                },
            ],
        }

        return food_data.get(food_name.lower(), [])

    async def _fallback_get_recommendations(
        self, health_condition: str
    ) -> List[Dict[str, Any]]:
        """Fallback method for health recommendations"""
        recommendations = {
            "diabetes": [
                {"food_name": "oats", "effectiveness": 0.8, "confidence": 0.9},
                {"food_name": "fish", "effectiveness": 0.7, "confidence": 0.85},
                {"food_name": "broccoli", "effectiveness": 0.75, "confidence": 0.8},
            ],
            "hypertension": [
                {"food_name": "banana", "effectiveness": 0.75, "confidence": 0.8},
                {"food_name": "spinach", "effectiveness": 0.7, "confidence": 0.85},
                {"food_name": "garlic", "effectiveness": 0.65, "confidence": 0.7},
            ],
        }

        return recommendations.get(health_condition.lower(), [])


# Global graph service instance
graph_service = None


def get_graph_service() -> GraphService:
    """Get or create graph service instance"""
    global graph_service
    if graph_service is None:
        graph_service = GraphService()
    return graph_service


async def initialize_graph_service(
    uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"
):
    """Initialize graph service on startup"""
    global graph_service
    try:
        graph_service = GraphService(uri, user, password)
        logger.info("Graph service initialized")
        return graph_service
    except Exception as e:
        logger.error(f"Failed to initialize graph service: {e}")
        # Return service anyway for fallback functionality
        graph_service = GraphService()
        return graph_service
