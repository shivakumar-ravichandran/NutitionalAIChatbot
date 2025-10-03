"""
Configuration management service for initializing and managing all chatbot services.
Handles service dependencies, health checks, and graceful startup/shutdown.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import Settings, get_settings
from database.database import SessionLocal, engine
from database import models
from services.nlp_service import NLPService
from services.graph_service import GraphService
from services.vector_service import VectorEmbeddingService
from services.hybrid_retrieval_service import HybridRetrievalService
from services.llm_service import LLMIntegrationService
from services.prompt_service import DynamicPromptService
from services.performance_service import PerformanceOptimizationService

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages initialization and lifecycle of all chatbot services"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.services: Dict[str, Any] = {}
        self.initialized = False
        self.health_status: Dict[str, bool] = {}

    async def initialize_services(self):
        """Initialize all services in proper dependency order"""
        try:
            logger.info("Starting service initialization...")

            # Create database tables
            await self._initialize_database()

            # Initialize core services first
            await self._initialize_core_services()

            # Initialize AI/ML services
            await self._initialize_ai_services()

            # Initialize performance services
            await self._initialize_performance_services()

            # Validate all services
            await self._validate_services()

            self.initialized = True
            logger.info("All services initialized successfully")

        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            await self.shutdown_services()
            raise

    async def _initialize_database(self):
        """Initialize database connection and tables"""
        try:
            logger.info("Initializing database...")

            # Create all tables
            models.Base.metadata.create_all(bind=engine)

            # Test database connection
            db = SessionLocal()
            try:
                db.execute("SELECT 1")
                self.health_status["database"] = True
                logger.info("Database connection established")
            except Exception as e:
                self.health_status["database"] = False
                logger.error(f"Database connection failed: {e}")
                raise
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def _initialize_core_services(self):
        """Initialize core NLP and graph services"""
        try:
            # Initialize NLP Service
            logger.info("Initializing NLP service...")
            nlp_service = NLPService()
            await nlp_service.initialize()
            self.services["nlp"] = nlp_service
            self.health_status["nlp"] = True
            logger.info("NLP service initialized")

            # Initialize Graph Service
            logger.info("Initializing graph service...")
            graph_service = GraphService(
                uri=self.settings.neo4j.uri,
                username=self.settings.neo4j.username,
                password=self.settings.neo4j.password,
            )

            if self.settings.neo4j.is_enabled:
                try:
                    await graph_service.initialize()
                    self.health_status["graph"] = True
                    logger.info("Graph service initialized with Neo4j")
                except Exception as e:
                    logger.warning(f"Neo4j connection failed, using fallback: {e}")
                    self.health_status["graph"] = False
            else:
                logger.info("Neo4j not configured, graph service will use fallback")
                self.health_status["graph"] = False

            self.services["graph"] = graph_service

        except Exception as e:
            logger.error(f"Core services initialization failed: {e}")
            raise

    async def _initialize_ai_services(self):
        """Initialize AI/ML services"""
        try:
            # Initialize Vector Embedding Service
            logger.info("Initializing vector embedding service...")
            vector_service = VectorEmbeddingService(
                model_name=self.settings.vector.model_name,
                cache_dir=str(self.settings.vector.cache_path),
            )
            await vector_service.initialize()
            self.services["vector"] = vector_service
            self.health_status["vector"] = True
            logger.info("Vector embedding service initialized")

            # Initialize LLM Integration Service
            logger.info("Initializing LLM integration service...")
            llm_service = LLMIntegrationService(
                openai_api_key=self.settings.openai.api_key,
                openai_model=self.settings.openai.model,
                hf_model_name=self.settings.huggingface.model_name,
                hf_cache_dir=str(self.settings.huggingface.cache_path),
            )
            await llm_service.initialize()
            self.services["llm"] = llm_service
            self.health_status["llm"] = True
            logger.info("LLM integration service initialized")

            # Initialize Dynamic Prompt Service
            logger.info("Initializing dynamic prompt service...")
            prompt_service = DynamicPromptService()
            await prompt_service.initialize()
            self.services["prompt"] = prompt_service
            self.health_status["prompt"] = True
            logger.info("Dynamic prompt service initialized")

            # Initialize Hybrid Retrieval Service
            logger.info("Initializing hybrid retrieval service...")
            hybrid_service = HybridRetrievalService(
                graph_service=self.services["graph"],
                vector_service=self.services["vector"],
                nlp_service=self.services["nlp"],
            )
            await hybrid_service.initialize()
            self.services["hybrid"] = hybrid_service
            self.health_status["hybrid"] = True
            logger.info("Hybrid retrieval service initialized")

        except Exception as e:
            logger.error(f"AI services initialization failed: {e}")
            raise

    async def _initialize_performance_services(self):
        """Initialize performance optimization services"""
        try:
            logger.info("Initializing performance service...")
            performance_service = PerformanceOptimizationService(
                redis_host=self.settings.redis.host,
                redis_port=self.settings.redis.port,
                redis_password=(
                    self.settings.redis.password
                    if self.settings.redis.password
                    else None
                ),
                max_workers=self.settings.performance.max_workers,
            )
            await performance_service.initialize()
            self.services["performance"] = performance_service
            self.health_status["performance"] = True
            logger.info("Performance service initialized")

        except Exception as e:
            logger.error(f"Performance services initialization failed: {e}")
            raise

    async def _validate_services(self):
        """Validate all services are working correctly"""
        logger.info("Validating services...")

        validation_results = {}

        # Test NLP service
        try:
            result = await self.services["nlp"].extract_entities(
                "I need protein rich food"
            )
            validation_results["nlp"] = len(result) >= 0
        except Exception as e:
            logger.warning(f"NLP service validation failed: {e}")
            validation_results["nlp"] = False

        # Test vector service
        try:
            embedding = await self.services["vector"].generate_embedding("test query")
            validation_results["vector"] = len(embedding) > 0
        except Exception as e:
            logger.warning(f"Vector service validation failed: {e}")
            validation_results["vector"] = False

        # Test LLM service
        try:
            response = await self.services["llm"].generate_response(
                "What is protein?", context="Nutrition question"
            )
            validation_results["llm"] = len(response.get("response", "")) > 0
        except Exception as e:
            logger.warning(f"LLM service validation failed: {e}")
            validation_results["llm"] = False

        # Log validation results
        for service, status in validation_results.items():
            if status:
                logger.info(f"✓ {service} service validation passed")
            else:
                logger.warning(f"✗ {service} service validation failed")

        # Update health status
        self.health_status.update(validation_results)

    async def shutdown_services(self):
        """Gracefully shutdown all services"""
        logger.info("Shutting down services...")

        shutdown_tasks = []

        for service_name, service in self.services.items():
            if hasattr(service, "cleanup"):
                shutdown_tasks.append(self._safe_shutdown(service_name, service))

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        self.services.clear()
        self.health_status.clear()
        self.initialized = False
        logger.info("All services shut down")

    async def _safe_shutdown(self, service_name: str, service: Any):
        """Safely shutdown a single service"""
        try:
            await service.cleanup()
            logger.info(f"✓ {service_name} service shut down")
        except Exception as e:
            logger.error(f"✗ Error shutting down {service_name} service: {e}")

    def get_service(self, service_name: str) -> Optional[Any]:
        """Get a service by name"""
        return self.services.get(service_name)

    def get_health_status(self) -> Dict[str, bool]:
        """Get health status of all services"""
        return self.health_status.copy()

    def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive service information"""
        return {
            "initialized": self.initialized,
            "total_services": len(self.services),
            "healthy_services": sum(
                1 for status in self.health_status.values() if status
            ),
            "services": {
                name: {
                    "initialized": name in self.services,
                    "healthy": self.health_status.get(name, False),
                    "type": type(service).__name__ if name in self.services else None,
                }
                for name in [
                    "database",
                    "nlp",
                    "graph",
                    "vector",
                    "llm",
                    "prompt",
                    "hybrid",
                    "performance",
                ]
            },
            "configuration": self.settings.to_dict(),
        }


# Global service manager instance
service_manager: Optional[ServiceManager] = None


async def get_service_manager() -> ServiceManager:
    """Get global service manager instance"""
    global service_manager
    if service_manager is None:
        settings = get_settings()
        service_manager = ServiceManager(settings)
        await service_manager.initialize_services()
    return service_manager


async def initialize_application():
    """Initialize the entire application"""
    try:
        logger.info("Starting application initialization...")

        # Load settings
        settings = get_settings()

        # Validate configuration
        config_issues = settings.validate_configuration()
        if config_issues:
            for issue in config_issues:
                logger.warning(f"Configuration issue: {issue}")

        # Initialize service manager
        manager = await get_service_manager()

        logger.info("Application initialization completed successfully")
        return manager

    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        raise


async def shutdown_application():
    """Shutdown the entire application"""
    global service_manager
    if service_manager:
        await service_manager.shutdown_services()
        service_manager = None


@asynccontextmanager
async def lifespan_manager():
    """Context manager for application lifespan"""
    manager = None
    try:
        manager = await initialize_application()
        yield manager
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        if manager:
            await manager.shutdown_services()


# Health check endpoint data
async def get_health_check() -> Dict[str, Any]:
    """Get comprehensive health check information"""
    try:
        manager = await get_service_manager()

        return {
            "status": "healthy" if manager.initialized else "unhealthy",
            "timestamp": asyncio.get_event_loop().time(),
            "services": manager.get_health_status(),
            "info": manager.get_service_info(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time(),
        }
