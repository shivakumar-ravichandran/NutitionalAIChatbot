"""
Configuration management for the Nutritional AI Chatbot
Provides centralized settings, environment variables, and service configuration.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings"""

    host: str = "localhost"
    port: int = 5432
    database: str = "nutritional_chatbot"
    username: str = "postgres"
    password: str = ""

    @property
    def url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class Neo4jConfig:
    """Neo4j graph database configuration"""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    max_pool_size: int = 10
    timeout: int = 30

    @property
    def is_enabled(self) -> bool:
        return bool(self.password)


@dataclass
class RedisConfig:
    """Redis cache configuration"""

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    decode_responses: bool = True
    max_connections: int = 10
    socket_timeout: int = 5

    @property
    def is_enabled(self) -> bool:
        return True  # Redis is optional with fallback


@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""

    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30

    @property
    def is_enabled(self) -> bool:
        return bool(self.api_key)


@dataclass
class HuggingFaceConfig:
    """Hugging Face model configuration"""

    model_name: str = "microsoft/DialoGPT-medium"
    cache_dir: str = "./models/huggingface"
    device: str = "cpu"  # or "cuda" if available
    max_length: int = 512

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)


@dataclass
class VectorConfig:
    """Vector embedding configuration"""

    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    faiss_index_type: str = "IndexFlatL2"
    cache_dir: str = "./models/embeddings"
    batch_size: int = 32

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)


@dataclass
class CacheConfig:
    """Caching configuration"""

    memory_cache_size: int = 1000
    memory_ttl: int = 3600  # 1 hour
    redis_ttl: int = 86400  # 24 hours
    preload_cache: bool = True
    cache_statistics: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""

    max_workers: int = 4
    task_timeout: int = 300  # 5 minutes
    batch_size: int = 10
    query_timeout: int = 30
    enable_metrics: bool = True


@dataclass
class SecurityConfig:
    """Security and safety configuration"""

    enable_content_filter: bool = True
    max_input_length: int = 2000
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class LoggingConfig:
    """Logging configuration"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


class Settings:
    """Main settings class that loads and manages all configuration"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.database = DatabaseConfig()
        self.neo4j = Neo4jConfig()
        self.redis = RedisConfig()
        self.openai = OpenAIConfig()
        self.huggingface = HuggingFaceConfig()
        self.vector = VectorConfig()
        self.cache = CacheConfig()
        self.performance = PerformanceConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()

        # Load configuration
        self._load_from_environment()
        if config_file:
            self._load_from_file(config_file)

        # Ensure directories exist
        self._create_directories()

    def _load_from_environment(self):
        """Load configuration from environment variables"""

        # Database configuration
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", self.database.port))
        self.database.database = os.getenv("DB_NAME", self.database.database)
        self.database.username = os.getenv("DB_USER", self.database.username)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)

        # Neo4j configuration
        self.neo4j.uri = os.getenv("NEO4J_URI", self.neo4j.uri)
        self.neo4j.username = os.getenv("NEO4J_USER", self.neo4j.username)
        self.neo4j.password = os.getenv("NEO4J_PASSWORD", self.neo4j.password)

        # Redis configuration
        self.redis.host = os.getenv("REDIS_HOST", self.redis.host)
        self.redis.port = int(os.getenv("REDIS_PORT", self.redis.port))
        self.redis.password = os.getenv("REDIS_PASSWORD", self.redis.password)

        # OpenAI configuration
        self.openai.api_key = os.getenv("OPENAI_API_KEY", self.openai.api_key)
        self.openai.model = os.getenv("OPENAI_MODEL", self.openai.model)

        # Vector model configuration
        self.vector.model_name = os.getenv("VECTOR_MODEL", self.vector.model_name)
        self.vector.cache_dir = os.getenv("VECTOR_CACHE_DIR", self.vector.cache_dir)

        # HuggingFace configuration
        self.huggingface.model_name = os.getenv("HF_MODEL", self.huggingface.model_name)
        self.huggingface.cache_dir = os.getenv(
            "HF_CACHE_DIR", self.huggingface.cache_dir
        )

        # Performance configuration
        self.performance.max_workers = int(
            os.getenv("MAX_WORKERS", self.performance.max_workers)
        )

        # Security configuration
        self.security.enable_content_filter = (
            os.getenv("ENABLE_CONTENT_FILTER", "true").lower() == "true"
        )

        # Logging configuration
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.file = os.getenv("LOG_FILE", self.logging.file)

    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, "r") as f:
                    config_data = json.load(f)

                # Update configurations from file
                self._update_from_dict(config_data)
                logger.info(f"Configuration loaded from {config_file}")
            else:
                logger.warning(f"Configuration file {config_file} not found")
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")

    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in config_data.items():
            if hasattr(self, section) and isinstance(getattr(self, section), type):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.vector.cache_path,
            self.huggingface.cache_path,
        ]

        if self.logging.file:
            log_dir = Path(self.logging.file).parent
            directories.append(log_dir)

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create directory {directory}: {e}")

    def get_enabled_services(self) -> Dict[str, bool]:
        """Get status of all services"""
        return {
            "database": True,  # Always required
            "neo4j": self.neo4j.is_enabled,
            "redis": self.redis.is_enabled,
            "openai": self.openai.is_enabled,
            "vector_embeddings": True,  # Sentence transformers doesn't need API key
            "huggingface": True,  # Local models
            "performance_optimization": True,
            "content_filtering": self.security.enable_content_filter,
        }

    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        issues = []

        # Check required configurations
        if not self.database.password:
            issues.append("Database password not configured")

        # Check optional but recommended configurations
        if not self.openai.api_key:
            issues.append("OpenAI API key not configured - will use fallback responses")

        if not self.neo4j.password:
            issues.append(
                "Neo4j password not configured - graph features will be limited"
            )

        # Check directory permissions
        try:
            self.vector.cache_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create vector cache directory: {e}")

        try:
            self.huggingface.cache_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create HuggingFace cache directory: {e}")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for serialization"""
        return {
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username,
                # Don't include password in serialization
            },
            "neo4j": {
                "uri": self.neo4j.uri,
                "username": self.neo4j.username,
                "enabled": self.neo4j.is_enabled,
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "enabled": self.redis.is_enabled,
            },
            "openai": {"model": self.openai.model, "enabled": self.openai.is_enabled},
            "vector": {
                "model_name": self.vector.model_name,
                "embedding_dim": self.vector.embedding_dim,
                "cache_dir": str(self.vector.cache_path),
            },
            "services": self.get_enabled_services(),
        }

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format,
            filename=self.logging.file,
            filemode="a",
        )

        if self.logging.file:
            # Add rotation for file logging
            from logging.handlers import RotatingFileHandler

            handler = RotatingFileHandler(
                self.logging.file,
                maxBytes=self.logging.max_bytes,
                backupCount=self.logging.backup_count,
            )
            handler.setFormatter(logging.Formatter(self.logging.format))

            # Get root logger and add handler
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)


# Global settings instance
settings = None


def get_settings(config_file: Optional[str] = None) -> Settings:
    """Get global settings instance"""
    global settings
    if settings is None:
        settings = Settings(config_file)
        settings.setup_logging()
    return settings


def load_settings(config_file: Optional[str] = None) -> Settings:
    """Load or reload settings"""
    global settings
    settings = Settings(config_file)
    settings.setup_logging()
    return settings


# Environment-specific configurations
class DevelopmentConfig(Settings):
    """Development environment configuration"""

    def __init__(self, config_file: Optional[str] = None):
        super().__init__(config_file)
        self.logging.level = "DEBUG"
        self.security.enable_content_filter = False
        self.performance.enable_metrics = True


class ProductionConfig(Settings):
    """Production environment configuration"""

    def __init__(self, config_file: Optional[str] = None):
        super().__init__(config_file)
        self.logging.level = "WARNING"
        self.security.enable_content_filter = True
        self.performance.enable_metrics = True
        self.cache.preload_cache = True


class TestConfig(Settings):
    """Test environment configuration"""

    def __init__(self, config_file: Optional[str] = None):
        super().__init__(config_file)
        self.database.database = "test_nutritional_chatbot"
        self.logging.level = "DEBUG"
        self.cache.memory_cache_size = 100
        self.performance.max_workers = 2


def get_config_for_environment(env: str = None) -> Settings:
    """Get configuration for specific environment"""
    if env is None:
        env = os.getenv("ENVIRONMENT", "development")

    env = env.lower()
    if env == "production":
        return ProductionConfig()
    elif env == "test":
        return TestConfig()
    else:
        return DevelopmentConfig()
