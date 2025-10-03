"""
Performance Optimization Service with caching, async processing, and query optimization
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable, List, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import weakref
from functools import wraps
import threading

# Import Redis when available
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels for different types of data"""

    MEMORY = "memory"  # In-memory cache
    REDIS = "redis"  # Redis cache
    PERSISTENT = "persistent"  # File-based cache


@dataclass
class CacheConfig:
    """Cache configuration"""

    default_ttl: int = 3600  # 1 hour
    max_memory_size: int = 1000  # Maximum items in memory cache
    redis_url: Optional[str] = None
    enable_compression: bool = True
    cache_levels: List[CacheLevel] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""

    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time: float = 0.0
    total_requests: int = 0
    active_connections: int = 0
    memory_usage_mb: float = 0.0


class AsyncTaskManager:
    """Manager for asynchronous task processing"""

    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.workers = []
        self.running = False
        self.results = {}

    async def start(self, num_workers: int = 3):
        """Start async task workers"""
        self.running = True

        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)

        logger.info(f"Started {num_workers} async task workers")

    async def stop(self):
        """Stop async task workers"""
        self.running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

        logger.info("Stopped async task workers")

    async def _worker(self, worker_name: str):
        """Worker coroutine for processing tasks"""
        logger.info(f"Started async worker: {worker_name}")

        while self.running:
            try:
                # Get task from queue with timeout
                task_id, func, args, kwargs = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )

                # Execute task
                start_time = time.time()
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    self.results[task_id] = {
                        "status": "completed",
                        "result": result,
                        "execution_time": time.time() - start_time,
                        "completed_at": datetime.now(),
                    }

                except Exception as e:
                    self.results[task_id] = {
                        "status": "failed",
                        "error": str(e),
                        "execution_time": time.time() - start_time,
                        "completed_at": datetime.now(),
                    }

                # Mark task as done
                self.task_queue.task_done()

            except asyncio.TimeoutError:
                continue  # No tasks in queue, continue polling
            except Exception as e:
                logger.error(f"Error in async worker {worker_name}: {e}")

    async def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task for async processing"""
        task_id = hashlib.md5(f"{func.__name__}_{time.time()}".encode()).hexdigest()

        await self.task_queue.put((task_id, func, args, kwargs))

        # Initialize result placeholder
        self.results[task_id] = {"status": "queued", "submitted_at": datetime.now()}

        return task_id

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a submitted task"""
        return self.results.get(task_id)

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "queue_size": self.task_queue.qsize(),
            "active_workers": len(self.workers),
            "total_results": len(self.results),
            "completed_tasks": len(
                [r for r in self.results.values() if r["status"] == "completed"]
            ),
            "failed_tasks": len(
                [r for r in self.results.values() if r["status"] == "failed"]
            ),
        }


class MultiLevelCache:
    """Multi-level caching system"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = {}
        self.memory_access_times = {}
        self.memory_lock = threading.RLock()

        # Initialize Redis if available
        self.redis_client = None
        if REDIS_AVAILABLE and config.redis_url:
            try:
                self.redis_client = redis.from_url(config.redis_url)
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")

        # Performance tracking
        self.metrics = PerformanceMetrics()

    def _get_cache_key(self, key: str, prefix: str = "nutri") -> str:
        """Generate standardized cache key"""
        return f"{prefix}:{key}"

    def _serialize_value(self, value: Any) -> str:
        """Serialize value for caching"""
        try:
            return json.dumps(value, default=str)
        except Exception:
            return str(value)

    def _deserialize_value(self, serialized: str) -> Any:
        """Deserialize cached value"""
        try:
            return json.loads(serialized)
        except Exception:
            return serialized

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache (multi-level)"""
        cache_key = self._get_cache_key(key)

        # Level 1: Memory cache
        with self.memory_lock:
            if cache_key in self.memory_cache:
                self.memory_access_times[cache_key] = datetime.now()
                self.metrics.cache_hits += 1
                logger.debug(f"Memory cache hit: {key}")
                return self.memory_cache[cache_key]

        # Level 2: Redis cache
        if self.redis_client:
            try:
                cached_value = self.redis_client.get(cache_key)
                if cached_value:
                    value = self._deserialize_value(cached_value.decode("utf-8"))

                    # Store in memory cache for faster access
                    await self.set_memory(key, value)

                    self.metrics.cache_hits += 1
                    logger.debug(f"Redis cache hit: {key}")
                    return value
            except Exception as e:
                logger.warning(f"Redis get error: {e}")

        # Cache miss
        self.metrics.cache_misses += 1
        logger.debug(f"Cache miss: {key}")
        return default

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache (all levels)"""
        ttl = ttl or self.config.default_ttl
        cache_key = self._get_cache_key(key)

        # Set in memory cache
        await self.set_memory(key, value)

        # Set in Redis cache
        if self.redis_client:
            try:
                serialized_value = self._serialize_value(value)
                self.redis_client.setex(cache_key, ttl, serialized_value)
                logger.debug(f"Set in Redis cache: {key}")
            except Exception as e:
                logger.warning(f"Redis set error: {e}")

        return True

    async def set_memory(self, key: str, value: Any) -> bool:
        """Set value in memory cache only"""
        cache_key = self._get_cache_key(key)

        with self.memory_lock:
            # Check memory limit
            if len(self.memory_cache) >= self.config.max_memory_size:
                await self._evict_memory_cache()

            self.memory_cache[cache_key] = value
            self.memory_access_times[cache_key] = datetime.now()

        logger.debug(f"Set in memory cache: {key}")
        return True

    async def _evict_memory_cache(self):
        """Evict least recently used items from memory cache"""
        with self.memory_lock:
            if not self.memory_access_times:
                return

            # Sort by access time and remove oldest 25%
            sorted_keys = sorted(self.memory_access_times.items(), key=lambda x: x[1])

            evict_count = max(1, len(sorted_keys) // 4)

            for key, _ in sorted_keys[:evict_count]:
                self.memory_cache.pop(key, None)
                self.memory_access_times.pop(key, None)

            logger.debug(f"Evicted {evict_count} items from memory cache")

    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels"""
        cache_key = self._get_cache_key(key)

        # Delete from memory
        with self.memory_lock:
            self.memory_cache.pop(cache_key, None)
            self.memory_access_times.pop(cache_key, None)

        # Delete from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(cache_key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")

        logger.debug(f"Deleted from cache: {key}")
        return True

    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries"""
        cleared_count = 0

        if pattern:
            # Clear specific pattern
            pattern_key = self._get_cache_key(pattern)

            # Clear from memory
            with self.memory_lock:
                keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    self.memory_cache.pop(key, None)
                    self.memory_access_times.pop(key, None)
                    cleared_count += 1

            # Clear from Redis
            if self.redis_client:
                try:
                    keys = self.redis_client.keys(f"{pattern_key}*")
                    if keys:
                        self.redis_client.delete(*keys)
                        cleared_count += len(keys)
                except Exception as e:
                    logger.warning(f"Redis pattern clear error: {e}")
        else:
            # Clear all
            with self.memory_lock:
                cleared_count = len(self.memory_cache)
                self.memory_cache.clear()
                self.memory_access_times.clear()

            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Redis clear error: {e}")

        logger.info(f"Cleared {cleared_count} cache entries")
        return cleared_count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.memory_lock:
            memory_size = len(self.memory_cache)

        redis_info = {}
        if self.redis_client:
            try:
                redis_info = self.redis_client.info("memory")
            except Exception:
                redis_info = {"error": "Cannot get Redis info"}

        hit_rate = (
            self.metrics.cache_hits
            / max(1, self.metrics.cache_hits + self.metrics.cache_misses)
        ) * 100

        return {
            "memory_cache_size": memory_size,
            "max_memory_size": self.config.max_memory_size,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "hit_rate_percent": hit_rate,
            "redis_available": self.redis_client is not None,
            "redis_info": redis_info,
        }


class PerformanceOptimizationService:
    """Main performance optimization service"""

    def __init__(self, cache_config: Optional[CacheConfig] = None):
        self.cache_config = cache_config or CacheConfig()
        self.cache = MultiLevelCache(self.cache_config)
        self.task_manager = AsyncTaskManager()
        self.metrics = PerformanceMetrics()

        # Query optimization patterns
        self.query_patterns = {}
        self.response_times = {}

        logger.info("Performance optimization service initialized")

    async def start(self):
        """Start performance optimization service"""
        await self.task_manager.start()
        logger.info("Performance optimization service started")

    async def stop(self):
        """Stop performance optimization service"""
        await self.task_manager.stop()
        logger.info("Performance optimization service stopped")

    def cache_result(
        self, ttl: Optional[int] = None, key_func: Optional[Callable] = None
    ):
        """Decorator for caching function results"""

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

                # Try to get from cache
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                start_time = time.time()

                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                execution_time = time.time() - start_time

                # Cache the result
                await self.cache.set(cache_key, result, ttl)

                # Update metrics
                self.metrics.total_requests += 1
                self.metrics.avg_response_time = (
                    self.metrics.avg_response_time * (self.metrics.total_requests - 1)
                    + execution_time
                ) / self.metrics.total_requests

                return result

            return wrapper

        return decorator

    async def submit_background_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task for background processing"""
        return await self.task_manager.submit_task(func, *args, **kwargs)

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of background task"""
        return self.task_manager.get_task_result(task_id)

    async def optimize_query(self, query: str, query_type: str = "general") -> str:
        """Optimize query for better performance"""
        # Simple query optimization patterns
        optimizations = {
            "general": {
                "remove_stop_words": True,
                "normalize_whitespace": True,
                "lowercase": True,
            },
            "vector_search": {
                "expand_synonyms": True,
                "remove_stop_words": True,
                "stem_words": False,
            },
            "graph_search": {
                "extract_entities": True,
                "normalize_entities": True,
                "remove_stop_words": False,
            },
        }

        config = optimizations.get(query_type, optimizations["general"])
        optimized_query = query

        if config.get("lowercase"):
            optimized_query = optimized_query.lower()

        if config.get("normalize_whitespace"):
            optimized_query = " ".join(optimized_query.split())

        # Track query patterns for future optimization
        self.query_patterns[query_type] = self.query_patterns.get(query_type, 0) + 1

        return optimized_query

    async def batch_process(
        self, items: List[Any], func: Callable, batch_size: int = 10
    ) -> List[Any]:
        """Process items in batches for better performance"""
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]

            # Process batch in parallel
            if asyncio.iscoroutinefunction(func):
                batch_tasks = [func(item) for item in batch]
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )
            else:
                batch_results = [func(item) for item in batch]

            results.extend(batch_results)

        return results

    async def preload_cache(self, keys_and_functions: List[Tuple[str, Callable]]):
        """Preload cache with frequently accessed data"""
        preload_tasks = []

        for cache_key, func in keys_and_functions:
            # Check if already cached
            cached = await self.cache.get(cache_key)
            if cached is None:
                if asyncio.iscoroutinefunction(func):
                    preload_tasks.append(self._preload_single(cache_key, func))
                else:
                    preload_tasks.append(self._preload_single_sync(cache_key, func))

        if preload_tasks:
            await asyncio.gather(*preload_tasks, return_exceptions=True)
            logger.info(f"Preloaded {len(preload_tasks)} cache entries")

    async def _preload_single(self, cache_key: str, func: Callable):
        """Preload single async function result"""
        try:
            result = await func()
            await self.cache.set(cache_key, result)
        except Exception as e:
            logger.warning(f"Failed to preload {cache_key}: {e}")

    async def _preload_single_sync(self, cache_key: str, func: Callable):
        """Preload single sync function result"""
        try:
            result = func()
            await self.cache.set(cache_key, result)
        except Exception as e:
            logger.warning(f"Failed to preload {cache_key}: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        cache_stats = self.cache.get_stats()
        task_stats = self.task_manager.get_queue_stats()

        return {
            "cache": cache_stats,
            "tasks": task_stats,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "avg_response_time_ms": self.metrics.avg_response_time * 1000,
                "active_connections": self.metrics.active_connections,
            },
            "query_patterns": self.query_patterns,
            "timestamp": datetime.now().isoformat(),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on performance systems"""
        health = {
            "cache": {"status": "healthy"},
            "task_manager": {"status": "healthy"},
            "redis": {"status": "not_configured"},
        }

        # Check cache
        try:
            test_key = "health_check_test"
            await self.cache.set(test_key, "test_value", 60)
            cached_value = await self.cache.get(test_key)
            if cached_value != "test_value":
                health["cache"]["status"] = "unhealthy"
                health["cache"]["error"] = "Cache read/write test failed"
            await self.cache.delete(test_key)
        except Exception as e:
            health["cache"]["status"] = "unhealthy"
            health["cache"]["error"] = str(e)

        # Check Redis if available
        if self.cache.redis_client:
            try:
                self.cache.redis_client.ping()
                health["redis"]["status"] = "healthy"
            except Exception as e:
                health["redis"]["status"] = "unhealthy"
                health["redis"]["error"] = str(e)

        # Check task manager
        task_stats = self.task_manager.get_queue_stats()
        if not self.task_manager.running or task_stats["active_workers"] == 0:
            health["task_manager"]["status"] = "unhealthy"
            health["task_manager"]["error"] = "No active workers"

        return health


# Global performance service instance
performance_service = None


def get_performance_service() -> PerformanceOptimizationService:
    """Get or create performance optimization service instance"""
    global performance_service
    if performance_service is None:
        performance_service = PerformanceOptimizationService()
    return performance_service


async def initialize_performance_service(
    cache_config: Optional[CacheConfig] = None,
) -> PerformanceOptimizationService:
    """Initialize performance optimization service on startup"""
    global performance_service
    try:
        performance_service = PerformanceOptimizationService(cache_config)
        await performance_service.start()
        logger.info("Performance optimization service initialized")
        return performance_service
    except Exception as e:
        logger.error(f"Failed to initialize performance service: {e}")
        # Return service anyway for basic functionality
        performance_service = PerformanceOptimizationService()
        return performance_service
