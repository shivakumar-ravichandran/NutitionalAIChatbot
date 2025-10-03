"""
Main FastAPI application for Nutritional AI Chatbot
Enhanced with advanced AI services and comprehensive configuration management
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path
import os
import logging
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from database.database import init_db, get_db
from routers import profile, chat, admin
from config.settings import get_settings, load_settings
from config.service_manager import ServiceManager, get_health_check
from services.nlp_service import initialize_nlp_service
from services.graph_service import initialize_graph_service

# Global service manager
service_manager: ServiceManager = None
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan manager with comprehensive service initialization"""
    global service_manager

    # Startup
    print("🚀 Starting Nutritional AI Chatbot with Advanced Services...")

    try:
        # Load configuration
        settings = get_settings()
        print(
            f"⚙️  Configuration loaded for {os.getenv('ENVIRONMENT', 'development')} environment"
        )

        # Initialize service manager
        service_manager = ServiceManager(settings)
        await service_manager.initialize_services()

        # Make services available to the app
        app.state.service_manager = service_manager
        app.state.nlp_service = service_manager.get_service("nlp")
        app.state.graph_service = service_manager.get_service("graph")
        app.state.vector_service = service_manager.get_service("vector")
        app.state.llm_service = service_manager.get_service("llm")
        app.state.hybrid_service = service_manager.get_service("hybrid")
        app.state.prompt_service = service_manager.get_service("prompt")
        app.state.performance_service = service_manager.get_service("performance")

        # Log service status
        health_status = service_manager.get_health_status()
        enabled_services = [name for name, status in health_status.items() if status]
        print(f"✅ Services initialized: {', '.join(enabled_services)}")

        if not health_status.get("openai", False):
            print("⚠️  OpenAI API not configured - using fallback responses")
        if not health_status.get("graph", False):
            print("⚠️  Neo4j not configured - using basic retrieval")

        print("🎉 All systems ready!")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        print(f"❌ Service initialization failed: {e}")
        raise

    yield

    # Shutdown
    print("🛑 Shutting down Nutritional AI Chatbot...")
    if service_manager:
        await service_manager.shutdown_services()
    print("✅ Shutdown complete")


# Load settings early
settings = get_settings()

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="Nutritional AI Chatbot API",
    description="Advanced AI-powered conversational system for personalized nutritional advice with hybrid retrieval, semantic search, and intelligent responses",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (will include enhanced chat and knowledge routers if they exist)
app.include_router(profile.router, prefix="/api/profile", tags=["profile"])

# Try to include enhanced chat router, fallback to basic chat
try:
    from routers.enhanced_chat import router as enhanced_chat_router

    app.include_router(enhanced_chat_router, prefix="/api/chat", tags=["enhanced_chat"])
    print("✅ Enhanced chat router loaded")
except ImportError:
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    print("⚠️  Using basic chat router - enhanced features not available")

# Try to include knowledge base router
try:
    from routers.knowledge import router as knowledge_router

    app.include_router(knowledge_router, prefix="/api/knowledge", tags=["knowledge"])
    print("✅ Knowledge base router loaded")
except ImportError:
    print("⚠️  Knowledge base router not available")

app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic information"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Nutritional AI Chatbot API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { color: #2e7d32; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1 class="header">🥗 Nutritional AI Chatbot API</h1>
        <p>Welcome to the Nutritional AI Chatbot backend service!</p>
        
        <h2>Available Endpoints:</h2>
        <div class="endpoint"><strong>GET /docs</strong> - Interactive API documentation</div>
        <div class="endpoint"><strong>GET /health</strong> - Comprehensive health check</div>
        <div class="endpoint"><strong>GET /api/status</strong> - Service status and configuration</div>
        <div class="endpoint"><strong>POST /api/profile/</strong> - Create user profile</div>
        <div class="endpoint"><strong>POST /api/chat/enhanced_message</strong> - Send message with AI-powered responses</div>
        <div class="endpoint"><strong>GET /api/knowledge/search</strong> - Search knowledge base</div>
        <div class="endpoint"><strong>GET /api/knowledge/graph</strong> - Query knowledge graph</div>
        <div class="endpoint"><strong>POST /api/admin/upload</strong> - Upload documents</div>
        
        <h2>Status:</h2>
        <p>✅ API is running and ready to serve requests</p>
        
        <p><a href="/docs">📚 View Interactive API Documentation</a></p>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint with service status"""
    try:
        health_data = await get_health_check()
        return health_data
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Nutritional AI Chatbot",
            "version": "2.0.0",
            "error": str(e),
            "message": "Service health check failed",
        }


@app.get("/api/status")
async def api_status():
    """Detailed API status with all component health and configuration"""
    try:
        if service_manager and service_manager.initialized:
            service_info = service_manager.get_service_info()
            health_status = service_manager.get_health_status()

            return {
                "api": "online",
                "version": "2.0.0",
                "environment": os.getenv("ENVIRONMENT", "development"),
                "total_services": service_info.get("total_services", 0),
                "healthy_services": service_info.get("healthy_services", 0),
                "services": {
                    "database": health_status.get("database", False),
                    "nlp": health_status.get("nlp", False),
                    "graph": health_status.get("graph", False),
                    "vector_embeddings": health_status.get("vector", False),
                    "llm_integration": health_status.get("llm", False),
                    "hybrid_retrieval": health_status.get("hybrid", False),
                    "dynamic_prompts": health_status.get("prompt", False),
                    "performance_optimization": health_status.get("performance", False),
                },
                "features": {
                    "enhanced_chat": health_status.get("llm", False),
                    "semantic_search": health_status.get("vector", False),
                    "graph_reasoning": health_status.get("graph", False),
                    "personalized_responses": health_status.get("prompt", False),
                    "caching": health_status.get("performance", False),
                },
                "endpoints": {
                    "profiles": "active",
                    "chat": "active",
                    "enhanced_chat": health_status.get("llm", False),
                    "knowledge_base": health_status.get("vector", False),
                    "admin": "active",
                },
                "configuration": service_info.get("configuration", {}),
            }
        else:
            return {
                "api": "online",
                "version": "2.0.0",
                "status": "initializing",
                "message": "Services are still initializing",
            }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"api": "online", "version": "2.0.0", "status": "error", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
