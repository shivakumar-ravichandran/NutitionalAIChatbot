# Nutritional AI Chatbot - Deployment Guide

# Phases 4 & 5 Implementation

This guide covers deploying the enhanced Nutritional AI Chatbot with advanced AI services including vector embeddings, hybrid retrieval, LLM integration, and performance optimization.

## Architecture Overview

The system now includes:

- **Phase 1-3**: FastAPI backend, PostgreSQL database, basic NLP
- **Phase 4-5**: Vector embeddings, hybrid retrieval, LLM integration, dynamic prompts, caching

### Service Components

1. **Vector Embedding Service**: Semantic search using sentence-transformers + FAISS
2. **Hybrid Retrieval Service**: Combines graph filtering with vector similarity
3. **LLM Integration Service**: OpenAI GPT + Hugging Face Transformers
4. **Dynamic Prompt Service**: Age-appropriate and culturally-aware prompts
5. **Performance Service**: Multi-level caching (memory + Redis) + async processing
6. **Enhanced Chat**: AI-powered conversational interface
7. **Knowledge Base**: Advanced search and graph visualization

## Prerequisites

### Required Services

- **PostgreSQL 12+**: Primary database
- **Python 3.8+**: Runtime environment

### Recommended Services (Optional but Enhances Performance)

- **Redis 6+**: Caching and session storage
- **Neo4j 4+**: Graph database for knowledge relationships

### API Keys (Optional but Enables Advanced Features)

- **OpenAI API Key**: For GPT-based responses
- **Hugging Face Hub**: For model downloads (free)

## Installation Steps

### 1. Environment Setup

```bash
# Clone repository
git clone <repository_url>
cd NutitionalAIChatbot

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install all dependencies
cd backend
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

### 3. Database Setup

#### PostgreSQL Setup

```sql
-- Create database and user
CREATE DATABASE nutritional_chatbot;
CREATE USER chatbot_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE nutritional_chatbot TO chatbot_user;
```

#### Neo4j Setup (Optional)

```bash
# Install Neo4j and start service
# Set password for neo4j user
# Default connection: bolt://localhost:7687
```

#### Redis Setup (Optional)

```bash
# Install Redis and start service
# Default connection: localhost:6379
```

### 4. Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit .env file with your settings
nano .env
```

#### Essential Configuration (.env):

```bash
# Database (Required)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nutritional_chatbot
DB_USER=chatbot_user
DB_PASSWORD=your_secure_password

# OpenAI API (Optional - enables advanced AI responses)
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo

# Neo4j (Optional - enables graph features)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Redis (Optional - improves performance)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
MAX_WORKERS=4
ENABLE_CONTENT_FILTER=true
```

### 5. First Run & Database Migration

```bash
# Initialize database and run server
cd backend
python main.py
```

The application will:

1. Create database tables automatically
2. Initialize all AI services
3. Download required ML models
4. Start the FastAPI server on http://localhost:8000

## Service Configuration Matrix

| Service               | Required | Fallback Behavior       | Performance Impact             |
| --------------------- | -------- | ----------------------- | ------------------------------ |
| PostgreSQL            | ✅ Yes   | None - App won't start  | Critical                       |
| OpenAI API            | ❌ No    | Rule-based responses    | High - reduces AI quality      |
| Neo4j                 | ❌ No    | Basic text search       | Medium - loses graph reasoning |
| Redis                 | ❌ No    | Memory-only caching     | Low - slower repeated queries  |
| Sentence Transformers | ✅ Yes   | Downloads automatically | Medium - initial startup delay |

## Performance Optimization

### Memory Requirements

- **Minimum**: 2GB RAM
- **Recommended**: 4GB+ RAM
- **With GPU**: 6GB+ VRAM for local LLM models

### Model Caching

Models are cached locally:

- **Vector models**: `./models/embeddings/`
- **Hugging Face models**: `./models/huggingface/`
- **First run**: Downloads ~1-2GB of models
- **Subsequent runs**: Loads from cache (fast startup)

### Caching Strategy

1. **Memory Cache**: Fast access to recent queries
2. **Redis Cache**: Persistent cache across restarts
3. **Vector Index**: Cached similarity searches
4. **LLM Responses**: Cached for repeated questions

## API Endpoints

### Enhanced Endpoints (Phase 4-5)

- `POST /api/chat/enhanced_message` - AI-powered chat with hybrid retrieval
- `GET /api/knowledge/search` - Semantic search across knowledge base
- `GET /api/knowledge/graph` - Query knowledge graph relationships
- `GET /api/knowledge/entities` - Extract and analyze entities
- `GET /health` - Comprehensive service health check
- `GET /api/status` - Detailed service status and configuration

### Existing Endpoints (Phase 1-3)

- `POST /api/profile/` - User profile management
- `POST /api/chat/message` - Basic chat (fallback)
- `POST /api/admin/upload` - Document upload
- `GET /docs` - Interactive API documentation

## Monitoring and Health Checks

### Health Check Endpoint

```bash
curl http://localhost:8000/health
```

Response includes:

- Overall system status
- Individual service health
- Configuration information
- Performance metrics

### Service Status

```bash
curl http://localhost:8000/api/status
```

Provides detailed information about:

- Service initialization status
- Feature availability
- Configuration summary
- Endpoint activity

## Troubleshooting

### Common Issues

#### 1. Service Initialization Failed

```
Error: Service initialization failed
```

**Solution**: Check database connection and credentials in `.env`

#### 2. OpenAI API Limit Exceeded

```
Warning: OpenAI API not configured - using fallback responses
```

**Solution**: Add valid `OPENAI_API_KEY` to `.env` or continue with fallback

#### 3. Model Download Errors

```
Error: Failed to download sentence-transformers model
```

**Solution**: Check internet connection and disk space (models ~1-2GB)

#### 4. Memory Issues

```
Error: CUDA out of memory / Killed
```

**Solution**: Reduce `MAX_WORKERS` or use CPU-only models

### Performance Issues

#### Slow First Request

- **Cause**: Model loading on first use
- **Solution**: Enable `PRELOAD_CACHE=true` in production

#### High Memory Usage

- **Cause**: Multiple models loaded simultaneously
- **Solution**: Reduce concurrent workers or use lighter models

#### Redis Connection Failed

- **Cause**: Redis not available
- **Solution**: Install Redis or set `REDIS_HOST=""` to disable

## Production Deployment

### Environment Variables

```bash
ENVIRONMENT=production
LOG_LEVEL=WARNING
ENABLE_CONTENT_FILTER=true
MAX_WORKERS=4
ALLOWED_ORIGINS=https://yourdomain.com
```

### Security Recommendations

1. Use strong database passwords
2. Enable HTTPS/TLS in production
3. Configure proper CORS origins
4. Enable content filtering
5. Set up rate limiting
6. Regular security updates

### Scaling Considerations

- **Horizontal Scaling**: Multiple FastAPI instances behind load balancer
- **Database**: PostgreSQL replication for read queries
- **Caching**: Redis cluster for distributed caching
- **Models**: Shared model cache across instances

## Advanced Configuration

### Custom Models

```bash
# Use custom sentence transformer model
VECTOR_MODEL=your-custom/sentence-transformer-model

# Use custom HuggingFace model
HF_MODEL=your-custom/huggingface-model
```

### Performance Tuning

```bash
# Increase worker processes
MAX_WORKERS=8

# Adjust cache sizes
MEMORY_CACHE_SIZE=2000
REDIS_TTL=7200

# Batch processing
BATCH_SIZE=20
```

### Logging Configuration

```bash
LOG_LEVEL=DEBUG
LOG_FILE=logs/chatbot.log
```

## Support and Maintenance

### Regular Maintenance

1. **Database**: Regular backups and maintenance
2. **Models**: Update sentence-transformers models quarterly
3. **Dependencies**: Keep Python packages updated
4. **Logs**: Rotate and monitor application logs
5. **Cache**: Clear Redis cache periodically

### Monitoring Metrics

- Response times for each service
- Cache hit rates
- Database query performance
- Model inference times
- Memory and CPU usage

## Next Steps

After successful deployment:

1. Test all endpoints using `/docs`
2. Upload initial knowledge documents via `/api/admin/upload`
3. Create user profiles and test enhanced chat
4. Monitor performance and adjust configuration
5. Set up automated backups and monitoring

For support, check the logs at `logs/chatbot.log` and service status at `/health` endpoint.
