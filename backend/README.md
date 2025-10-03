# Nutritional AI Chatbot Backend

A comprehensive FastAPI backend for an intelligent nutritional chatbot that provides personalized dietary advice and guidance.

## 🚀 Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **SQLite Database**: Lightweight database for user profiles and chat history
- **NLP Integration**: spaCy-powered entity extraction and text analysis
- **Knowledge Graph**: Neo4j integration for nutritional knowledge management
- **Document Processing**: Support for PDF, DOCX, and TXT file uploads
- **Intelligent Chat**: Context-aware responses with nutritional expertise
- **Profile Management**: Personalized user profiles with dietary preferences
- **Admin Interface**: Document upload and processing capabilities

## 📁 Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── test_backend.py        # Backend validation tests
├── database/
│   ├── __init__.py
│   ├── database.py        # Database configuration
│   └── models.py          # SQLAlchemy models
├── routers/
│   ├── __init__.py
│   ├── profile.py         # User profile management
│   ├── chat.py           # Chat endpoints
│   └── admin.py          # Document management
├── schemas/
│   ├── __init__.py
│   └── schemas.py        # Pydantic models
├── services/
│   ├── __init__.py
│   ├── nlp_service.py    # Natural language processing
│   └── graph_service.py  # Knowledge graph management
└── uploads/              # Document upload directory
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Setup

1. **Clone the repository**

   ```bash
   cd c:\Personal\Thesis\NutitionalAIChatbot\backend
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Test the backend**

   ```bash
   python test_backend.py
   ```

6. **Start the server**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once the server is running, visit:

- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🔗 API Endpoints

### Core Endpoints

| Method | Endpoint      | Description                        |
| ------ | ------------- | ---------------------------------- |
| GET    | `/`           | Root endpoint with API information |
| GET    | `/health`     | Health check                       |
| GET    | `/api/status` | Detailed service status            |

### Profile Management

| Method | Endpoint                      | Description         |
| ------ | ----------------------------- | ------------------- |
| POST   | `/api/profile/`               | Create user profile |
| GET    | `/api/profile/{profile_uuid}` | Get profile         |
| PUT    | `/api/profile/{profile_uuid}` | Update profile      |
| DELETE | `/api/profile/{profile_uuid}` | Delete profile      |

### Chat System

| Method | Endpoint                            | Description          |
| ------ | ----------------------------------- | -------------------- |
| POST   | `/api/chat/message`                 | Send chat message    |
| GET    | `/api/chat/sessions/{profile_uuid}` | Get chat sessions    |
| GET    | `/api/chat/session/{session_uuid}`  | Get session details  |
| POST   | `/api/chat/suggestions`             | Get chat suggestions |

### Admin & Document Management

| Method | Endpoint                               | Description             |
| ------ | -------------------------------------- | ----------------------- |
| POST   | `/api/admin/upload`                    | Upload document         |
| POST   | `/api/admin/process/{document_uuid}`   | Process document        |
| GET    | `/api/admin/documents`                 | List documents          |
| GET    | `/api/admin/entities`                  | List extracted entities |
| DELETE | `/api/admin/documents/{document_uuid}` | Delete document         |

## 🧠 AI Features

### Natural Language Processing

- **Entity Extraction**: Food items, nutrients, health conditions
- **Relationship Mapping**: Food-nutrient relationships
- **Intent Recognition**: Understanding user queries
- **Context Awareness**: Maintaining conversation context

### Knowledge Graph

- **Food Ontology**: Structured food and nutrient data
- **Health Relationships**: Links between food and health conditions
- **Recommendation Engine**: Personalized food suggestions
- **Hybrid Search**: Vector + graph-based retrieval

### Intelligent Responses

- **Nutritional Guidance**: Evidence-based dietary advice
- **Personalization**: Tailored to user profiles and preferences
- **Multi-modal**: Text processing with future image support
- **Contextual**: Remembers conversation history

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Database
DATABASE_URL=sqlite:///./nutritional_chatbot.db

# Neo4j (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# CORS
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### Database Schema

The application automatically creates the following tables:

- `user_profiles` - User profile information
- `chat_sessions` - Chat session metadata
- `chat_messages` - Individual messages
- `documents` - Uploaded documents
- `extracted_entities` - NLP-extracted entities
- `entity_relationships` - Entity relationships
- `allergies` - User allergies
- `health_conditions` - User health conditions

## 🧪 Testing

Run the backend validation tests:

```bash
python test_backend.py
```

This will test:

- Module imports
- Service initialization
- Database connectivity
- API endpoint availability

## 🚀 Deployment

### Local Development

```bash
python main.py
```

### Production

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Future Enhancement)

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📊 Database Management

### SQLite Operations

```bash
# View database schema
sqlite3 nutritional_chatbot.db ".schema"

# View tables
sqlite3 nutritional_chatbot.db ".tables"

# Query profiles
sqlite3 nutritional_chatbot.db "SELECT * FROM user_profiles;"
```

### Backup & Restore

```bash
# Backup
sqlite3 nutritional_chatbot.db ".backup backup.db"

# Restore
sqlite3 nutritional_chatbot.db ".restore backup.db"
```

## 🔍 Monitoring

### Health Endpoints

- `/health` - Basic health check
- `/api/status` - Detailed component status

### Logging

The application logs to console with different levels:

- INFO: General operation information
- WARNING: Non-critical issues
- ERROR: Error conditions
- DEBUG: Detailed debugging information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   ```

2. **spaCy Model Missing**

   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Database Issues**

   ```bash
   # Delete and recreate database
   rm nutritional_chatbot.db
   python main.py
   ```

4. **Port Already in Use**
   ```bash
   # Use different port
   uvicorn main:app --port 8001
   ```

### Getting Help

- Check the logs for detailed error messages
- Run `python test_backend.py` to validate setup
- Visit `/docs` for interactive API documentation
- Review the database schema in `/api/status`

## 🎯 Next Steps

1. Install dependencies and test the backend
2. Set up Neo4j for knowledge graph features
3. Create frontend client to interact with API
4. Add more sophisticated NLP models
5. Implement vector search capabilities
6. Add comprehensive testing suite
7. Deploy to cloud infrastructure
