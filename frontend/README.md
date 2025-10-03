# Nutritional AI Chatbot - Frontend Deployment Guide

## Overview

This directory contains the Streamlit frontend for the Nutritional AI Chatbot. The interface provides a modern, user-friendly web application for interacting with the advanced AI-powered nutrition system.

## Features

### 🏠 **Home Dashboard**

- System overview and health status
- Feature availability indicators
- Quick start guide and navigation

### 💬 **Enhanced Chat Interface**

- Real-time AI-powered conversations
- Personalized nutrition advice
- Message history and session management
- Quick suggestion buttons
- Typing indicators and response metadata

### 👤 **User Profile Management**

- Comprehensive profile creation wizard
- Age, culture, dietary preferences
- Health conditions and allergies
- Goals and response style preferences
- Profile validation and storage

### 🔍 **Knowledge Base Explorer**

- Advanced search capabilities (hybrid, vector, graph, text search)
- Entity exploration and filtering
- Interactive graph visualization
- Text analysis and NLP insights
- Export and sharing features

### ⚙️ **Admin Dashboard**

- Document upload and management
- System monitoring and health checks
- User analytics (planned)
- Configuration management
- Performance metrics

### 📊 **System Status**

- Real-time service health monitoring
- Performance metrics and charts
- Configuration summaries
- Error tracking and diagnostics

## Architecture

```
frontend/
├── streamlit_app.py          # Main application entry point
├── pages/                    # Page components
│   ├── profile.py           # User authentication & profiles
│   ├── chat.py              # Enhanced chat interface
│   ├── knowledge.py         # Knowledge base explorer
│   └── admin.py             # Admin dashboard
├── utils/                    # Utilities and components
│   └── ui_components.py     # Reusable UI components
├── .streamlit/              # Streamlit configuration
│   └── config.toml          # App configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation & Setup

### 1. Prerequisites

- **Python 3.8+**
- **Backend API running** (see backend/README.md)
- **Network access** to backend API (default: http://localhost:8000)

### 2. Environment Setup

```bash
# Navigate to frontend directory
cd frontend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

The frontend automatically connects to the backend API. Default configuration:

- **Backend URL**: `http://localhost:8000`
- **Frontend Port**: `8501`

To customize, edit the `API_BASE_URL` in `streamlit_app.py`:

```python
API_BASE_URL = "http://your-backend-host:port"
```

### 4. Run the Application

```bash
# Start Streamlit app
streamlit run streamlit_app.py

# Or with custom configuration
streamlit run streamlit_app.py --server.port 8501
```

The application will open in your browser at `http://localhost:8501`

## Usage Guide

### Getting Started

1. **Start the Backend**: Ensure the FastAPI backend is running first
2. **Launch Frontend**: Run the Streamlit application
3. **Create Profile**: Navigate to Profile page and create your nutrition profile
4. **Start Chatting**: Use the Chat page for personalized nutrition advice
5. **Explore Knowledge**: Browse the knowledge base for detailed information

### User Workflow

1. **Registration/Login**

   - Create account with email and password
   - Set up comprehensive nutrition profile
   - Save preferences and health information

2. **Personalized Chat**

   - Ask nutrition questions in natural language
   - Receive AI-powered, personalized responses
   - View sources and follow-up suggestions
   - Access conversation history

3. **Knowledge Exploration**

   - Search nutrition database with multiple methods
   - Explore food entities and relationships
   - Visualize knowledge graph connections
   - Analyze text for nutritional insights

4. **Admin Functions** (Admin users only)
   - Upload nutrition documents
   - Monitor system performance
   - Manage user data and configurations

### Key Features

#### 🎯 **Personalization**

- Responses tailored to age, culture, and health conditions
- Dietary preferences and allergy considerations
- Custom response styles (simple, detailed, motivational)

#### 🤖 **AI Integration**

- Multiple AI providers (OpenAI, Hugging Face)
- Fallback mechanisms for reliability
- Confidence scores and source attribution

#### 🔍 **Advanced Search**

- Hybrid search combining multiple techniques
- Vector similarity search for semantic understanding
- Graph-based relationship exploration
- Traditional text search for exact matches

#### 📊 **Rich Visualizations**

- Interactive charts and graphs
- Network visualizations for knowledge relationships
- Performance metrics and system health dashboards
- Export capabilities for data and insights

## API Integration

The frontend communicates with the backend through REST API calls:

### Authentication Endpoints

- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/profile/` - Create/update profile

### Chat Endpoints

- `POST /api/chat/enhanced_message` - Send chat message
- `GET /api/chat/history` - Retrieve chat history

### Knowledge Base Endpoints

- `GET /api/knowledge/search` - Search knowledge base
- `GET /api/knowledge/entities` - Get entities
- `GET /api/knowledge/graph` - Graph visualization data
- `POST /api/knowledge/analyze` - Text analysis

### Admin Endpoints

- `POST /api/admin/upload` - Upload documents
- `GET /api/admin/documents` - List documents
- `POST /api/admin/process/{id}` - Process document

### System Endpoints

- `GET /health` - Health check
- `GET /api/status` - Detailed system status

## Customization

### UI Theming

The application uses a green/health-focused color scheme. To customize:

1. **Edit Streamlit Config** (`.streamlit/config.toml`):

```toml
[theme]
primaryColor = "#YOUR_COLOR"
backgroundColor = "#YOUR_BG_COLOR"
```

2. **Modify CSS** (`utils/ui_components.py`):

```python
# Update color variables in UIComponents.load_custom_css()
```

### Adding New Pages

1. **Create Page File**: Add new file in `pages/` directory
2. **Implement Render Function**: Create `render_page_name()` function
3. **Update Main App**: Add page to navigation in `streamlit_app.py`
4. **Add Route**: Include routing logic in `main()` function

### Custom Components

Use the `UIComponents` class for consistent styling:

```python
from utils.ui_components import UIComponents

# Create info card
UIComponents.create_info_card("Title", "Content", "🎯")

# Create metric card
UIComponents.create_metric_card("Users", "1,234", "Active users", "👥")

# Create status badge
status_html = UIComponents.create_status_badge("healthy", "System Online")
st.markdown(status_html, unsafe_allow_html=True)
```

## Deployment

### Local Deployment

```bash
# Development mode with auto-reload
streamlit run streamlit_app.py --server.runOnSave true

# Production mode
streamlit run streamlit_app.py --server.enableCORS false
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment

#### Streamlit Cloud

1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Configure environment variables
4. Deploy with automatic updates

#### Heroku

```bash
# Create Procfile
echo "web: sh setup.sh && streamlit run streamlit_app.py" > Procfile

# Create setup.sh
echo "mkdir -p ~/.streamlit/
echo \"[server]
headless = true
port = \$PORT
enableCORS = false
\" > ~/.streamlit/config.toml" > setup.sh

# Deploy
heroku create your-app-name
git push heroku main
```

#### AWS/GCP/Azure

- Use Docker container deployment
- Configure load balancer for multiple instances
- Set up environment variables for API connections
- Enable SSL/HTTPS for production

## Performance Optimization

### Caching

- Streamlit's `@st.cache_data` for API responses
- Session state for user data persistence
- Browser caching for static assets

### Memory Management

- Limit chat history storage
- Paginated data loading for large datasets
- Efficient DataFrame operations

### Network Optimization

- Connection pooling for API requests
- Request timeout configuration
- Retry mechanisms for failed requests

## Troubleshooting

### Common Issues

#### Backend Connection Failed

```
Error: Connection error: Connection refused
```

**Solution**: Ensure backend is running on correct port (default: 8000)

#### Import Errors

```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution**: Install requirements: `pip install -r requirements.txt`

#### Slow Performance

**Solutions**:

- Clear browser cache
- Check network connection to backend
- Optimize data queries and caching
- Reduce concurrent API calls

#### UI Elements Not Loading

**Solutions**:

- Check browser console for JavaScript errors
- Disable browser extensions
- Try different browser
- Clear Streamlit cache: `streamlit cache clear`

### Debug Mode

Enable debug logging:

```bash
# Set debug environment variable
export STREAMLIT_LOG_LEVEL=debug

# Run with verbose logging
streamlit run streamlit_app.py --logger.level debug
```

### Health Checks

The application includes built-in health monitoring:

- Backend connectivity checks every 30 seconds
- Service status indicators in sidebar
- Error reporting and fallback mechanisms
- System status page with detailed diagnostics

## Contributing

### Development Setup

1. **Fork Repository**: Create your own fork
2. **Create Branch**: `git checkout -b feature/your-feature`
3. **Install Dev Dependencies**: `pip install -r requirements.txt`
4. **Make Changes**: Follow coding standards
5. **Test Changes**: Ensure all features work
6. **Submit PR**: Create pull request with description

### Coding Standards

- **Python**: Follow PEP 8 guidelines
- **Documentation**: Add docstrings for functions
- **Comments**: Explain complex logic
- **Error Handling**: Implement proper exception handling
- **Testing**: Add tests for new features

### Component Guidelines

- **Reusability**: Create reusable components in `utils/`
- **Consistency**: Use established UI patterns
- **Accessibility**: Ensure components are accessible
- **Performance**: Optimize for speed and memory usage

## Support

For support and issues:

1. **Documentation**: Check this README and inline comments
2. **GitHub Issues**: Report bugs and feature requests
3. **Backend Integration**: Ensure backend is properly configured
4. **Community**: Join discussions and share feedback

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Nutritional AI Chatbot Frontend** - Empowering personalized nutrition through advanced AI and intuitive user experience! 🥗🤖
