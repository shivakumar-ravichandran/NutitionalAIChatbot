"""
Nutritional AI Chatbot - Streamlit Frontend Application
Modern, user-friendly interface for the enhanced AI chatbot system
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import time

# Page configuration
st.set_page_config(
    page_title="🥗 Nutritional AI Chatbot",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/shivakumar-ravichandran/NutitionalAIChatbot",
        "Report a bug": "https://github.com/shivakumar-ravichandran/NutitionalAIChatbot/issues",
        "About": "# Nutritional AI Chatbot\nYour personalized nutrition assistant powered by AI",
    },
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background-color: #e3f2fd !important;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: #f1f8e9 !important;
        margin-right: 2rem;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .status-healthy {
        color: #4caf50;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ff9800;
        font-weight: bold;
    }
    
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
    
    .chat-input {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
    }
    
    # Hide Streamlit default elements
    .stDeployButton {
        display: none;
    }
    
    footer {
        visibility: hidden;
    }
    
    .stActionButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .sidebar-content {
        padding: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Backend API configuration
API_BASE_URL = "http://localhost:8000"


class SessionState:
    """Manage Streamlit session state"""

    @staticmethod
    def initialize():
        """Initialize session state variables"""
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if "user_data" not in st.session_state:
            st.session_state.user_data = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = None
        if "page" not in st.session_state:
            st.session_state.page = "🏠 Home"
        if "api_status" not in st.session_state:
            st.session_state.api_status = None
        if "last_health_check" not in st.session_state:
            st.session_state.last_health_check = None


class APIClient:
    """Handle API communication with the backend"""

    @staticmethod
    def check_backend_health() -> Dict[str, Any]:
        """Check if backend is running and healthy"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                return {"status": "healthy", "data": response.json()}
            else:
                return {
                    "status": "unhealthy",
                    "error": f"Status code: {response.status_code}",
                }
        except requests.exceptions.RequestException as e:
            return {"status": "unavailable", "error": str(e)}

    @staticmethod
    def get_api_status() -> Dict[str, Any]:
        """Get detailed API status"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/status", timeout=5)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": f"Status code: {response.status_code}",
                }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def send_chat_message(message: str, user_profile: Dict = None) -> Dict[str, Any]:
        """Send message to enhanced chat endpoint"""
        try:
            payload = {
                "message": message,
                "user_profile": user_profile or {},
                "use_enhanced_features": True,
            }
            response = requests.post(
                f"{API_BASE_URL}/api/chat/enhanced_message", json=payload, timeout=30
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": f"Status code: {response.status_code}",
                }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def search_knowledge_base(
        query: str, search_type: str = "hybrid"
    ) -> Dict[str, Any]:
        """Search the knowledge base"""
        try:
            params = {"query": query, "search_type": search_type, "limit": 10}
            response = requests.get(
                f"{API_BASE_URL}/api/knowledge/search", params=params, timeout=15
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": f"Status code: {response.status_code}",
                }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}


def show_header():
    """Display the application header"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
        <div style="text-align: center;">
            <h1>🥗 Nutritional AI Chatbot</h1>
            <p style="font-size: 18px; color: #666;">
                Your personalized nutrition assistant powered by advanced AI
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def show_sidebar():
    """Display the sidebar with navigation and status"""
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

        # Logo and title
        st.image(
            "https://via.placeholder.com/150x80/4CAF50/white?text=🥗+NutriAI", width=150
        )

        # Navigation menu
        selected = option_menu(
            menu_title="Navigation",
            options=[
                "🏠 Home",
                "💬 Chat",
                "🔍 Knowledge",
                "👤 Profile",
                "⚙️ Admin",
                "📊 Status",
            ],
            icons=["house", "chat-dots", "search", "person", "gear", "bar-chart"],
            menu_icon="list",
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#4CAF50", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#4CAF50"},
            },
        )
        st.session_state.page = selected

        # System status
        st.markdown("---")
        st.subheader("System Status")

        # Check backend health periodically
        current_time = datetime.now()
        if (
            st.session_state.last_health_check is None
            or (current_time - st.session_state.last_health_check).seconds > 30
        ):

            with st.spinner("Checking system health..."):
                st.session_state.api_status = APIClient.check_backend_health()
                st.session_state.last_health_check = current_time

        # Display status
        status = st.session_state.api_status
        if status and status["status"] == "healthy":
            st.markdown("🟢 **Backend**: Healthy")
            if "data" in status and "services" in status["data"]:
                services = status["data"]["services"]
                healthy_count = sum(1 for s in services.values() if s)
                total_count = len(services)
                st.markdown(f"📊 **Services**: {healthy_count}/{total_count} online")
        elif status and status["status"] == "unhealthy":
            st.markdown("🟡 **Backend**: Issues detected")
        else:
            st.markdown("🔴 **Backend**: Unavailable")

        # User status
        if st.session_state.authenticated:
            st.markdown("🟢 **User**: Authenticated")
            if st.session_state.user_data:
                st.markdown(
                    f"👤 **Profile**: {st.session_state.user_data.get('name', 'User')}"
                )
        else:
            st.markdown("🔴 **User**: Not logged in")

        st.markdown("</div>", unsafe_allow_html=True)


def show_home_page():
    """Display the home page"""
    st.title("Welcome to Nutritional AI Chatbot! 🥗")

    # Quick access options for new users
    if not st.session_state.authenticated:
        st.markdown("### 🚀 Get Started")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <h3 style="color: white;">🌟 Try Guest Mode</h3>
                <p style="color: white;">Start chatting immediately! Get general nutrition knowledge and healthy eating tips without creating an account.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Start Chat as Guest", type="primary", use_container_width=True
            ):
                st.session_state.page = "💬 Chat"
                st.rerun()

        with col2:
            st.markdown(
                """
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3 style="color: white;">🎯 Get Personalized</h3>
                <p style="color: white;">Create a profile for personalized meal plans, dietary advice tailored to your health goals and preferences.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            if st.button("Create Profile", type="secondary", use_container_width=True):
                st.session_state.page = "👤 Profile"
                st.rerun()

    # Feature highlights
    st.markdown("### ✨ Key Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <h3>🧠 AI-Powered</h3>
            <p>Advanced LLM integration with personalized responses based on your profile, age, and cultural preferences.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <h3>🔍 Smart Search</h3>
            <p>Hybrid retrieval system combining knowledge graphs and vector embeddings for accurate information.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <h3>🎯 Personalized</h3>
            <p>Tailored advice considering your age, culture, dietary preferences, and health conditions.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Quick stats
    st.subheader("System Overview")

    # Get API status for detailed info
    api_status = APIClient.get_api_status()

    if api_status.get("success"):
        data = api_status["data"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("API Version", data.get("version", "Unknown"), delta=None)

        with col2:
            services = data.get("services", {})
            healthy_services = sum(1 for s in services.values() if s)
            total_services = len(services)
            st.metric(
                "Services Online",
                f"{healthy_services}/{total_services}",
                delta=(
                    f"{healthy_services-total_services}"
                    if healthy_services < total_services
                    else None
                ),
            )

        with col3:
            features = data.get("features", {})
            active_features = sum(1 for f in features.values() if f)
            total_features = len(features)
            st.metric(
                "Features Active", f"{active_features}/{total_features}", delta=None
            )

        with col4:
            environment = data.get("environment", "unknown")
            st.metric("Environment", environment.title(), delta=None)

        # Feature status
        st.subheader("Available Features")

        features = data.get("features", {})
        feature_names = {
            "enhanced_chat": "🤖 Enhanced Chat",
            "semantic_search": "🔍 Semantic Search",
            "graph_reasoning": "🌐 Graph Reasoning",
            "personalized_responses": "🎯 Personalized Responses",
            "caching": "⚡ Performance Caching",
        }

        feature_cols = st.columns(len(features))
        for i, (feature_key, status) in enumerate(features.items()):
            with feature_cols[i]:
                feature_name = feature_names.get(
                    feature_key, feature_key.replace("_", " ").title()
                )
                status_emoji = "✅" if status else "❌"
                st.markdown(f"**{status_emoji} {feature_name}**")

    else:
        st.error(
            "Unable to connect to backend API. Please check if the server is running."
        )

    # Getting started
    st.subheader("Getting Started")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### 🚀 Quick Start
        1. **Create Profile**: Set up your personal nutrition profile
        2. **Start Chatting**: Ask questions about nutrition and health
        3. **Explore Knowledge**: Search our comprehensive knowledge base
        4. **Get Personalized**: Receive advice tailored to your needs
        """
        )

    with col2:
        st.markdown(
            """
        ### 💡 Pro Tips
        - Be specific about your dietary preferences
        - Mention any allergies or health conditions
        - Ask follow-up questions for detailed explanations
        - Use the knowledge base for research
        """
        )

    # Call to action
    if not st.session_state.authenticated:
        st.info(
            "👆 Create your profile to get started with personalized nutrition advice!"
        )
        if st.button("Create Profile", type="primary"):
            st.session_state.page = "👤 Profile"
            st.rerun()


def show_status_page():
    """Display detailed status page"""
    st.title("📊 System Status & Analytics")

    # Get comprehensive status
    api_status = APIClient.get_api_status()
    health_status = APIClient.check_backend_health()

    if api_status.get("success") and health_status.get("status") == "healthy":
        data = api_status["data"]
        health_data = health_status.get("data", {})

        # Overall status
        st.success("🟢 All systems operational")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("API Version", data.get("version", "Unknown"))

        with col2:
            services = data.get("services", {})
            healthy_count = sum(1 for s in services.values() if s)
            total_count = len(services)
            st.metric("Services Online", f"{healthy_count}/{total_count}")

        with col3:
            features = data.get("features", {})
            active_count = sum(1 for f in features.values() if f)
            total_features = len(features)
            st.metric("Features Active", f"{active_count}/{total_features}")

        with col4:
            environment = data.get("environment", "unknown")
            st.metric("Environment", environment.title())

        # Service details
        st.subheader("🔧 Service Details")

        service_data = []
        service_names = {
            "database": "Database",
            "nlp": "NLP Processing",
            "graph": "Graph Database",
            "vector_embeddings": "Vector Search",
            "llm_integration": "AI Language Model",
            "hybrid_retrieval": "Hybrid Search",
            "dynamic_prompts": "Dynamic Prompts",
            "performance_optimization": "Performance Cache",
        }

        for service_key, status in services.items():
            service_data.append(
                {
                    "Service": service_names.get(
                        service_key, service_key.replace("_", " ").title()
                    ),
                    "Status": "🟢 Online" if status else "🔴 Offline",
                    "Health": "Healthy" if status else "Unavailable",
                }
            )

        df = pd.DataFrame(service_data)
        st.dataframe(df, use_container_width=True)

        # Configuration summary
        st.subheader("⚙️ Configuration Summary")

        config = data.get("configuration", {})
        if config:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Database Configuration:**")
                db_config = config.get("database", {})
                st.text(f"Host: {db_config.get('host', 'N/A')}")
                st.text(f"Database: {db_config.get('database', 'N/A')}")

                st.markdown("**Vector Configuration:**")
                vector_config = config.get("vector", {})
                st.text(f"Model: {vector_config.get('model_name', 'N/A')}")
                st.text(f"Dimensions: {vector_config.get('embedding_dim', 'N/A')}")

            with col2:
                st.markdown("**External Services:**")
                neo4j_config = config.get("neo4j", {})
                redis_config = config.get("redis", {})
                openai_config = config.get("openai", {})

                st.text(
                    f"Neo4j: {'✅ Enabled' if neo4j_config.get('enabled') else '❌ Disabled'}"
                )
                st.text(
                    f"Redis: {'✅ Enabled' if redis_config.get('enabled') else '❌ Disabled'}"
                )
                st.text(
                    f"OpenAI: {'✅ Enabled' if openai_config.get('enabled') else '❌ Disabled'}"
                )

        # Performance metrics visualization
        st.subheader("📈 Performance Metrics")

        # Simulate some performance data
        import random
        from datetime import timedelta

        time_range = pd.date_range(
            start=datetime.now() - timedelta(hours=6), end=datetime.now(), freq="30min"
        )

        response_times = [random.uniform(0.2, 1.5) for _ in time_range]

        fig = px.line(
            x=time_range,
            y=response_times,
            title="API Response Times (6 hours)",
            labels={"x": "Time", "y": "Response Time (seconds)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("🔴 System status unavailable - backend connection failed")

        if health_status.get("error"):
            st.error(f"Error details: {health_status['error']}")


def main():
    """Main application entry point"""
    # Initialize session state
    SessionState.initialize()

    # Show header
    show_header()

    # Show sidebar
    show_sidebar()

    # Route to appropriate page based on selection
    page = st.session_state.page

    try:
        if page == "🏠 Home":
            show_home_page()
        elif page == "💬 Chat":
            from pages.chat import render_chat_page

            render_chat_page()
        elif page == "🔍 Knowledge":
            from pages.knowledge import render_knowledge_page

            render_knowledge_page()
        elif page == "👤 Profile":
            from pages.profile import render_profile_page

            render_profile_page()
        elif page == "⚙️ Admin":
            from pages.admin import render_admin_page

            render_admin_page()
        elif page == "📊 Status":
            show_status_page()
    except ImportError as e:
        st.error(f"Error loading page: {e}")
        st.info("Please ensure all required dependencies are installed.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.info("Please refresh the page or contact support.")


if __name__ == "__main__":
    main()
