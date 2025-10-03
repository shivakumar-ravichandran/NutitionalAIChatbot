"""
Admin Dashboard for Nutritional AI Chatbot
Document upload, processing, system management, and monitoring tools
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
import io
import base64

# Import our main app's API client
from streamlit_app import APIClient, API_BASE_URL


class AdminManager:
    """Handle admin operations"""

    @staticmethod
    def authenticate_admin(username: str, password: str) -> Dict[str, Any]:
        """Authenticate admin user"""
        try:
            payload = {"username": username, "password": password}

            response = requests.post(
                f"{API_BASE_URL}/api/admin/login", json=payload, timeout=10
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": "Invalid credentials"}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @staticmethod
    def upload_document(
        file_data: bytes, filename: str, document_type: str
    ) -> Dict[str, Any]:
        """Upload document to the system"""
        try:
            files = {"file": (filename, file_data, "application/octet-stream")}
            data = {
                "document_type": document_type,
                "description": f"Uploaded via admin dashboard: {filename}",
            }

            response = requests.post(
                f"{API_BASE_URL}/api/admin/upload", files=files, data=data, timeout=60
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                error_detail = "Upload failed"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", str(response.status_code))
                except:
                    error_detail = f"HTTP {response.status_code}"
                return {"success": False, "error": error_detail}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @staticmethod
    def get_documents() -> Dict[str, Any]:
        """Get list of uploaded documents"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/admin/documents", timeout=15)

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": f"Status code: {response.status_code}",
                }

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @staticmethod
    def process_document(document_id: str) -> Dict[str, Any]:
        """Process uploaded document"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/admin/process/{document_id}", timeout=120
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": f"Status code: {response.status_code}",
                }

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            # Get API status
            status_response = requests.get(f"{API_BASE_URL}/api/status", timeout=10)
            health_response = requests.get(f"{API_BASE_URL}/health", timeout=10)

            system_stats = {}

            if status_response.status_code == 200:
                system_stats["status"] = status_response.json()

            if health_response.status_code == 200:
                system_stats["health"] = health_response.json()

            return {"success": True, "data": system_stats}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}


def show_admin_login():
    """Display admin login form"""
    st.title("🔐 Admin Login")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("admin_login"):
            st.markdown("### Administrator Access")

            username = st.text_input("Username", placeholder="admin")
            password = st.text_input(
                "Password", type="password", placeholder="Enter admin password"
            )

            login_button = st.form_submit_button("Login", type="primary")

            if login_button:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    with st.spinner("Authenticating..."):
                        result = AdminManager.authenticate_admin(username, password)

                    if result["success"]:
                        st.session_state.admin_authenticated = True
                        st.session_state.admin_data = result["data"]
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(f"Login failed: {result['error']}")


def show_document_management():
    """Display document management interface"""
    st.subheader("📄 Document Management")

    # Document upload section
    st.markdown("### Upload New Document")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt", "md"],
            help="Supported formats: PDF, DOCX, TXT, MD",
        )

    with col2:
        document_type = st.selectbox(
            "Document Type",
            [
                "nutrition_data",
                "age_guidance",
                "cultural_info",
                "food_availability",
                "health_conditions",
                "general_knowledge",
            ],
            help="Categorize the document for better processing",
        )

    if uploaded_file is not None:
        st.info(f"File selected: {uploaded_file.name} ({uploaded_file.size} bytes)")

        if st.button("Upload Document", type="primary"):
            with st.spinner("Uploading document..."):
                file_data = uploaded_file.read()
                result = AdminManager.upload_document(
                    file_data, uploaded_file.name, document_type
                )

            if result["success"]:
                st.success("Document uploaded successfully!")
                st.json(result["data"])
                st.rerun()
            else:
                st.error(f"Upload failed: {result['error']}")

    # Document list section
    st.markdown("### Uploaded Documents")

    if st.button("Refresh Document List"):
        st.rerun()

    with st.spinner("Loading documents..."):
        result = AdminManager.get_documents()

    if result["success"]:
        documents = result["data"].get("documents", [])

        if documents:
            # Create DataFrame for better display
            doc_data = []
            for doc in documents:
                doc_data.append(
                    {
                        "ID": doc.get("id", "N/A"),
                        "Filename": doc.get("filename", "Unknown"),
                        "Type": doc.get("document_type", "Unknown"),
                        "Status": doc.get("status", "Unknown"),
                        "Upload Date": doc.get("upload_date", "Unknown"),
                        "File Size": f"{doc.get('file_size', 0)} bytes",
                        "Processed": "✅" if doc.get("processed", False) else "❌",
                    }
                )

            df = pd.DataFrame(doc_data)

            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Documents", len(df))
            with col2:
                processed_count = df["Processed"].value_counts().get("✅", 0)
                st.metric("Processed", processed_count)
            with col3:
                st.metric("Document Types", df["Type"].nunique())
            with col4:
                total_size = sum(int(size.split()[0]) for size in df["File Size"])
                st.metric("Total Size", f"{total_size:,} bytes")

            # Document type distribution
            if len(df) > 0:
                fig = px.pie(df, names="Type", title="Document Type Distribution")
                st.plotly_chart(fig, use_container_width=True)

            # Document table
            st.dataframe(df, use_container_width=True)

            # Document processing
            st.markdown("### Process Documents")

            unprocessed_docs = [
                doc for doc in documents if not doc.get("processed", False)
            ]

            if unprocessed_docs:
                selected_doc = st.selectbox(
                    "Select document to process:",
                    options=[
                        f"{doc['filename']} (ID: {doc['id']})"
                        for doc in unprocessed_docs
                    ],
                )

                if selected_doc and st.button(
                    "Process Selected Document", type="primary"
                ):
                    doc_id = selected_doc.split("(ID: ")[1].rstrip(")")

                    with st.spinner(
                        "Processing document... This may take a few minutes."
                    ):
                        result = AdminManager.process_document(doc_id)

                    if result["success"]:
                        st.success("Document processed successfully!")
                        processing_info = result["data"]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Entities Extracted",
                                processing_info.get("entities_count", 0),
                            )
                        with col2:
                            st.metric(
                                "Processing Time",
                                f"{processing_info.get('processing_time', 'N/A')}s",
                            )

                        if processing_info.get("summary"):
                            st.markdown("**Processing Summary:**")
                            st.info(processing_info["summary"])

                        st.rerun()
                    else:
                        st.error(f"Processing failed: {result['error']}")
            else:
                st.info("All documents are processed!")

        else:
            st.info("No documents uploaded yet.")

    else:
        st.error(f"Failed to load documents: {result['error']}")


def show_system_monitoring():
    """Display system monitoring dashboard"""
    st.subheader("📊 System Monitoring")

    # Get system statistics
    with st.spinner("Loading system statistics..."):
        result = AdminManager.get_system_stats()

    if result["success"]:
        stats = result["data"]

        # API Status
        if "status" in stats:
            status_data = stats["status"]

            st.markdown("### API Status")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("API Version", status_data.get("version", "Unknown"))

            with col2:
                environment = status_data.get("environment", "unknown")
                st.metric("Environment", environment.title())

            with col3:
                services = status_data.get("services", {})
                healthy_services = sum(1 for s in services.values() if s)
                total_services = len(services)
                st.metric("Services Online", f"{healthy_services}/{total_services}")

            with col4:
                features = status_data.get("features", {})
                active_features = sum(1 for f in features.values() if f)
                total_features = len(features)
                st.metric("Features Active", f"{active_features}/{total_features}")

            # Service status details
            st.markdown("### Service Health")

            service_names = {
                "database": "🗄️ Database",
                "nlp": "🧠 NLP Service",
                "graph": "🌐 Graph Database",
                "vector_embeddings": "🔍 Vector Search",
                "llm_integration": "🤖 LLM Integration",
                "hybrid_retrieval": "🔗 Hybrid Retrieval",
                "dynamic_prompts": "📝 Dynamic Prompts",
                "performance_optimization": "⚡ Performance Cache",
            }

            cols = st.columns(4)
            for i, (service_key, status) in enumerate(services.items()):
                with cols[i % 4]:
                    service_name = service_names.get(
                        service_key, service_key.replace("_", " ").title()
                    )
                    status_emoji = "🟢" if status else "🔴"
                    status_text = "Online" if status else "Offline"
                    st.markdown(
                        f"**{status_emoji} {service_name}**<br>{status_text}",
                        unsafe_allow_html=True,
                    )

            # Feature status
            st.markdown("### Feature Status")

            feature_names = {
                "enhanced_chat": "💬 Enhanced Chat",
                "semantic_search": "🔍 Semantic Search",
                "graph_reasoning": "🧠 Graph Reasoning",
                "personalized_responses": "🎯 Personalized AI",
                "caching": "⚡ Performance Caching",
            }

            feature_cols = st.columns(len(features))
            for i, (feature_key, status) in enumerate(features.items()):
                with feature_cols[i]:
                    feature_name = feature_names.get(
                        feature_key, feature_key.replace("_", " ").title()
                    )
                    status_emoji = "✅" if status else "❌"
                    st.markdown(
                        f"**{status_emoji}**<br>{feature_name}", unsafe_allow_html=True
                    )

        # Health Check Details
        if "health" in stats:
            health_data = stats["health"]

            st.markdown("### Health Check Details")

            overall_status = health_data.get("status", "unknown")
            if overall_status == "healthy":
                st.success("🟢 System is healthy")
            elif overall_status == "unhealthy":
                st.warning("🟡 System has issues")
            else:
                st.error("🔴 System is unavailable")

            # Detailed health info
            if "info" in health_data:
                info = health_data["info"]

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Services", info.get("total_services", 0))
                    st.metric("Healthy Services", info.get("healthy_services", 0))

                with col2:
                    if "configuration" in info:
                        config = info["configuration"]
                        st.metric(
                            "Database Host",
                            config.get("database", {}).get("host", "N/A"),
                        )
                        st.metric("Environment", config.get("environment", "N/A"))

        # Performance metrics (simulated)
        st.markdown("### Performance Metrics")

        # Create sample performance data
        time_range = pd.date_range(
            start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq="H"
        )

        # Simulate response times and request counts
        import random

        response_times = [random.uniform(0.1, 2.0) for _ in time_range]
        request_counts = [random.randint(10, 100) for _ in time_range]

        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                x=time_range,
                y=response_times,
                title="Response Time (24h)",
                labels={"x": "Time", "y": "Response Time (s)"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                x=time_range[-12:],  # Last 12 hours
                y=request_counts[-12:],
                title="Request Volume (12h)",
                labels={"x": "Time", "y": "Requests"},
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error(f"Failed to load system statistics: {result['error']}")


def show_user_management():
    """Display user management interface"""
    st.subheader("👥 User Management")

    # Simulated user data (replace with actual API calls)
    st.info(
        "User management features will be implemented with user analytics from the backend."
    )

    # Sample user statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Users", "156")

    with col2:
        st.metric("Active Users (7d)", "89")

    with col3:
        st.metric("New Users (7d)", "12")

    with col4:
        st.metric("Avg Session Time", "8.5 min")

    # User activity chart
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30), end=datetime.now(), freq="D"
    )
    activity = [random.randint(20, 80) for _ in dates]

    fig = px.line(
        x=dates,
        y=activity,
        title="User Activity (30 days)",
        labels={"x": "Date", "y": "Active Users"},
    )
    st.plotly_chart(fig, use_container_width=True)


def show_configuration():
    """Display system configuration interface"""
    st.subheader("⚙️ System Configuration")

    st.info("Configuration management interface for system settings and parameters.")

    # Configuration categories
    tab1, tab2, tab3 = st.tabs(["API Settings", "AI Models", "Performance"])

    with tab1:
        st.markdown("### API Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.text_input("API Base URL", value=API_BASE_URL, disabled=True)
            st.number_input("Request Timeout", value=30, min_value=5, max_value=300)

        with col2:
            st.selectbox(
                "Environment", ["development", "staging", "production"], index=0
            )
            st.checkbox("Enable Debug Mode", value=False)

    with tab2:
        st.markdown("### AI Model Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.text_input("Vector Model", value="all-MiniLM-L6-v2")
            st.text_input("OpenAI Model", value="gpt-3.5-turbo")

        with col2:
            st.text_input("HuggingFace Model", value="microsoft/DialoGPT-medium")
            st.number_input("Max Tokens", value=1000, min_value=100, max_value=4000)

    with tab3:
        st.markdown("### Performance Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.number_input("Cache Size", value=1000, min_value=100, max_value=10000)
            st.number_input("Max Workers", value=4, min_value=1, max_value=16)

        with col2:
            st.number_input("Batch Size", value=32, min_value=1, max_value=128)
            st.checkbox("Enable Caching", value=True)


def render_admin_page():
    """Main function to render the admin page"""
    # Check admin authentication
    if not st.session_state.get("admin_authenticated", False):
        show_admin_login()
        return

    st.title("⚙️ Admin Dashboard")

    # Admin header
    if st.session_state.get("admin_data"):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown("### Welcome, Administrator! 👨‍💼")

        with col3:
            if st.button("🚪 Logout", type="secondary"):
                st.session_state.admin_authenticated = False
                st.session_state.admin_data = None
                st.rerun()

    # Admin navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📄 Documents", "📊 Monitoring", "👥 Users", "⚙️ Config", "🔧 Tools"]
    )

    with tab1:
        show_document_management()

    with tab2:
        show_system_monitoring()

    with tab3:
        show_user_management()

    with tab4:
        show_configuration()

    with tab5:
        st.subheader("🔧 Admin Tools")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Restart Services", type="secondary"):
                st.info("Service restart functionality would be implemented here.")

            if st.button("🗑️ Clear Cache", type="secondary"):
                st.info("Cache clearing functionality would be implemented here.")

        with col2:
            if st.button("📥 Export Data", type="secondary"):
                st.info("Data export functionality would be implemented here.")

            if st.button("🔍 Run Diagnostics", type="secondary"):
                st.info("System diagnostics would run here.")
