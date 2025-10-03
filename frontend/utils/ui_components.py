"""
UI Components and Utilities for Nutritional AI Chatbot Streamlit Frontend
Reusable components, custom styling, and utility functions
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
import base64


class UIComponents:
    """Reusable UI components for the application"""

    @staticmethod
    def load_custom_css():
        """Load enhanced custom CSS for better styling"""
        st.markdown(
            """
        <style>
            /* Main app styling */
            .main > div {
                padding-top: 1rem;
            }
            
            /* Header styling */
            .app-header {
                background: linear-gradient(90deg, #4CAF50 0%, #45A049 100%);
                padding: 2rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
            }
            
            .app-header h1 {
                color: white !important;
                margin-bottom: 0.5rem;
            }
            
            .app-header p {
                color: rgba(255, 255, 255, 0.9);
                font-size: 1.1rem;
            }
            
            /* Card components */
            .info-card {
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin: 1rem 0;
                border-left: 4px solid #4CAF50;
            }
            
            .metric-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 1.5rem;
                border-radius: 10px;
                text-align: center;
                margin: 0.5rem 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            .metric-card h3 {
                color: #2c3e50;
                margin-bottom: 1rem;
            }
            
            .metric-card .metric-value {
                font-size: 2rem;
                font-weight: bold;
                color: #4CAF50;
            }
            
            /* Chat message styling */
            .chat-message {
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 10px;
                max-width: 85%;
            }
            
            .user-message {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                margin-left: auto;
                margin-right: 0;
            }
            
            .assistant-message {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                margin-left: 0;
                margin-right: auto;
            }
            
            /* Status indicators */
            .status-indicator {
                display: inline-block;
                padding: 0.3rem 0.8rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: bold;
                text-transform: uppercase;
            }
            
            .status-healthy {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .status-warning {
                background-color: #fff3cd;
                color: #856404;
                border: 1px solid #ffeaa7;
            }
            
            .status-error {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            /* Navigation styling */
            .nav-item {
                padding: 0.8rem 1.2rem;
                margin: 0.2rem 0;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .nav-item:hover {
                background-color: #f0f2f6;
                transform: translateX(5px);
            }
            
            .nav-item.active {
                background-color: #4CAF50;
                color: white;
            }
            
            /* Form styling */
            .form-container {
                background: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin: 1rem 0;
            }
            
            .form-section {
                margin-bottom: 2rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid #eee;
            }
            
            .form-section:last-child {
                border-bottom: none;
            }
            
            /* Button styling */
            .stButton > button {
                border-radius: 20px;
                border: none;
                padding: 0.5rem 1.5rem;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            
            /* Primary button */
            .stButton > button[kind="primary"] {
                background: linear-gradient(45deg, #4CAF50, #45A049);
                color: white;
            }
            
            /* Secondary button */
            .stButton > button[kind="secondary"] {
                background: linear-gradient(45deg, #607D8B, #546E7A);
                color: white;
            }
            
            /* Alert styling */
            .custom-alert {
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                border-left: 4px solid;
            }
            
            .alert-success {
                background-color: #d4edda;
                border-left-color: #28a745;
                color: #155724;
            }
            
            .alert-warning {
                background-color: #fff3cd;
                border-left-color: #ffc107;
                color: #856404;
            }
            
            .alert-error {
                background-color: #f8d7da;
                border-left-color: #dc3545;
                color: #721c24;
            }
            
            .alert-info {
                background-color: #cce7ff;
                border-left-color: #007bff;
                color: #004085;
            }
            
            /* Progress bar */
            .progress-container {
                background-color: #f0f0f0;
                border-radius: 10px;
                padding: 0.2rem;
            }
            
            .progress-bar {
                background: linear-gradient(90deg, #4CAF50, #45A049);
                height: 1rem;
                border-radius: 8px;
                transition: width 0.3s ease;
            }
            
            /* Sidebar enhancements */
            .sidebar-logo {
                text-align: center;
                padding: 1rem;
                margin-bottom: 1rem;
            }
            
            .sidebar-section {
                margin: 1rem 0;
                padding: 1rem;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            
            /* Data table styling */
            .stDataFrame {
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            /* Hide Streamlit elements */
            .stDeployButton {
                display: none;
            }
            
            footer {
                visibility: hidden;
            }
            
            .stActionButton > button {
                background-color: transparent;
                border: none;
            }
            
            /* Loading spinner */
            .loading-container {
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 2rem;
            }
            
            .loading-spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #4CAF50;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                .main > div {
                    padding: 1rem;
                }
                
                .app-header {
                    padding: 1rem;
                }
                
                .metric-card {
                    margin: 0.5rem 0;
                }
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def create_info_card(title: str, content: str, icon: str = "ℹ️"):
        """Create an information card component"""
        st.markdown(
            f"""
        <div class="info-card">
            <h3>{icon} {title}</h3>
            <p>{content}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def create_metric_card(
        title: str, value: str, description: str = "", icon: str = "📊"
    ):
        """Create a metric card component"""
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>{icon} {title}</h3>
            <div class="metric-value">{value}</div>
            {f'<p>{description}</p>' if description else ''}
        </div>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def create_status_badge(status: str, text: str = None):
        """Create a status badge"""
        if text is None:
            text = status

        status_class = {
            "healthy": "status-healthy",
            "warning": "status-warning",
            "error": "status-error",
            "online": "status-healthy",
            "offline": "status-error",
            "active": "status-healthy",
            "inactive": "status-error",
        }.get(status.lower(), "status-warning")

        return f'<span class="status-indicator {status_class}">{text}</span>'

    @staticmethod
    def create_progress_bar(progress: float, label: str = ""):
        """Create a progress bar"""
        progress_percent = min(100, max(0, progress * 100))

        st.markdown(
            f"""
        <div class="progress-container">
            <div class="progress-bar" style="width: {progress_percent}%"></div>
        </div>
        {f'<p style="text-align: center; margin-top: 0.5rem;">{label}</p>' if label else ''}
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def create_alert(message: str, alert_type: str = "info", dismissible: bool = False):
        """Create custom alert"""
        alert_class = f"alert-{alert_type}"
        dismiss_button = (
            """
        <button onclick="this.parentElement.style.display='none'" 
                style="float: right; background: none; border: none; font-size: 1.2rem; cursor: pointer;">
            ×
        </button>
        """
            if dismissible
            else ""
        )

        st.markdown(
            f"""
        <div class="custom-alert {alert_class}">
            {dismiss_button}
            {message}
        </div>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def create_feature_showcase(features: List[Dict[str, str]]):
        """Create a feature showcase grid"""
        cols = st.columns(min(len(features), 3))

        for i, feature in enumerate(features):
            with cols[i % len(cols)]:
                UIComponents.create_info_card(
                    feature.get("title", "Feature"),
                    feature.get("description", ""),
                    feature.get("icon", "⭐"),
                )


class DataVisualization:
    """Data visualization utilities"""

    @staticmethod
    def create_service_health_chart(services: Dict[str, bool]):
        """Create service health visualization"""
        health_data = []
        for service, status in services.items():
            health_data.append(
                {
                    "Service": service.replace("_", " ").title(),
                    "Status": "Online" if status else "Offline",
                    "Health": 1 if status else 0,
                }
            )

        df = pd.DataFrame(health_data)

        fig = px.bar(
            df,
            x="Service",
            y="Health",
            color="Status",
            title="Service Health Status",
            color_discrete_map={"Online": "#4CAF50", "Offline": "#f44336"},
        )

        fig.update_layout(
            showlegend=True,
            yaxis_title="Status",
            yaxis=dict(
                tickmode="array", tickvals=[0, 1], ticktext=["Offline", "Online"]
            ),
        )

        return fig

    @staticmethod
    def create_feature_status_pie(features: Dict[str, bool]):
        """Create feature status pie chart"""
        active_count = sum(1 for status in features.values() if status)
        inactive_count = len(features) - active_count

        fig = px.pie(
            values=[active_count, inactive_count],
            names=["Active", "Inactive"],
            title="Feature Status Distribution",
            color_discrete_map={"Active": "#4CAF50", "Inactive": "#ff9800"},
        )

        return fig

    @staticmethod
    def create_performance_timeline(data: List[Dict]):
        """Create performance timeline chart"""
        df = pd.DataFrame(data)

        fig = px.line(
            df,
            x="timestamp",
            y="response_time",
            title="API Performance Timeline",
            labels={"response_time": "Response Time (s)", "timestamp": "Time"},
        )

        fig.update_traces(line_color="#4CAF50", line_width=3)
        fig.update_layout(hovermode="x unified", showlegend=False)

        return fig


class FormHelpers:
    """Form helper utilities"""

    @staticmethod
    def create_profile_form_sections():
        """Create structured profile form sections"""
        sections = {
            "Personal Information": [
                {
                    "type": "number",
                    "key": "age",
                    "label": "Age",
                    "min_value": 1,
                    "max_value": 120,
                },
                {
                    "type": "selectbox",
                    "key": "gender",
                    "label": "Gender",
                    "options": ["Male", "Female", "Other", "Prefer not to say"],
                },
                {"type": "text", "key": "location", "label": "Location"},
            ],
            "Health & Lifestyle": [
                {
                    "type": "selectbox",
                    "key": "activity_level",
                    "label": "Activity Level",
                    "options": [
                        "Sedentary",
                        "Lightly Active",
                        "Moderately Active",
                        "Very Active",
                    ],
                },
                {
                    "type": "multiselect",
                    "key": "health_conditions",
                    "label": "Health Conditions",
                    "options": ["None", "Diabetes", "Hypertension", "High Cholesterol"],
                },
                {
                    "type": "multiselect",
                    "key": "allergies",
                    "label": "Allergies",
                    "options": ["None", "Nuts", "Dairy", "Gluten", "Shellfish"],
                },
            ],
            "Dietary Preferences": [
                {
                    "type": "selectbox",
                    "key": "dietary_preference",
                    "label": "Diet Type",
                    "options": ["Omnivore", "Vegetarian", "Vegan", "Pescatarian"],
                },
                {
                    "type": "selectbox",
                    "key": "culture",
                    "label": "Cultural Background",
                    "options": [
                        "Western/American",
                        "Indian",
                        "Mediterranean",
                        "East Asian",
                    ],
                },
                {
                    "type": "multiselect",
                    "key": "health_goals",
                    "label": "Health Goals",
                    "options": [
                        "Weight Loss",
                        "Weight Gain",
                        "Muscle Building",
                        "General Health",
                    ],
                },
            ],
        }

        return sections

    @staticmethod
    def render_form_section(section_title: str, fields: List[Dict]):
        """Render a form section with validation"""
        st.markdown(f"### {section_title}")

        form_data = {}

        for field in fields:
            field_type = field["type"]
            key = field["key"]
            label = field["label"]

            if field_type == "text":
                form_data[key] = st.text_input(label, key=f"form_{key}")
            elif field_type == "number":
                form_data[key] = st.number_input(
                    label,
                    min_value=field.get("min_value", 0),
                    max_value=field.get("max_value", 100),
                    key=f"form_{key}",
                )
            elif field_type == "selectbox":
                form_data[key] = st.selectbox(
                    label, field["options"], key=f"form_{key}"
                )
            elif field_type == "multiselect":
                form_data[key] = st.multiselect(
                    label, field["options"], key=f"form_{key}"
                )

        return form_data


class UtilityFunctions:
    """General utility functions"""

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        size = float(size_bytes)

        while size >= 1024.0 and i < len(size_names) - 1:
            size /= 1024.0
            i += 1

        return f"{size:.1f} {size_names[i]}"

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human readable format"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    @staticmethod
    def get_health_color(status: bool) -> str:
        """Get color for health status"""
        return "#4CAF50" if status else "#f44336"

    @staticmethod
    def truncate_text(text: str, max_length: int = 100) -> str:
        """Truncate text with ellipsis"""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    @staticmethod
    def create_download_link(data: str, filename: str, link_text: str = "Download"):
        """Create a download link for data"""
        b64 = base64.b64encode(data.encode()).decode()
        return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'


# Apply custom styling when module is imported
UIComponents.load_custom_css()
