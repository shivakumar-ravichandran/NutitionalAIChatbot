"""
Interactive Chat Interface for Nutritional AI Chatbot
Enhanced chat experience with message history, typing indicators, and smart features
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import time
import uuid

# Import our main app's API client
from streamlit_app import APIClient, API_BASE_URL


class ChatManager:
    """Handle chat operations and message management"""

    @staticmethod
    def initialize_chat_session():
        """Initialize a new chat session"""
        if (
            "current_session_id" not in st.session_state
            or st.session_state.current_session_id is None
        ):
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.session_start_time = datetime.now()

    @staticmethod
    def add_message(role: str, content: str, metadata: Dict = None):
        """Add a message to chat history"""
        message = {
            "id": str(uuid.uuid4()),
            "role": role,  # 'user' or 'assistant'
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata or {},
        }
        st.session_state.chat_history.append(message)

    @staticmethod
    def get_user_profile():
        """Get user profile for personalized responses"""
        if st.session_state.authenticated and st.session_state.user_data:
            return st.session_state.user_data.get("profile", {})
        return {}

    @staticmethod
    def send_general_message(message: str) -> Dict[str, Any]:
        """Send message to general chat endpoint for guest users"""
        payload = {
            "message": message,
            "session_id": st.session_state.current_session_id,
            "chat_history": [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.chat_history[
                    -5:
                ]  # Last 5 messages for context
            ],
            "use_enhanced_features": False,
            "include_suggestions": True,
            "include_sources": False,
        }

        try:
            # Try enhanced endpoint first, fallback to general if needed
            response = requests.post(
                f"{API_BASE_URL}/api/chat/general_message",
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                # Fallback response for demo purposes
                return {
                    "success": True,
                    "data": {
                        "response": "I'm here to help with general nutrition questions! For personalized advice, please create a profile.",
                        "suggestions": [
                            "What are the benefits of eating vegetables?",
                            "How can I eat more protein?",
                            "Tell me about healthy cooking methods",
                            "What foods boost immunity?",
                        ],
                    },
                }

        except requests.exceptions.RequestException:
            # Fallback response when API is not available
            return {
                "success": True,
                "data": {
                    "response": "I'd be happy to help with nutrition questions! While I can't access my full knowledge base right now, I can provide general nutrition guidance. For detailed personalized advice, please create a profile.",
                    "suggestions": [
                        "What makes a balanced meal?",
                        "How do I read nutrition labels?",
                        "What are essential nutrients?",
                        "Tell me about portion sizes",
                    ],
                },
            }

    @staticmethod
    def send_enhanced_message(message: str) -> Dict[str, Any]:
        """Send message to enhanced chat endpoint for authenticated users"""
        user_profile = ChatManager.get_user_profile()

        # Prepare enhanced payload
        payload = {
            "message": message,
            "session_id": st.session_state.current_session_id,
            "user_profile": user_profile,
            "chat_history": [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.chat_history[
                    -5:
                ]  # Last 5 messages for context
            ],
            "use_enhanced_features": True,
            "include_suggestions": True,
            "include_sources": True,
        }

        try:
            response = requests.post(
                f"{API_BASE_URL}/api/chat/enhanced_message",
                json=payload,
                timeout=45,  # Longer timeout for AI processing
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", str(response.status_code))
                except:
                    error_detail = f"HTTP {response.status_code}"
                return {"success": False, "error": error_detail}

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out. The AI is thinking hard - please try again.",
            }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @staticmethod
    def get_quick_suggestions() -> List[str]:
        """Get contextual quick suggestions based on user profile"""
        user_profile = ChatManager.get_user_profile()

        # General suggestions for guest users
        general_suggestions = [
            "What should I eat for breakfast?",
            "What foods are high in protein?",
            "Tell me about vitamins and minerals",
            "What are superfoods?",
            "How much water should I drink daily?",
            "What are healthy snack options?",
        ]

        # If no profile (guest user), return general suggestions
        if not user_profile:
            return general_suggestions[:4]  # Return first 4 general suggestions

        # Personalized suggestions for authenticated users
        base_suggestions = [
            "Help me plan a personalized meal",
            "What should I eat for my goals?",
            "Review my daily nutrition",
            "Suggest meals for my dietary preference",
        ]

        # Personalized suggestions based on profile
        if user_profile:
            age = user_profile.get("age", 25)
            dietary_pref = user_profile.get("dietary_preference", "")
            health_goals = user_profile.get("health_goals", [])
            allergies = user_profile.get("allergies", [])

            personalized = []

            # Age-based suggestions
            if age < 18:
                personalized.extend(
                    [
                        "What snacks are good for school?",
                        "Foods to help me grow taller and stronger",
                    ]
                )
            elif age > 60:
                personalized.extend(
                    [
                        "Foods for bone health and aging",
                        "Easy to digest nutritious meals",
                    ]
                )

            # Dietary preference suggestions
            if dietary_pref == "Vegetarian":
                personalized.append("Best vegetarian protein sources")
            elif dietary_pref == "Vegan":
                personalized.append("Complete vegan nutrition guide")

            # Health goal suggestions
            if "Weight Loss" in health_goals:
                personalized.append("Low calorie filling foods")
            if "Muscle Building" in health_goals:
                personalized.append("Post-workout nutrition tips")

            # Allergy-aware suggestions
            if "Dairy" in allergies:
                personalized.append("Dairy-free calcium sources")
            if "Gluten" in allergies:
                personalized.append("Gluten-free meal ideas")

            return personalized[:4] if personalized else base_suggestions

        return base_suggestions


def display_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with proper styling"""
    with st.container():
        col1, col2, col3 = st.columns([1, 10, 1] if is_user else [1, 10, 1])

        with col2:
            message_class = "user-message" if is_user else "assistant-message"
            avatar = "👤" if is_user else "🤖"

            # Message header
            timestamp = message["timestamp"]
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except:
                    timestamp = datetime.now()

            time_str = timestamp.strftime("%H:%M")

            st.markdown(
                f"""
            <div class="stChatMessage {message_class}">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.2rem; margin-right: 0.5rem;">{avatar}</span>
                    <strong>{"You" if is_user else "Nutritional AI"}</strong>
                    <span style="margin-left: auto; color: #666; font-size: 0.8rem;">{time_str}</span>
                </div>
                <div style="margin-left: 1.7rem;">
                    {message["content"]}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Display metadata for assistant messages
            if not is_user and message.get("metadata"):
                metadata = message["metadata"]

                # Sources
                if metadata.get("sources"):
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(metadata["sources"][:3], 1):
                            st.markdown(f"**{i}.** {source}")

                # Suggestions
                if metadata.get("suggestions"):
                    st.markdown("**💡 Follow-up suggestions:**")
                    cols = st.columns(min(len(metadata["suggestions"]), 3))
                    for i, suggestion in enumerate(metadata["suggestions"][:3]):
                        with cols[i]:
                            if st.button(
                                suggestion, key=f"suggestion_{message['id']}_{i}"
                            ):
                                st.session_state.suggested_message = suggestion
                                st.rerun()

                # Confidence and processing info
                if metadata.get("confidence_score"):
                    confidence = metadata["confidence_score"]
                    color = (
                        "green"
                        if confidence > 0.8
                        else "orange" if confidence > 0.6 else "red"
                    )
                    st.markdown(
                        f"""
                    <small style="color: {color};">
                        Confidence: {confidence:.1%} | 
                        Processing time: {metadata.get('processing_time', 'N/A')}
                    </small>
                    """,
                        unsafe_allow_html=True,
                    )


def show_chat_interface():
    """Display the main chat interface"""
    st.title("💬 Nutritional AI Chat")

    # Initialize chat session
    ChatManager.initialize_chat_session()

    # Authentication status indicator
    if not st.session_state.authenticated:
        # Guest mode - allow general nutrition chat
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info(
                "🌟 **Guest Mode**: Get general nutrition knowledge! Create a profile for personalized advice."
            )
        with col2:
            if st.button("Create Profile", type="primary", use_container_width=True):
                st.session_state.page = "👤 Profile"
                st.rerun()
        with col3:
            if st.button("Login", type="secondary", use_container_width=True):
                st.session_state.page = "👤 Profile"
                st.rerun()
    else:
        # Authenticated mode indicator
        st.success(
            f"🎯 **Personalized Mode**: Welcome back, {st.session_state.user_data.get('name', 'User')}!"
        )

    # Chat header with user info
    if st.session_state.user_data:
        profile = st.session_state.user_data.get("profile", {})
        if profile:
            with st.expander("🎯 Your Profile Summary", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"**Age:** {profile.get('age', 'N/A')}")
                with col2:
                    st.markdown(f"**Diet:** {profile.get('dietary_preference', 'N/A')}")
                with col3:
                    allergies = profile.get("allergies", ["None"])
                    allergy_text = (
                        ", ".join(allergies) if allergies != ["None"] else "None"
                    )
                    st.markdown(f"**Allergies:** {allergy_text}")
                with col4:
                    goals = profile.get("health_goals", ["General Health"])
                    st.markdown(f"**Goal:** {goals[0] if goals else 'N/A'}")

    # Chat history container
    chat_container = st.container()

    with chat_container:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                display_message(message, is_user=(message["role"] == "user"))
        else:
            # Welcome message - different for authenticated vs guest users
            if st.session_state.authenticated:
                welcome_text = """
                    Hi there! I'm your personalized nutrition assistant. I can help you with meal planning, 
                    nutritional advice, dietary questions, and health goals. 
                    
                    Thanks to your profile, I can provide advice tailored specifically to your needs, 
                    preferences, and health conditions. What would you like to know about nutrition today?
                """
            else:
                welcome_text = """
                    Hi there! I'm your nutrition assistant. I can help you with general nutrition knowledge, 
                    food facts, healthy eating tips, and answer your dietary questions.
                    
                    For personalized meal plans and advice tailored to your specific needs, health conditions, 
                    and goals, please create a profile or login. What would you like to know about nutrition today?
                """

            st.markdown(
                f"""
            <div class="stChatMessage assistant-message">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.2rem; margin-right: 0.5rem;">🤖</span>
                    <strong>Nutritional AI</strong>
                </div>
                <div style="margin-left: 1.7rem;">
                    {welcome_text}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Mode-specific information
    if not st.session_state.authenticated:
        with st.expander("ℹ️ About Guest Mode", expanded=False):
            st.markdown(
                """
            **In Guest Mode, you can:**
            - Ask general nutrition questions
            - Learn about food facts and healthy eating
            - Get basic dietary guidance
            - Explore nutrition concepts
            
            **Create a profile to unlock:**
            - Personalized meal plans
            - Advice tailored to your health goals
            - Dietary restriction considerations  
            - Nutrition tracking and progress monitoring
            """
            )

    # Quick suggestions
    st.markdown("### 💡 Quick Suggestions")
    suggestions = ChatManager.get_quick_suggestions()

    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"quick_suggestion_{i}"):
                st.session_state.suggested_message = suggestion
                st.rerun()

    # Message input
    st.markdown("### 💭 Your Message")

    # Handle suggested message
    default_message = ""
    if "suggested_message" in st.session_state:
        default_message = st.session_state.suggested_message
        del st.session_state.suggested_message

    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        # Different placeholder based on authentication status
        if st.session_state.authenticated:
            placeholder_text = (
                "Example: Plan a meal for my dietary goals and preferences"
            )
            label_text = "Ask me anything about your personalized nutrition..."
        else:
            placeholder_text = (
                "Example: What should I eat for breakfast to boost my energy?"
            )
            label_text = "Ask me about general nutrition..."

        message = st.text_area(
            label_text,
            value=default_message,
            placeholder=placeholder_text,
            height=100,
            max_chars=1000,
        )

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            send_button = st.form_submit_button("Send Message 🚀", type="primary")

        with col2:
            clear_button = st.form_submit_button("Clear Chat 🗑️")

        with col3:
            new_session_button = st.form_submit_button("New Session ✨")

        if send_button and message.strip():
            # Add user message to history
            ChatManager.add_message("user", message.strip())

            # Show typing indicator
            thinking_message = (
                "🤖 AI is thinking..."
                if st.session_state.authenticated
                else "🤖 Getting nutrition info..."
            )
            with st.spinner(thinking_message):
                # Choose appropriate chat method based on authentication
                if st.session_state.authenticated:
                    result = ChatManager.send_enhanced_message(message.strip())
                else:
                    result = ChatManager.send_general_message(message.strip())

            if result["success"]:
                response_data = result["data"]
                ai_response = response_data.get(
                    "response", "I'm sorry, I couldn't generate a response."
                )

                # Extract metadata
                metadata = {
                    "confidence_score": response_data.get("confidence_score", 0.0),
                    "processing_time": response_data.get("processing_time", "N/A"),
                    "sources": response_data.get("sources", []),
                    "suggestions": response_data.get("suggestions", []),
                    "retrieval_info": response_data.get("retrieval_info", {}),
                }

                # Add AI response to history
                ChatManager.add_message("assistant", ai_response, metadata)

            else:
                # Handle error
                error_message = (
                    f"I apologize, but I encountered an error: {result['error']}"
                )
                ChatManager.add_message("assistant", error_message, {"error": True})

            # Rerun to display new messages
            st.rerun()

        elif send_button:
            st.warning("Please enter a message before sending.")

        if clear_button:
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()

        if new_session_button:
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.session_start_time = datetime.now()
            st.success("New chat session started!")
            st.rerun()

    # Promotional message for guest users
    if not st.session_state.authenticated and st.session_state.chat_history:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info(
                "🌟 **Enjoying the chat?** Create a profile for personalized nutrition advice!"
            )
            if st.button(
                "Get Personalized Advice", type="primary", use_container_width=True
            ):
                st.session_state.page = "👤 Profile"
                st.rerun()


def show_chat_stats():
    """Display chat statistics and insights"""
    if not st.session_state.chat_history:
        st.info("Start chatting to see statistics!")
        return

    st.markdown("### 📊 Chat Statistics")

    col1, col2, col3, col4 = st.columns(4)

    # Calculate stats
    total_messages = len(st.session_state.chat_history)
    user_messages = len(
        [m for m in st.session_state.chat_history if m["role"] == "user"]
    )
    ai_messages = len(
        [m for m in st.session_state.chat_history if m["role"] == "assistant"]
    )

    # Session duration
    if "session_start_time" in st.session_state:
        duration = datetime.now() - st.session_state.session_start_time
        duration_str = f"{duration.seconds // 60}m {duration.seconds % 60}s"
    else:
        duration_str = "N/A"

    with col1:
        st.metric("Total Messages", total_messages)

    with col2:
        st.metric("Your Messages", user_messages)

    with col3:
        st.metric("AI Responses", ai_messages)

    with col4:
        st.metric("Session Duration", duration_str)

    # Topics discussed (simple keyword extraction)
    if st.session_state.chat_history:
        user_messages_text = " ".join(
            [m["content"] for m in st.session_state.chat_history if m["role"] == "user"]
        ).lower()

        # Common nutrition keywords
        nutrition_keywords = [
            "protein",
            "vitamin",
            "calorie",
            "fiber",
            "fat",
            "carbs",
            "sugar",
            "breakfast",
            "lunch",
            "dinner",
            "snack",
            "meal",
            "diet",
            "weight",
            "healthy",
            "nutrition",
            "food",
            "exercise",
            "energy",
        ]

        found_topics = [kw for kw in nutrition_keywords if kw in user_messages_text]

        if found_topics:
            st.markdown("**Topics Discussed:**")
            topic_cols = st.columns(min(len(found_topics), 5))
            for i, topic in enumerate(found_topics[:5]):
                with topic_cols[i]:
                    st.markdown(f"• {topic.title()}")


def render_chat_page():
    """Main function to render the chat page"""
    # Create tabs for different views
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Statistics"])

    with tab1:
        show_chat_interface()

    with tab2:
        show_chat_stats()
