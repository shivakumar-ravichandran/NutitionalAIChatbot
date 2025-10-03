"""
User Authentication and Profile Management for Nutritional AI Chatbot
Handles user login, registration, and profile creation/editing
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
import re

# Import our main app's API client
from streamlit_app import APIClient, API_BASE_URL


class AuthenticationManager:
    """Handle user authentication operations"""

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return re.match(pattern, email) is not None

    @staticmethod
    def validate_password(password: str) -> tuple[bool, str]:
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"
        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"
        if not re.search(r"[0-9]", password):
            return False, "Password must contain at least one number"
        return True, "Password is valid"

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password for storage"""
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def register_user(user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new user"""
        try:
            # Hash password before sending
            user_data["password"] = AuthenticationManager.hash_password(
                user_data["password"]
            )

            response = requests.post(
                f"{API_BASE_URL}/api/auth/register", json=user_data, timeout=10
            )

            if response.status_code == 201:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": response.json().get("detail", "Registration failed"),
                }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @staticmethod
    def login_user(email: str, password: str) -> Dict[str, Any]:
        """Login user"""
        try:
            login_data = {
                "email": email,
                "password": AuthenticationManager.hash_password(password),
            }

            response = requests.post(
                f"{API_BASE_URL}/api/auth/login", json=login_data, timeout=10
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": response.json().get("detail", "Login failed"),
                }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}


class ProfileManager:
    """Handle user profile operations"""

    @staticmethod
    def create_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create user profile"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/profile/", json=profile_data, timeout=10
            )

            if response.status_code == 201:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": response.json().get("detail", "Profile creation failed"),
                }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @staticmethod
    def get_profile(user_id: str) -> Dict[str, Any]:
        """Get user profile"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/profile/{user_id}", timeout=10)

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": "Profile not found"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @staticmethod
    def update_profile(user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile"""
        try:
            response = requests.put(
                f"{API_BASE_URL}/api/profile/{user_id}", json=profile_data, timeout=10
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": response.json().get("detail", "Profile update failed"),
                }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}


def show_login_form():
    """Display login form"""
    st.subheader("🔑 Login to Your Account")

    with st.form("login_form"):
        email = st.text_input("Email Address", placeholder="your.email@example.com")
        password = st.text_input(
            "Password", type="password", placeholder="Enter your password"
        )

        col1, col2 = st.columns(2)
        with col1:
            login_button = st.form_submit_button("Login", type="primary")
        with col2:
            register_button = st.form_submit_button("Create Account")

        if login_button:
            if not email or not password:
                st.error("Please fill in all fields")
            elif not AuthenticationManager.validate_email(email):
                st.error("Please enter a valid email address")
            else:
                with st.spinner("Logging you in..."):
                    result = AuthenticationManager.login_user(email, password)

                if result["success"]:
                    st.session_state.authenticated = True
                    st.session_state.user_data = result["data"]
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error(f"Login failed: {result['error']}")

        if register_button:
            st.session_state.show_register = True
            st.rerun()


def show_register_form():
    """Display registration form"""
    st.subheader("📝 Create Your Account")

    with st.form("register_form"):
        col1, col2 = st.columns(2)

        with col1:
            first_name = st.text_input("First Name", placeholder="John")
            email = st.text_input("Email Address", placeholder="john@example.com")
            password = st.text_input(
                "Password", type="password", placeholder="Strong password"
            )

        with col2:
            last_name = st.text_input("Last Name", placeholder="Doe")
            username = st.text_input("Username", placeholder="johndoe")
            confirm_password = st.text_input(
                "Confirm Password", type="password", placeholder="Repeat password"
            )

        col1, col2 = st.columns(2)
        with col1:
            register_button = st.form_submit_button("Create Account", type="primary")
        with col2:
            back_button = st.form_submit_button("Back to Login")

        if register_button:
            # Validation
            errors = []
            if not all(
                [first_name, last_name, email, username, password, confirm_password]
            ):
                errors.append("Please fill in all fields")
            if not AuthenticationManager.validate_email(email):
                errors.append("Please enter a valid email address")
            if password != confirm_password:
                errors.append("Passwords do not match")

            password_valid, password_message = AuthenticationManager.validate_password(
                password
            )
            if not password_valid:
                errors.append(password_message)

            if errors:
                for error in errors:
                    st.error(error)
            else:
                user_data = {
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": email,
                    "username": username,
                    "password": password,
                }

                with st.spinner("Creating your account..."):
                    result = AuthenticationManager.register_user(user_data)

                if result["success"]:
                    st.success("Account created successfully! Please log in.")
                    st.session_state.show_register = False
                    st.rerun()
                else:
                    st.error(f"Registration failed: {result['error']}")

        if back_button:
            st.session_state.show_register = False
            st.rerun()


def show_profile_form():
    """Display profile creation/editing form"""
    st.subheader("👤 Your Nutrition Profile")

    # Get existing profile if available
    existing_profile = None
    if st.session_state.user_data and "id" in st.session_state.user_data:
        profile_result = ProfileManager.get_profile(st.session_state.user_data["id"])
        if profile_result["success"]:
            existing_profile = profile_result["data"]

    with st.form("profile_form"):
        st.markdown("### Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=existing_profile.get("age", 25) if existing_profile else 25,
            )

        with col2:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Other", "Prefer not to say"],
                index=(
                    ["Male", "Female", "Other", "Prefer not to say"].index(
                        existing_profile.get("gender", "Prefer not to say")
                    )
                    if existing_profile
                    else 3
                ),
            )

        with col3:
            activity_level = st.selectbox(
                "Activity Level",
                [
                    "Sedentary",
                    "Lightly Active",
                    "Moderately Active",
                    "Very Active",
                    "Extremely Active",
                ],
                index=(
                    [
                        "Sedentary",
                        "Lightly Active",
                        "Moderately Active",
                        "Very Active",
                        "Extremely Active",
                    ].index(existing_profile.get("activity_level", "Moderately Active"))
                    if existing_profile
                    else 2
                ),
            )

        col1, col2 = st.columns(2)

        with col1:
            location = st.text_input(
                "Location (City, State)",
                value=existing_profile.get("location", "") if existing_profile else "",
                placeholder="New York, NY",
            )

        with col2:
            culture = st.selectbox(
                "Cultural Background",
                [
                    "Western/American",
                    "Indian",
                    "Mediterranean",
                    "East Asian",
                    "Latin American",
                    "Middle Eastern",
                    "African",
                    "Other",
                ],
                index=(
                    0
                    if not existing_profile
                    else max(
                        0,
                        [
                            "Western/American",
                            "Indian",
                            "Mediterranean",
                            "East Asian",
                            "Latin American",
                            "Middle Eastern",
                            "African",
                            "Other",
                        ].index(existing_profile.get("culture", "Western/American")),
                    )
                ),
            )

        st.markdown("### Dietary Preferences & Restrictions")

        col1, col2 = st.columns(2)

        with col1:
            dietary_preference = st.selectbox(
                "Dietary Preference",
                ["Omnivore", "Vegetarian", "Vegan", "Pescatarian", "Flexitarian"],
                index=(
                    0
                    if not existing_profile
                    else max(
                        0,
                        [
                            "Omnivore",
                            "Vegetarian",
                            "Vegan",
                            "Pescatarian",
                            "Flexitarian",
                        ].index(existing_profile.get("dietary_preference", "Omnivore")),
                    )
                ),
            )

        with col2:
            # Common allergies
            common_allergies = [
                "None",
                "Nuts",
                "Dairy",
                "Gluten",
                "Shellfish",
                "Eggs",
                "Soy",
                "Fish",
                "Sesame",
            ]
            existing_allergies = (
                existing_profile.get("allergies", ["None"])
                if existing_profile
                else ["None"]
            )
            allergies = st.multiselect(
                "Allergies",
                common_allergies,
                default=(
                    existing_allergies
                    if isinstance(existing_allergies, list)
                    else ["None"]
                ),
            )

        # Health conditions
        st.markdown("### Health Information")

        common_conditions = [
            "None",
            "Diabetes",
            "Hypertension",
            "Heart Disease",
            "High Cholesterol",
            "Obesity",
            "Anemia",
            "Thyroid Issues",
        ]
        existing_conditions = (
            existing_profile.get("health_conditions", ["None"])
            if existing_profile
            else ["None"]
        )
        health_conditions = st.multiselect(
            "Health Conditions",
            common_conditions,
            default=(
                existing_conditions
                if isinstance(existing_conditions, list)
                else ["None"]
            ),
        )

        # Goals
        col1, col2 = st.columns(2)

        with col1:
            health_goals = st.multiselect(
                "Health Goals",
                [
                    "Weight Loss",
                    "Weight Gain",
                    "Muscle Building",
                    "General Health",
                    "Disease Management",
                    "Athletic Performance",
                    "Energy Boost",
                ],
                default=(
                    existing_profile.get("health_goals", ["General Health"])
                    if existing_profile
                    else ["General Health"]
                ),
            )

        with col2:
            response_style = st.selectbox(
                "Preferred Response Style",
                [
                    "Simple & Direct",
                    "Detailed & Educational",
                    "Motivational & Encouraging",
                ],
                index=(
                    0
                    if not existing_profile
                    else max(
                        0,
                        [
                            "Simple & Direct",
                            "Detailed & Educational",
                            "Motivational & Encouraging",
                        ].index(
                            existing_profile.get(
                                "response_style", "Detailed & Educational"
                            )
                        ),
                    )
                ),
            )

        # Submit button
        submit_button = st.form_submit_button("Save Profile", type="primary")

        if submit_button:
            profile_data = {
                "age": age,
                "gender": gender,
                "location": location,
                "culture": culture,
                "dietary_preference": dietary_preference,
                "allergies": allergies,
                "health_conditions": health_conditions,
                "health_goals": health_goals,
                "activity_level": activity_level,
                "response_style": response_style,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            if st.session_state.user_data:
                profile_data["user_id"] = st.session_state.user_data.get("id")

            with st.spinner("Saving your profile..."):
                if existing_profile:
                    result = ProfileManager.update_profile(
                        st.session_state.user_data["id"], profile_data
                    )
                else:
                    result = ProfileManager.create_profile(profile_data)

            if result["success"]:
                st.success("Profile saved successfully!")
                # Update session state with profile data
                if "user_data" not in st.session_state:
                    st.session_state.user_data = {}
                st.session_state.user_data["profile"] = profile_data
                st.rerun()
            else:
                st.error(f"Failed to save profile: {result['error']}")


def show_profile_page():
    """Main profile page logic"""
    if not st.session_state.authenticated:
        st.title("🔐 Authentication Required")

        # Toggle between login and register
        if "show_register" not in st.session_state:
            st.session_state.show_register = False

        if st.session_state.show_register:
            show_register_form()
        else:
            show_login_form()

    else:
        # User is authenticated, show profile
        st.title("👤 Your Profile")

        # User info header
        if st.session_state.user_data:
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.markdown(
                    f"""
                <div style="text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;">
                    <h3>Welcome, {st.session_state.user_data.get('first_name', 'User')}! 👋</h3>
                    <p>Manage your nutrition profile and preferences below.</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Logout button
        if st.sidebar.button("🚪 Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.user_data = None
            st.session_state.chat_history = []
            st.success("Logged out successfully!")
            st.rerun()

        # Profile form
        show_profile_form()

        # Profile summary
        if st.session_state.user_data and "profile" in st.session_state.user_data:
            st.markdown("---")
            st.subheader("📋 Profile Summary")

            profile = st.session_state.user_data["profile"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Age", f"{profile.get('age', 'N/A')} years")
                st.metric("Activity Level", profile.get("activity_level", "N/A"))

            with col2:
                st.metric(
                    "Dietary Preference", profile.get("dietary_preference", "N/A")
                )
                st.metric("Cultural Background", profile.get("culture", "N/A"))

            with col3:
                allergies = profile.get("allergies", ["None"])
                allergy_text = ", ".join(allergies) if allergies != ["None"] else "None"
                st.metric("Allergies", allergy_text)

                goals = profile.get("health_goals", ["General Health"])
                st.metric("Primary Goal", goals[0] if goals else "N/A")


# This function will be called from the main app
def render_profile_page():
    """Render the profile page"""
    show_profile_page()
