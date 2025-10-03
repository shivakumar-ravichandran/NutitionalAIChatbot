"""
Dynamic Prompt Generation Service for age-appropriate and culturally-aware responses
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AgeGroup(Enum):
    """Age group classifications"""

    CHILDREN = "children"  # 0-17
    YOUNG_ADULTS = "young_adults"  # 18-35
    ADULTS = "adults"  # 36-64
    ELDERLY = "elderly"  # 65+


class CulturalContext(Enum):
    """Cultural context classifications"""

    INDIAN = "indian"
    WESTERN = "western"
    ASIAN = "asian"
    MEDITERRANEAN = "mediterranean"
    MIDDLE_EASTERN = "middle_eastern"
    AFRICAN = "african"
    LATIN_AMERICAN = "latin_american"


@dataclass
class UserProfile:
    """User profile for prompt personalization"""

    age: Optional[int] = None
    culture: Optional[str] = None
    dietary_preferences: Optional[str] = None
    allergies: List[str] = None
    health_conditions: List[str] = None
    activity_level: Optional[str] = None
    response_style: Optional[str] = None
    language: str = "en"


@dataclass
class PromptTemplate:
    """Template structure for dynamic prompts"""

    system_prompt: str
    context_instructions: str
    tone_guidelines: str
    safety_instructions: str
    cultural_considerations: str
    response_format: str


class DynamicPromptService:
    """Service for generating dynamic, personalized prompts"""

    def __init__(self):
        self.age_templates = self._initialize_age_templates()
        self.cultural_templates = self._initialize_cultural_templates()
        self.dietary_templates = self._initialize_dietary_templates()
        self.health_templates = self._initialize_health_templates()
        self.style_templates = self._initialize_style_templates()

        logger.info("Dynamic prompt service initialized")

    def _initialize_age_templates(self) -> Dict[AgeGroup, Dict[str, Any]]:
        """Initialize age-specific prompt templates"""
        return {
            AgeGroup.CHILDREN: {
                "tone": "friendly, simple, encouraging, and fun",
                "vocabulary": "basic words, avoid complex medical terms",
                "structure": "short sentences, bullet points, use analogies",
                "safety_level": "high",
                "examples": "use food characters or fun comparisons",
                "disclaimers": "always mention talking to parents/guardians",
                "response_length": "concise, 100-200 words max",
                "system_prompt": """You are a friendly nutritional helper for children. Use simple, fun language and always encourage talking to parents about food choices. Keep explanations short and use analogies kids can understand.""",
            },
            AgeGroup.YOUNG_ADULTS: {
                "tone": "energetic, motivational, relatable, and modern",
                "vocabulary": "contemporary language, fitness terms, lifestyle-focused",
                "structure": "engaging paragraphs, practical tips, actionable advice",
                "safety_level": "moderate",
                "examples": "use fitness, sports, and lifestyle contexts",
                "disclaimers": "mention consulting professionals for specific conditions",
                "response_length": "moderate, 200-400 words",
                "system_prompt": """You are a modern nutritional guide for young adults. Focus on practical advice for busy lifestyles, fitness goals, and building healthy habits. Use contemporary language and relate to their active lifestyle.""",
            },
            AgeGroup.ADULTS: {
                "tone": "professional, balanced, informative, and supportive",
                "vocabulary": "standard terminology, professional language when appropriate",
                "structure": "well-organized information, clear explanations, evidence-based",
                "safety_level": "standard",
                "examples": "use work-life balance, family health contexts",
                "disclaimers": "standard healthcare consultation advice",
                "response_length": "comprehensive, 300-500 words",
                "system_prompt": """You are a knowledgeable nutritional advisor for adults. Provide balanced, evidence-based information that considers busy lifestyles, family responsibilities, and health maintenance goals.""",
            },
            AgeGroup.ELDERLY: {
                "tone": "respectful, patient, clear, and reassuring",
                "vocabulary": "clear language, avoid jargon, explain technical terms",
                "structure": "step-by-step explanations, organized sections, easy to follow",
                "safety_level": "high",
                "examples": "use age-appropriate health contexts, medication interactions",
                "disclaimers": "strongly emphasize healthcare provider consultation",
                "response_length": "detailed but clear, 250-450 words",
                "system_prompt": """You are a respectful nutritional counselor for older adults. Provide clear, detailed explanations with special attention to medication interactions, chronic conditions, and age-related nutritional needs.""",
            },
        }

    def _initialize_cultural_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize culture-specific prompt templates"""
        return {
            "indian": {
                "food_context": "Indian cuisine, spices, vegetarian options, regional variations",
                "dietary_patterns": "rice/wheat staples, dal, vegetables, dairy, seasonal eating",
                "cooking_methods": "traditional cooking, spice usage, fermentation",
                "health_traditions": "Ayurvedic principles, hot/cold foods, digestive health",
                "cultural_considerations": "vegetarian prevalence, religious dietary laws, family meals",
                "examples": "use familiar Indian foods like dal, roti, sabji, curd",
            },
            "western": {
                "food_context": "Western cuisine, processed foods, restaurant culture",
                "dietary_patterns": "meat consumption, dairy, processed foods, eating out",
                "cooking_methods": "baking, grilling, convenience foods",
                "health_traditions": "calorie counting, supplements, fitness culture",
                "cultural_considerations": "individualistic eating, convenience focus, diet trends",
                "examples": "use familiar foods like salads, sandwiches, pasta, grilled chicken",
            },
            "asian": {
                "food_context": "Asian cuisine varieties, rice-based diets, seafood, vegetables",
                "dietary_patterns": "rice staples, vegetables, seafood, fermented foods",
                "cooking_methods": "stir-frying, steaming, fermentation, minimal processing",
                "health_traditions": "balance concepts, tea culture, seasonal eating",
                "cultural_considerations": "communal dining, respect for elders' food choices",
                "examples": "use familiar foods like rice, vegetables, fish, tofu, tea",
            },
            "mediterranean": {
                "food_context": "Mediterranean diet, olive oil, fresh produce, seafood",
                "dietary_patterns": "olive oil, fish, vegetables, fruits, whole grains",
                "cooking_methods": "simple preparation, fresh ingredients, minimal processing",
                "health_traditions": "Mediterranean diet benefits, wine culture, seasonal eating",
                "cultural_considerations": "social dining, fresh local ingredients",
                "examples": "use foods like olive oil, fish, tomatoes, herbs, whole grains",
            },
        }

    def _initialize_dietary_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize dietary preference templates"""
        return {
            "vegetarian": {
                "focus": "plant-based proteins, B12, iron absorption, complete proteins",
                "alternatives": "suggest plant-based alternatives to animal products",
                "concerns": "address common vegetarian nutritional concerns",
                "examples": "use vegetarian food examples throughout",
            },
            "vegan": {
                "focus": "B12, protein combining, calcium, omega-3, vitamin D",
                "alternatives": "provide vegan alternatives for all suggestions",
                "concerns": "address potential nutrient deficiencies",
                "examples": "use only plant-based food examples",
            },
            "pescatarian": {
                "focus": "fish as primary animal protein, omega-3 benefits",
                "alternatives": "include fish and plant-based options",
                "concerns": "mercury levels, sustainable seafood",
                "examples": "include fish and vegetarian options",
            },
            "keto": {
                "focus": "low-carb, high-fat, ketosis, electrolyte balance",
                "alternatives": "low-carb alternatives to high-carb foods",
                "concerns": "nutrient density, sustainability, medical supervision",
                "examples": "focus on low-carb, high-fat foods",
            },
        }

    def _initialize_health_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize health condition templates"""
        return {
            "diabetes": {
                "focus": "blood sugar management, carbohydrate counting, glycemic index",
                "restrictions": "limit simple sugars, refined carbs",
                "recommendations": "complex carbs, fiber, protein timing",
                "monitoring": "blood glucose monitoring, medication timing",
                "disclaimer": "work closely with healthcare team for management",
            },
            "hypertension": {
                "focus": "sodium reduction, potassium increase, DASH diet principles",
                "restrictions": "limit processed foods, added salt",
                "recommendations": "fruits, vegetables, whole grains, lean proteins",
                "monitoring": "blood pressure tracking",
                "disclaimer": "coordinate with healthcare provider for medication interactions",
            },
            "heart_disease": {
                "focus": "heart-healthy fats, cholesterol management, omega-3s",
                "restrictions": "limit saturated fats, trans fats, sodium",
                "recommendations": "Mediterranean diet patterns, fish, nuts, olive oil",
                "monitoring": "cholesterol levels, weight management",
                "disclaimer": "follow cardiology team recommendations",
            },
            "kidney_disease": {
                "focus": "protein moderation, phosphorus, potassium, sodium control",
                "restrictions": "limit protein, phosphorus-rich foods, potassium",
                "recommendations": "work with renal dietitian",
                "monitoring": "lab values, fluid intake",
                "disclaimer": "strict adherence to renal diet essential",
            },
        }

    def _initialize_style_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize response style templates"""
        return {
            "simple": {
                "structure": "basic explanations, minimal technical terms",
                "length": "short and to the point",
                "examples": "simple, relatable examples",
            },
            "detailed": {
                "structure": "comprehensive explanations, include mechanisms",
                "length": "thorough coverage of topic",
                "examples": "detailed examples with context",
            },
            "motivational": {
                "structure": "encouraging tone, focus on benefits and progress",
                "length": "balanced detail with motivational elements",
                "examples": "success-oriented examples",
            },
            "professional": {
                "structure": "evidence-based, technical accuracy, formal tone",
                "length": "comprehensive with scientific backing",
                "examples": "research-based examples",
            },
        }

    def generate_dynamic_prompt(
        self,
        query: str,
        user_profile: UserProfile,
        retrieved_context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate a dynamic, personalized prompt"""

        try:
            # Determine age group
            age_group = self._get_age_group(user_profile.age)
            age_template = self.age_templates[age_group]

            # Get cultural context
            cultural_template = self._get_cultural_template(user_profile.culture)

            # Get dietary preferences
            dietary_template = self._get_dietary_template(
                user_profile.dietary_preferences
            )

            # Get health considerations
            health_templates = self._get_health_templates(
                user_profile.health_conditions or []
            )

            # Get response style
            style_template = self._get_style_template(user_profile.response_style)

            # Build the dynamic prompt
            prompt = self._build_comprehensive_prompt(
                query=query,
                user_profile=user_profile,
                retrieved_context=retrieved_context,
                age_template=age_template,
                cultural_template=cultural_template,
                dietary_template=dietary_template,
                health_templates=health_templates,
                style_template=style_template,
                conversation_history=conversation_history,
            )

            return prompt

        except Exception as e:
            logger.error(f"Error generating dynamic prompt: {e}")
            return self._generate_fallback_prompt(query, retrieved_context)

    def _get_age_group(self, age: Optional[int]) -> AgeGroup:
        """Determine age group from age"""
        if not age:
            return AgeGroup.ADULTS

        if age < 18:
            return AgeGroup.CHILDREN
        elif age <= 35:
            return AgeGroup.YOUNG_ADULTS
        elif age <= 64:
            return AgeGroup.ADULTS
        else:
            return AgeGroup.ELDERLY

    def _get_cultural_template(self, culture: Optional[str]) -> Dict[str, Any]:
        """Get cultural template based on user's culture"""
        if not culture:
            return self.cultural_templates.get("western", {})

        culture_key = culture.lower().replace(" ", "_")
        return self.cultural_templates.get(
            culture_key, self.cultural_templates.get("western", {})
        )

    def _get_dietary_template(
        self, dietary_preference: Optional[str]
    ) -> Dict[str, Any]:
        """Get dietary template based on user's preferences"""
        if not dietary_preference:
            return {}

        return self.dietary_templates.get(dietary_preference.lower(), {})

    def _get_health_templates(
        self, health_conditions: List[str]
    ) -> List[Dict[str, Any]]:
        """Get health templates for user's conditions"""
        templates = []
        for condition in health_conditions:
            condition_key = condition.lower().replace(" ", "_")
            template = self.health_templates.get(condition_key)
            if template:
                templates.append(template)
        return templates

    def _get_style_template(self, response_style: Optional[str]) -> Dict[str, Any]:
        """Get style template based on user's preference"""
        if not response_style:
            return self.style_templates.get("balanced", {})

        return self.style_templates.get(response_style.lower(), {})

    def _build_comprehensive_prompt(
        self,
        query: str,
        user_profile: UserProfile,
        retrieved_context: str,
        age_template: Dict[str, Any],
        cultural_template: Dict[str, Any],
        dietary_template: Dict[str, Any],
        health_templates: List[Dict[str, Any]],
        style_template: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> str:
        """Build comprehensive personalized prompt"""

        # System prompt based on age
        system_prompt = age_template.get(
            "system_prompt", "You are a helpful nutritional assistant."
        )

        # User context section
        user_context = self._build_user_context_section(
            user_profile, cultural_template, dietary_template
        )

        # Health considerations
        health_considerations = self._build_health_considerations_section(
            health_templates
        )

        # Cultural and dietary guidelines
        cultural_guidelines = self._build_cultural_guidelines_section(
            cultural_template, dietary_template
        )

        # Response guidelines
        response_guidelines = self._build_response_guidelines_section(
            age_template, style_template
        )

        # Safety and disclaimer instructions
        safety_instructions = self._build_safety_instructions_section(
            age_template, health_templates
        )

        # Context information
        context_section = (
            f"Relevant Context Information:\n{retrieved_context}\n"
            if retrieved_context
            else ""
        )

        # Conversation history
        history_section = self._build_conversation_history_section(conversation_history)

        # Final prompt assembly
        prompt = f"""{system_prompt}

{user_context}

{health_considerations}

{cultural_guidelines}

{response_guidelines}

{safety_instructions}

{context_section}{history_section}

User Query: {query}

Please provide a helpful, accurate, and personalized response that follows all the above guidelines while being culturally appropriate and age-suitable."""

        return prompt.strip()

    def _build_user_context_section(
        self,
        user_profile: UserProfile,
        cultural_template: Dict[str, Any],
        dietary_template: Dict[str, Any],
    ) -> str:
        """Build user context section of prompt"""

        context_parts = ["User Profile Context:"]

        if user_profile.age:
            age_group = self._get_age_group(user_profile.age).value
            context_parts.append(f"- Age Group: {age_group} (age {user_profile.age})")

        if user_profile.culture:
            context_parts.append(f"- Cultural Background: {user_profile.culture}")
            if cultural_template:
                context_parts.append(
                    f"- Food Context: {cultural_template.get('food_context', '')}"
                )

        if user_profile.dietary_preferences:
            context_parts.append(
                f"- Dietary Preference: {user_profile.dietary_preferences}"
            )
            if dietary_template:
                context_parts.append(
                    f"- Focus Areas: {dietary_template.get('focus', '')}"
                )

        if user_profile.allergies:
            context_parts.append(f"- Allergies: {', '.join(user_profile.allergies)}")

        if user_profile.activity_level:
            context_parts.append(f"- Activity Level: {user_profile.activity_level}")

        return "\n".join(context_parts)

    def _build_health_considerations_section(
        self, health_templates: List[Dict[str, Any]]
    ) -> str:
        """Build health considerations section"""
        if not health_templates:
            return ""

        considerations = ["Health Considerations:"]

        for template in health_templates:
            if template.get("focus"):
                considerations.append(f"- Focus: {template['focus']}")
            if template.get("restrictions"):
                considerations.append(f"- Restrictions: {template['restrictions']}")
            if template.get("recommendations"):
                considerations.append(
                    f"- Recommendations: {template['recommendations']}"
                )
            if template.get("disclaimer"):
                considerations.append(f"- Important: {template['disclaimer']}")

        return "\n".join(considerations)

    def _build_cultural_guidelines_section(
        self, cultural_template: Dict[str, Any], dietary_template: Dict[str, Any]
    ) -> str:
        """Build cultural and dietary guidelines section"""
        guidelines = []

        if cultural_template:
            guidelines.append("Cultural Guidelines:")
            if cultural_template.get("examples"):
                guidelines.append(
                    f"- Use familiar foods: {cultural_template['examples']}"
                )
            if cultural_template.get("cultural_considerations"):
                guidelines.append(
                    f"- Consider: {cultural_template['cultural_considerations']}"
                )

        if dietary_template:
            if guidelines:
                guidelines.append("")
            guidelines.append("Dietary Guidelines:")
            if dietary_template.get("alternatives"):
                guidelines.append(f"- Alternatives: {dietary_template['alternatives']}")
            if dietary_template.get("examples"):
                guidelines.append(f"- Examples: {dietary_template['examples']}")

        return "\n".join(guidelines) if guidelines else ""

    def _build_response_guidelines_section(
        self, age_template: Dict[str, Any], style_template: Dict[str, Any]
    ) -> str:
        """Build response guidelines section"""
        guidelines = ["Response Guidelines:"]

        guidelines.append(
            f"- Tone: {age_template.get('tone', 'helpful and informative')}"
        )
        guidelines.append(
            f"- Vocabulary: {age_template.get('vocabulary', 'appropriate for audience')}"
        )
        guidelines.append(
            f"- Structure: {age_template.get('structure', 'clear and organized')}"
        )
        guidelines.append(
            f"- Length: {age_template.get('response_length', 'appropriate for query')}"
        )

        if style_template.get("structure"):
            guidelines.append(f"- Style: {style_template['structure']}")

        return "\n".join(guidelines)

    def _build_safety_instructions_section(
        self, age_template: Dict[str, Any], health_templates: List[Dict[str, Any]]
    ) -> str:
        """Build safety and disclaimer instructions"""
        instructions = ["Safety Instructions:"]

        safety_level = age_template.get("safety_level", "standard")
        if safety_level == "high":
            instructions.append("- Use extra caution with recommendations")
            instructions.append("- Always emphasize consulting healthcare providers")

        if age_template.get("disclaimers"):
            instructions.append(f"- Include disclaimer: {age_template['disclaimers']}")

        if health_templates:
            instructions.append(
                "- Emphasize coordination with healthcare team for medical conditions"
            )
            instructions.append("- Avoid contradicting medical advice")

        instructions.append("- Do not provide specific medical diagnoses or treatments")
        instructions.append("- Focus on general nutritional education")

        return "\n".join(instructions)

    def _build_conversation_history_section(
        self, conversation_history: Optional[List[Dict[str, str]]]
    ) -> str:
        """Build conversation history section"""
        if not conversation_history:
            return ""

        history_lines = ["Recent Conversation History:"]

        # Include last few exchanges for context
        for exchange in conversation_history[-3:]:  # Last 3 exchanges
            if exchange.get("user"):
                history_lines.append(f"User: {exchange['user']}")
            if exchange.get("assistant"):
                history_lines.append(f"Assistant: {exchange['assistant']}")

        history_lines.append("")  # Empty line before query

        return "\n".join(history_lines)

    def _generate_fallback_prompt(self, query: str, retrieved_context: str) -> str:
        """Generate simple fallback prompt"""
        return f"""You are a helpful nutritional assistant. Provide accurate, safe nutritional information.

Context Information:
{retrieved_context}

User Query: {query}

Please provide a helpful response while recommending consultation with healthcare professionals for specific medical advice."""

    def get_prompt_templates_info(self) -> Dict[str, Any]:
        """Get information about available prompt templates"""
        return {
            "age_groups": list(self.age_templates.keys()),
            "cultures": list(self.cultural_templates.keys()),
            "dietary_preferences": list(self.dietary_templates.keys()),
            "health_conditions": list(self.health_templates.keys()),
            "response_styles": list(self.style_templates.keys()),
        }


# Global prompt service instance
prompt_service = None


def get_prompt_service() -> DynamicPromptService:
    """Get or create dynamic prompt service instance"""
    global prompt_service
    if prompt_service is None:
        prompt_service = DynamicPromptService()
    return prompt_service


async def initialize_prompt_service() -> DynamicPromptService:
    """Initialize dynamic prompt service on startup"""
    global prompt_service
    try:
        prompt_service = DynamicPromptService()
        logger.info("Dynamic prompt service initialized")
        return prompt_service
    except Exception as e:
        logger.error(f"Failed to initialize prompt service: {e}")
        # Return service anyway
        prompt_service = DynamicPromptService()
        return prompt_service
