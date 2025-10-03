"""
LLM Integration Service for AI response generation with multiple providers
"""

import asyncio
import openai
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import logging
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum
import re
import time

# Import transformers when available
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""

    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    HUGGINGFACE_LOCAL = "huggingface_local"
    FALLBACK = "fallback"


class ResponseStyle(Enum):
    """Response styles for different contexts"""

    SIMPLE = "simple"
    DETAILED = "detailed"
    MOTIVATIONAL = "motivational"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"


@dataclass
class LLMRequest:
    """Request structure for LLM generation"""

    prompt: str
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = None
    stream: bool = False


@dataclass
class LLMResponse:
    """Response structure from LLM"""

    content: str
    provider: str
    model: str
    tokens_used: int
    response_time_ms: float
    confidence_score: Optional[float] = None
    safety_passed: bool = True
    metadata: Dict[str, Any] = None


class LLMIntegrationService:
    """Service for integrating with various LLM providers"""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.providers = {}
        self.default_provider = LLMProvider.FALLBACK
        self.local_models = {}

        # Safety filters and content moderation
        self.safety_keywords = {
            "harmful": [
                "poison",
                "toxic",
                "dangerous",
                "overdose",
                "excessive consumption",
                "raw meat",
                "unpasteurized",
                "contaminated",
            ],
            "medical_advice": [
                "diagnose",
                "treatment",
                "cure",
                "medication",
                "prescription",
                "medical condition",
                "disease",
                "illness",
            ],
        }

        # Response templates for different scenarios
        self.safety_responses = {
            "harmful_content": "I can't provide advice on potentially harmful nutritional practices. Please consult a healthcare professional for specific medical guidance.",
            "medical_advice": "I can provide general nutritional information, but for specific medical conditions or treatments, please consult with a qualified healthcare provider.",
            "insufficient_info": "I don't have enough reliable information to provide a complete answer to your question. Let me share what I do know and suggest you consult additional sources.",
        }

        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available LLM providers"""

        # Initialize OpenAI
        if self.openai_api_key:
            try:
                openai.api_key = self.openai_api_key
                self.providers[LLMProvider.OPENAI_GPT4] = {
                    "model": "gpt-4",
                    "available": True,
                }
                self.providers[LLMProvider.OPENAI_GPT35] = {
                    "model": "gpt-3.5-turbo",
                    "available": True,
                }
                self.default_provider = LLMProvider.OPENAI_GPT4
                logger.info("OpenAI providers initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")

        # Initialize HuggingFace local models
        if TRANSFORMERS_AVAILABLE:
            try:
                self._initialize_local_models()
            except Exception as e:
                logger.error(f"Failed to initialize local models: {e}")

        # Always have fallback available
        self.providers[LLMProvider.FALLBACK] = {
            "model": "rule_based",
            "available": True,
        }

        logger.info(f"LLM providers initialized: {list(self.providers.keys())}")

    def _initialize_local_models(self):
        """Initialize local HuggingFace models"""
        try:
            # Use a small, efficient model for local inference
            model_name = "microsoft/DialoGPT-small"

            # Initialize tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.local_models["dialogpt"] = {
                "tokenizer": tokenizer,
                "model": model,
                "pipeline": pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                ),
            }

            self.providers[LLMProvider.HUGGINGFACE_LOCAL] = {
                "model": model_name,
                "available": True,
            }

            logger.info("Local HuggingFace models initialized")

        except Exception as e:
            logger.error(f"Failed to initialize local models: {e}")

    async def generate_response(
        self,
        request: LLMRequest,
        provider: Optional[LLMProvider] = None,
        fallback_on_error: bool = True,
    ) -> LLMResponse:
        """Generate response using specified or default provider"""

        start_time = time.time()

        # Use default provider if none specified
        if provider is None:
            provider = self.default_provider

        # Check if provider is available
        if provider not in self.providers or not self.providers[provider]["available"]:
            if fallback_on_error:
                provider = LLMProvider.FALLBACK
            else:
                raise ValueError(f"Provider {provider} not available")

        try:
            # Apply safety filtering to prompt
            safety_check = self._check_prompt_safety(request.prompt)
            if not safety_check["safe"]:
                return LLMResponse(
                    content=self.safety_responses[safety_check["reason"]],
                    provider=provider.value,
                    model=self.providers[provider]["model"],
                    tokens_used=0,
                    response_time_ms=(time.time() - start_time) * 1000,
                    safety_passed=False,
                    metadata={"safety_reason": safety_check["reason"]},
                )

            # Generate response based on provider
            if provider in [LLMProvider.OPENAI_GPT4, LLMProvider.OPENAI_GPT35]:
                response = await self._generate_openai_response(request, provider)
            elif provider == LLMProvider.HUGGINGFACE_LOCAL:
                response = await self._generate_local_response(request)
            else:  # FALLBACK
                response = await self._generate_fallback_response(request)

            # Apply post-generation safety check
            content_safety = self._check_content_safety(response.content)
            response.safety_passed = content_safety["safe"]

            if not content_safety["safe"]:
                response.content = self.safety_responses.get(
                    content_safety["reason"],
                    "I apologize, but I cannot provide that information for safety reasons.",
                )

            # Calculate response time
            response.response_time_ms = (time.time() - start_time) * 1000

            return response

        except Exception as e:
            logger.error(f"Error generating response with {provider}: {e}")

            if fallback_on_error and provider != LLMProvider.FALLBACK:
                return await self.generate_response(
                    request, LLMProvider.FALLBACK, fallback_on_error=False
                )
            else:
                return LLMResponse(
                    content="I apologize, but I'm experiencing technical difficulties. Please try again later.",
                    provider=provider.value,
                    model="error",
                    tokens_used=0,
                    response_time_ms=(time.time() - start_time) * 1000,
                    safety_passed=True,
                    metadata={"error": str(e)},
                )

    async def _generate_openai_response(
        self, request: LLMRequest, provider: LLMProvider
    ) -> LLMResponse:
        """Generate response using OpenAI API"""

        model_name = self.providers[provider]["model"]

        try:
            # Prepare messages for chat completion
            messages = [
                {
                    "role": "system",
                    "content": "You are a knowledgeable nutritional AI assistant. Provide helpful, accurate, and safe nutritional advice while being clear about the limitations of your knowledge.",
                },
                {"role": "user", "content": request.prompt},
            ]

            # Make API call
            response = await openai.ChatCompletion.acreate(
                model=model_name,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop_sequences,
            )

            # Extract response content
            content = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens

            return LLMResponse(
                content=content,
                provider=provider.value,
                model=model_name,
                tokens_used=tokens_used,
                response_time_ms=0,  # Will be set by caller
                confidence_score=self._calculate_confidence_score(content),
                safety_passed=True,
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise e

    async def _generate_local_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using local HuggingFace model"""

        if "dialogpt" not in self.local_models:
            raise ValueError("Local model not available")

        try:
            model_info = self.local_models["dialogpt"]
            generator = model_info["pipeline"]

            # Generate response
            outputs = generator(
                request.prompt,
                max_length=len(request.prompt) + request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=model_info["tokenizer"].eos_token_id,
            )

            # Extract generated text (remove prompt)
            generated_text = outputs[0]["generated_text"]
            response_content = generated_text[len(request.prompt) :].strip()

            # Estimate token usage
            tokens_used = len(model_info["tokenizer"].encode(response_content))

            return LLMResponse(
                content=response_content,
                provider=LLMProvider.HUGGINGFACE_LOCAL.value,
                model=self.providers[LLMProvider.HUGGINGFACE_LOCAL]["model"],
                tokens_used=tokens_used,
                response_time_ms=0,
                confidence_score=self._calculate_confidence_score(response_content),
                safety_passed=True,
            )

        except Exception as e:
            logger.error(f"Local model error: {e}")
            raise e

    async def _generate_fallback_response(self, request: LLMRequest) -> LLMResponse:
        """Generate fallback response using rule-based system"""

        prompt_lower = request.prompt.lower()

        # Nutritional knowledge base responses
        if any(word in prompt_lower for word in ["protein", "amino acid"]):
            content = "Protein is essential for building and repairing tissues. Good sources include lean meats, fish, eggs, dairy, legumes, and nuts. Adults typically need 0.8g per kg of body weight daily."

        elif any(word in prompt_lower for word in ["vitamin c", "ascorbic acid"]):
            content = "Vitamin C is important for immune function and collagen synthesis. Rich sources include citrus fruits, berries, bell peppers, and leafy greens. Daily requirement is about 75-90mg for adults."

        elif any(word in prompt_lower for word in ["calcium", "bone health"]):
            content = "Calcium is crucial for bone and teeth health. Dairy products, leafy greens, sardines, and fortified foods are excellent sources. Adults need about 1000-1200mg daily."

        elif any(word in prompt_lower for word in ["fiber", "digestive health"]):
            content = "Dietary fiber supports digestive health and may help lower cholesterol. Whole grains, fruits, vegetables, and legumes are high in fiber. Aim for 25-35g daily."

        elif any(word in prompt_lower for word in ["iron", "anemia"]):
            content = "Iron is essential for oxygen transport in blood. Red meat, poultry, fish, beans, and fortified cereals are good sources. Vitamin C enhances iron absorption."

        elif any(word in prompt_lower for word in ["weight loss", "lose weight"]):
            content = "Healthy weight loss involves creating a moderate calorie deficit through balanced nutrition and regular physical activity. Focus on whole foods, portion control, and sustainable habits."

        elif any(word in prompt_lower for word in ["diabetes", "blood sugar"]):
            content = "For blood sugar management, focus on complex carbohydrates, fiber-rich foods, lean proteins, and healthy fats. Regular meal timing and portion control are important. Consult healthcare providers for personalized advice."

        elif any(word in prompt_lower for word in ["hydration", "water intake"]):
            content = "Proper hydration is essential for health. Aim for about 8 glasses (64oz) of water daily, more if active or in hot weather. Water, herbal teas, and water-rich foods contribute to hydration."

        else:
            content = "I'd be happy to help with your nutritional question. For specific dietary advice, especially related to health conditions, I recommend consulting with a registered dietitian or healthcare provider who can provide personalized guidance."

        # Add disclaimer
        content += "\n\nNote: This is general information and not personalized medical advice. Consult healthcare professionals for specific dietary needs."

        return LLMResponse(
            content=content,
            provider=LLMProvider.FALLBACK.value,
            model="rule_based",
            tokens_used=len(content.split()),
            response_time_ms=0,
            confidence_score=0.7,
            safety_passed=True,
            metadata={"fallback_reason": "rule_based_match"},
        )

    def _check_prompt_safety(self, prompt: str) -> Dict[str, Any]:
        """Check if prompt is safe to process"""
        prompt_lower = prompt.lower()

        # Check for harmful content requests
        for keyword in self.safety_keywords["harmful"]:
            if keyword in prompt_lower:
                return {"safe": False, "reason": "harmful_content"}

        # Check for medical advice requests
        medical_indicators = ["diagnose", "treat", "cure", "medication", "prescription"]
        if any(indicator in prompt_lower for indicator in medical_indicators):
            return {"safe": False, "reason": "medical_advice"}

        return {"safe": True, "reason": None}

    def _check_content_safety(self, content: str) -> Dict[str, Any]:
        """Check if generated content is safe"""
        content_lower = content.lower()

        # Check for potentially harmful advice
        dangerous_phrases = [
            "take large amounts",
            "consume excessive",
            "ignore medical advice",
            "stop taking medication",
            "raw meat is safe",
            "unpasteurized is fine",
        ]

        for phrase in dangerous_phrases:
            if phrase in content_lower:
                return {"safe": False, "reason": "harmful_content"}

        return {"safe": True, "reason": None}

    def _calculate_confidence_score(self, content: str) -> float:
        """Calculate confidence score based on content characteristics"""

        # Base confidence
        confidence = 0.7

        # Boost for specific nutritional facts
        if any(
            word in content.lower()
            for word in ["studies show", "research indicates", "according to"]
        ):
            confidence += 0.1

        # Boost for including disclaimers
        if any(
            phrase in content.lower()
            for phrase in ["consult", "healthcare", "professional advice"]
        ):
            confidence += 0.1

        # Reduce for vague language
        if any(
            word in content.lower()
            for word in ["maybe", "possibly", "might", "could be"]
        ):
            confidence -= 0.1

        # Ensure confidence is within bounds
        return max(0.1, min(0.95, confidence))

    async def generate_stream_response(
        self, request: LLMRequest, provider: Optional[LLMProvider] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response (if supported by provider)"""

        # For now, simulate streaming by yielding chunks
        response = await self.generate_response(request, provider)

        # Split content into chunks for streaming effect
        words = response.content.split()
        chunk_size = 5  # Words per chunk

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if i + chunk_size < len(words):
                chunk += " "

            yield chunk
            await asyncio.sleep(0.1)  # Small delay for streaming effect

    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        return [
            provider.value
            for provider, info in self.providers.items()
            if info["available"]
        ]

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about LLM providers"""
        return {
            "available_providers": self.get_available_providers(),
            "default_provider": self.default_provider.value,
            "openai_configured": self.openai_api_key is not None,
            "local_models_available": TRANSFORMERS_AVAILABLE,
            "local_models_loaded": list(self.local_models.keys()),
        }


# Global LLM service instance
llm_service = None


def get_llm_service() -> LLMIntegrationService:
    """Get or create LLM integration service instance"""
    global llm_service
    if llm_service is None:
        llm_service = LLMIntegrationService()
    return llm_service


async def initialize_llm_service(
    openai_api_key: Optional[str] = None,
) -> LLMIntegrationService:
    """Initialize LLM integration service on startup"""
    global llm_service
    try:
        llm_service = LLMIntegrationService(openai_api_key)
        logger.info("LLM integration service initialized")
        return llm_service
    except Exception as e:
        logger.error(f"Failed to initialize LLM service: {e}")
        # Return service anyway for fallback functionality
        llm_service = LLMIntegrationService()
        return llm_service
