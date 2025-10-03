"""
Enhanced Chat API endpoints with hybrid retrieval and LLM integration
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import uuid
import time
import asyncio
from datetime import datetime

from database.database import get_db
from database.models import UserProfile, ChatSession, ChatMessage
from schemas import MessageCreate, MessageResponse, ChatSessionResponse, ChatResponse

# Import advanced services
from services.hybrid_retrieval_service import (
    get_hybrid_retrieval_service,
    UserContext,
    RetrievalMode,
)
from services.llm_service import get_llm_service, LLMRequest, LLMProvider
from services.prompt_service import get_prompt_service, UserProfile as PromptUserProfile
from services.vector_service import get_vector_service
from services.graph_service import get_graph_service

router = APIRouter()


@router.post("/message", response_model=ChatResponse)
async def send_enhanced_message(
    message_data: MessageCreate, request: Request, db: Session = Depends(get_db)
):
    """Send a message to the enhanced chatbot with hybrid retrieval and LLM integration"""
    start_time = time.time()

    try:
        # Get or create user profile
        profile = None
        if message_data.profile_uuid:
            profile = (
                db.query(UserProfile)
                .filter(UserProfile.profile_uuid == message_data.profile_uuid)
                .first()
            )

        # Get or create active chat session
        session = await get_or_create_session(db, profile, message_data.content)

        # Save user message
        user_message = ChatMessage(
            session_id=session.id, message_type="user", content=message_data.content
        )
        db.add(user_message)
        db.flush()

        # Get conversation history for context
        conversation_history = await get_conversation_history(db, session.id, limit=5)

        # Generate enhanced AI response
        ai_response_content, confidence_score, metadata = (
            await generate_enhanced_ai_response(
                query=message_data.content,
                profile=profile,
                conversation_history=conversation_history,
                request=request,
            )
        )

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Save AI message
        ai_message = ChatMessage(
            session_id=session.id,
            message_type="assistant",
            content=ai_response_content,
            response_time_ms=response_time_ms,
            confidence_score=confidence_score,
            metadata=metadata,
        )
        db.add(ai_message)

        # Generate intelligent suggestions
        suggestions = await generate_intelligent_suggestions(
            message_data.content, ai_response_content, profile
        )

        db.commit()
        db.refresh(user_message)
        db.refresh(ai_message)

        return ChatResponse(
            session_uuid=session.session_uuid,
            user_message=user_message,
            assistant_message=ai_message,
            suggestions=suggestions,
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}",
        )


async def get_or_create_session(
    db: Session, profile: Optional[UserProfile], content: str
) -> ChatSession:
    """Get existing session or create new one"""
    session = None
    if profile:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.profile_id == profile.id, ChatSession.is_active == True)
            .first()
        )

    if not session:
        # Create new session
        session = ChatSession(
            session_uuid=str(uuid.uuid4()),
            profile_id=profile.id if profile else None,
            title=(content[:50] + "..." if len(content) > 50 else content),
            is_active=True,
        )
        db.add(session)
        db.flush()

    return session


async def get_conversation_history(
    db: Session, session_id: int, limit: int = 5
) -> List[Dict[str, str]]:
    """Get recent conversation history for context"""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp.desc())
        .limit(limit * 2)  # Get more to account for user/assistant pairs
        .all()
    )

    # Convert to conversation format
    history = []
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            # Assume messages come in user/assistant pairs
            user_msg = (
                messages[i] if messages[i].message_type == "user" else messages[i + 1]
            )
            assistant_msg = (
                messages[i + 1]
                if messages[i + 1].message_type == "assistant"
                else messages[i]
            )

            history.append(
                {"user": user_msg.content, "assistant": assistant_msg.content}
            )

    return list(reversed(history))  # Return in chronological order


async def generate_enhanced_ai_response(
    query: str,
    profile: Optional[UserProfile],
    conversation_history: List[Dict[str, str]],
    request: Request,
) -> tuple[str, float, Dict[str, Any]]:
    """Generate enhanced AI response using hybrid retrieval and LLM"""

    metadata = {
        "retrieval_results": 0,
        "llm_provider": "fallback",
        "processing_steps": [],
    }

    try:
        # Step 1: Build user context for retrieval
        user_context = UserContext(
            age=profile.age if profile else None,
            culture=profile.culture if profile else None,
            dietary_preferences=profile.dietary_preferences if profile else None,
            allergies=(
                [allergy.allergy for allergy in profile.allergies]
                if profile and profile.allergies
                else []
            ),
            health_conditions=(
                [condition.condition_name for condition in profile.health_conditions]
                if profile and profile.health_conditions
                else []
            ),
            activity_level=profile.activity_level if profile else None,
            response_style=profile.response_style if profile else None,
        )
        metadata["processing_steps"].append("user_context_built")

        # Step 2: Hybrid retrieval for relevant context
        hybrid_service = get_hybrid_retrieval_service()
        retrieval_results = await hybrid_service.hybrid_retrieve(
            query=query, user_context=user_context, k=5, mode=RetrievalMode.ADAPTIVE
        )

        retrieved_context = "\n".join(
            [f"- {result.content}" for result in retrieval_results]
        )
        metadata["retrieval_results"] = len(retrieval_results)
        metadata["processing_steps"].append("hybrid_retrieval_completed")

        # Step 3: Build user profile for prompt service
        prompt_user_profile = PromptUserProfile(
            age=profile.age if profile else None,
            culture=profile.culture if profile else None,
            dietary_preferences=profile.dietary_preferences if profile else None,
            allergies=(
                [allergy.allergy for allergy in profile.allergies]
                if profile and profile.allergies
                else []
            ),
            health_conditions=(
                [condition.condition_name for condition in profile.health_conditions]
                if profile and profile.health_conditions
                else []
            ),
            activity_level=profile.activity_level if profile else None,
            response_style=profile.response_style if profile else None,
        )

        # Step 4: Generate dynamic prompt
        prompt_service = get_prompt_service()
        dynamic_prompt = prompt_service.generate_dynamic_prompt(
            query=query,
            user_profile=prompt_user_profile,
            retrieved_context=retrieved_context,
            conversation_history=conversation_history,
        )
        metadata["processing_steps"].append("dynamic_prompt_generated")

        # Step 5: Generate LLM response
        llm_service = get_llm_service()
        llm_request = LLMRequest(prompt=dynamic_prompt, max_tokens=500, temperature=0.7)

        # Try to use the best available provider
        available_providers = llm_service.get_available_providers()
        provider = None
        if "openai_gpt4" in available_providers:
            provider = LLMProvider.OPENAI_GPT4
        elif "openai_gpt35" in available_providers:
            provider = LLMProvider.OPENAI_GPT35
        elif "huggingface_local" in available_providers:
            provider = LLMProvider.HUGGINGFACE_LOCAL

        llm_response = await llm_service.generate_response(llm_request, provider)
        metadata["llm_provider"] = llm_response.provider
        metadata["llm_tokens"] = llm_response.tokens_used
        metadata["llm_response_time"] = llm_response.response_time_ms
        metadata["processing_steps"].append("llm_response_generated")

        return llm_response.content, llm_response.confidence_score or 0.8, metadata

    except Exception as e:
        # Fallback to basic response
        metadata["processing_steps"].append(f"error_fallback: {str(e)}")
        fallback_response = await generate_fallback_response(query, profile)
        return fallback_response, 0.6, metadata


async def generate_fallback_response(query: str, profile: Optional[UserProfile]) -> str:
    """Generate fallback response when advanced services fail"""
    query_lower = query.lower()

    # Basic responses based on query content
    if any(word in query_lower for word in ["protein", "amino acid"]):
        response = "Protein is essential for building and repairing tissues. Good sources include lean meats, fish, eggs, dairy, legumes, and nuts."
    elif any(word in query_lower for word in ["vitamin", "nutrient"]):
        response = "Vitamins and minerals are important for various body functions. A balanced diet with fruits, vegetables, whole grains, and lean proteins helps ensure adequate nutrition."
    elif any(word in query_lower for word in ["weight loss", "lose weight"]):
        response = "Healthy weight management involves balanced nutrition and regular physical activity. Focus on whole foods, portion control, and sustainable habits."
    elif any(word in query_lower for word in ["diabetes", "blood sugar"]):
        response = "For blood sugar management, focus on complex carbohydrates, fiber-rich foods, and regular meal timing. Always consult healthcare providers for personalized advice."
    else:
        response = "I'd be happy to help with your nutritional question. For personalized dietary advice, especially for specific health conditions, I recommend consulting with a registered dietitian."

    # Add profile-specific adjustments
    if profile:
        if profile.dietary_preferences == "vegetarian":
            response += " For vegetarian options, consider plant-based proteins like legumes, nuts, and seeds."
        elif profile.dietary_preferences == "vegan":
            response += " For vegan options, focus on diverse plant-based foods and consider B12 supplementation."

    response += "\n\nNote: This is general information. Please consult healthcare professionals for specific dietary needs."

    return response


async def generate_intelligent_suggestions(
    user_query: str, ai_response: str, profile: Optional[UserProfile]
) -> List[str]:
    """Generate intelligent follow-up suggestions"""

    try:
        # Get NLP service for entity extraction
        from services.nlp_service import get_nlp_service

        nlp_service = get_nlp_service()

        # Extract entities from query and response
        query_entities = (
            await nlp_service.extract_entities(user_query) if nlp_service else []
        )

        suggestions = []
        query_lower = user_query.lower()

        # Topic-based suggestions
        if any(word in query_lower for word in ["protein", "amino acid"]):
            suggestions.extend(
                [
                    "What are the best plant-based protein sources?",
                    "How much protein do I need daily?",
                    "Can you explain complete vs. incomplete proteins?",
                ]
            )
        elif any(word in query_lower for word in ["vitamin", "mineral"]):
            suggestions.extend(
                [
                    "Which foods are rich in vitamin D?",
                    "How can I improve iron absorption?",
                    "What are signs of vitamin deficiency?",
                ]
            )
        elif any(word in query_lower for word in ["weight", "calories"]):
            suggestions.extend(
                [
                    "How do I calculate my daily calorie needs?",
                    "What are healthy weight loss strategies?",
                    "Can you suggest low-calorie meal ideas?",
                ]
            )
        elif any(word in query_lower for word in ["diabetes", "blood sugar"]):
            suggestions.extend(
                [
                    "What is the glycemic index?",
                    "Which foods help stabilize blood sugar?",
                    "How does timing of meals affect diabetes?",
                ]
            )

        # Profile-specific suggestions
        if profile:
            if profile.dietary_preferences == "vegetarian":
                suggestions.append(
                    "What nutrients should vegetarians pay attention to?"
                )
            elif profile.dietary_preferences == "vegan":
                suggestions.append("Which supplements are important for vegans?")

            if profile.health_conditions:
                for condition in profile.health_conditions:
                    if condition.condition_name.lower() in [
                        "diabetes",
                        "hypertension",
                        "heart disease",
                    ]:
                        suggestions.append(
                            f"What foods are best for {condition.condition_name}?"
                        )

        # Entity-based suggestions
        for entity in query_entities:
            if entity.get("label") == "FOOD_ITEM":
                suggestions.append(
                    f"What are the nutritional benefits of {entity['text']}?"
                )
            elif entity.get("label") == "NUTRIENT":
                suggestions.append(f"Which foods are high in {entity['text']}?")

        # Generic helpful suggestions
        base_suggestions = [
            "How can I plan balanced meals?",
            "What are some healthy snack ideas?",
            "How do I read nutrition labels effectively?",
        ]

        # Combine and limit suggestions
        all_suggestions = suggestions + base_suggestions
        unique_suggestions = list(dict.fromkeys(all_suggestions))  # Remove duplicates

        return unique_suggestions[:5]  # Return top 5 suggestions

    except Exception as e:
        # Fallback suggestions
        return [
            "How can I improve my diet?",
            "What are some healthy meal ideas?",
            "Can you help me understand nutrition labels?",
            "What nutrients are most important?",
            "How do I maintain a balanced diet?",
        ]


@router.get("/sessions/{profile_uuid}", response_model=List[ChatSessionResponse])
async def get_chat_sessions(profile_uuid: str, db: Session = Depends(get_db)):
    """Get chat sessions for a user profile"""
    profile = (
        db.query(UserProfile).filter(UserProfile.profile_uuid == profile_uuid).first()
    )

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found"
        )

    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.profile_id == profile.id)
        .order_by(ChatSession.started_at.desc())
        .all()
    )

    return sessions


@router.get("/sessions/{session_uuid}/messages", response_model=List[MessageResponse])
async def get_chat_history(session_uuid: str, db: Session = Depends(get_db)):
    """Get chat history for a session"""
    session = (
        db.query(ChatSession).filter(ChatSession.session_uuid == session_uuid).first()
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found"
        )

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )

    return messages


@router.post("/sessions/{session_uuid}/end")
async def end_chat_session(session_uuid: str, db: Session = Depends(get_db)):
    """End a chat session"""
    session = (
        db.query(ChatSession).filter(ChatSession.session_uuid == session_uuid).first()
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found"
        )

    session.is_active = False
    session.ended_at = datetime.utcnow()

    db.commit()

    return {"message": "Chat session ended successfully"}


@router.post("/suggestions", response_model=List[str])
async def get_chat_suggestions(
    query: str, profile_uuid: Optional[str] = None, db: Session = Depends(get_db)
):
    """Get intelligent chat suggestions based on query"""

    profile = None
    if profile_uuid:
        profile = (
            db.query(UserProfile)
            .filter(UserProfile.profile_uuid == profile_uuid)
            .first()
        )

    suggestions = await generate_intelligent_suggestions(query, "", profile)
    return suggestions


@router.get("/analytics/{session_uuid}")
async def get_session_analytics(session_uuid: str, db: Session = Depends(get_db)):
    """Get analytics for a chat session"""
    session = (
        db.query(ChatSession).filter(ChatSession.session_uuid == session_uuid).first()
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found"
        )

    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).all()

    # Calculate analytics
    total_messages = len(messages)
    user_messages = [m for m in messages if m.message_type == "user"]
    assistant_messages = [m for m in messages if m.message_type == "assistant"]

    avg_response_time = sum(
        m.response_time_ms for m in assistant_messages if m.response_time_ms
    ) / max(1, len([m for m in assistant_messages if m.response_time_ms]))

    avg_confidence = sum(
        m.confidence_score for m in assistant_messages if m.confidence_score
    ) / max(1, len([m for m in assistant_messages if m.confidence_score]))

    return {
        "session_uuid": session_uuid,
        "total_messages": total_messages,
        "user_messages": len(user_messages),
        "assistant_messages": len(assistant_messages),
        "session_duration_minutes": (
            (session.ended_at or datetime.utcnow()) - session.started_at
        ).total_seconds()
        / 60,
        "average_response_time_ms": avg_response_time,
        "average_confidence_score": avg_confidence,
        "session_active": session.is_active,
    }
