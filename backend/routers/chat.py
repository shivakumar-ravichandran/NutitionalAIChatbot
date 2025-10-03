"""
Chat API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import time
from datetime import datetime

from database.database import get_db
from database.models import UserProfile, ChatSession, ChatMessage
from schemas import MessageCreate, MessageResponse, ChatSessionResponse, ChatResponse

router = APIRouter()


@router.post("/message", response_model=ChatResponse)
async def send_message(
    message_data: MessageCreate, request: Request, db: Session = Depends(get_db)
):
    """Send a message to the chatbot"""
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
        session = None
        if profile:
            session = (
                db.query(ChatSession)
                .filter(
                    ChatSession.profile_id == profile.id, ChatSession.is_active == True
                )
                .first()
            )

        if not session:
            # Create new session
            session = ChatSession(
                session_uuid=str(uuid.uuid4()),
                profile_id=profile.id if profile else None,
                title=(
                    message_data.content[:50] + "..."
                    if len(message_data.content) > 50
                    else message_data.content
                ),
                is_active=True,
            )
            db.add(session)
            db.flush()

        # Save user message
        user_message = ChatMessage(
            session_id=session.id, message_type="user", content=message_data.content
        )
        db.add(user_message)
        db.flush()

        # Generate AI response (placeholder for now)
        ai_response_content = await generate_ai_response(
            message_data.content,
            profile,
            (
                request.app.state.nlp_service
                if hasattr(request.app.state, "nlp_service")
                else None
            ),
        )

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Save AI message
        ai_message = ChatMessage(
            session_id=session.id,
            message_type="assistant",
            content=ai_response_content,
            response_time_ms=response_time_ms,
            confidence_score=0.85,  # Placeholder
        )
        db.add(ai_message)

        db.commit()
        db.refresh(user_message)
        db.refresh(ai_message)

        return ChatResponse(
            session_uuid=session.session_uuid,
            user_message=user_message,
            assistant_message=ai_message,
            suggestions=generate_suggestions(message_data.content),
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}",
        )


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


@router.post("/sessions", response_model=ChatSessionResponse)
async def create_chat_session(
    profile_uuid: Optional[str] = None, db: Session = Depends(get_db)
):
    """Create a new chat session"""
    try:
        profile = None
        if profile_uuid:
            profile = (
                db.query(UserProfile)
                .filter(UserProfile.profile_uuid == profile_uuid)
                .first()
            )

        session = ChatSession(
            session_uuid=str(uuid.uuid4()),
            profile_id=profile.id if profile else None,
            title="New Chat",
            is_active=True,
        )

        db.add(session)
        db.commit()
        db.refresh(session)

        return session

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating chat session: {str(e)}",
        )


@router.delete("/sessions/{session_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(session_uuid: str, db: Session = Depends(get_db)):
    """Delete a chat session"""
    session = (
        db.query(ChatSession).filter(ChatSession.session_uuid == session_uuid).first()
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found"
        )

    try:
        db.delete(session)
        db.commit()

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting chat session: {str(e)}",
        )


async def generate_ai_response(
    message: str, profile: Optional[UserProfile], nlp_service
) -> str:
    """Generate AI response based on user message and profile"""

    # Basic response generation (placeholder)
    if not message:
        return "I didn't receive your message. Could you please try again?"

    message_lower = message.lower()

    # Nutrition-related responses
    if any(word in message_lower for word in ["calorie", "calories", "kcal"]):
        if profile and profile.dietary_preferences == "vegetarian":
            return """Based on your vegetarian preference, here are some healthy, calorie-conscious options:

🥗 **High-protein, low-calorie foods:**
- Lentils (116 cal/100g, 9g protein)
- Greek yogurt (59 cal/100g, 10g protein)
- Quinoa (120 cal/100g, 4.4g protein)
- Tofu (76 cal/100g, 8g protein)

🍎 **Low-calorie snacks:**
- Apple (52 cal/100g)
- Cucumber (16 cal/100g)
- Berries (32-57 cal/100g)

Would you like specific meal suggestions or information about a particular food?"""
        else:
            return """Here's information about calories and healthy eating:

⚡ **Daily calorie needs vary by:**
- Age, gender, and activity level
- Generally 1,800-2,400 calories for adults

🥘 **Balanced meal components:**
- 45-65% carbohydrates
- 20-35% fats
- 10-35% protein

📊 **Calorie-dense vs nutrient-dense:**
- Focus on nutrient-dense foods (more vitamins/minerals per calorie)
- Examples: leafy greens, lean proteins, whole grains

What specific calorie information are you looking for?"""

    elif any(word in message_lower for word in ["protein", "proteins"]):
        return """🥩 **Protein Information:**

**Daily protein needs:**
- Sedentary adults: 0.8g per kg of body weight
- Active individuals: 1.2-2.0g per kg
- Athletes: 1.6-2.2g per kg

**Excellent protein sources:**
- Chicken breast: 31g protein per 100g
- Fish (salmon): 25g protein per 100g
- Eggs: 13g protein per 100g
- Greek yogurt: 10g protein per 100g
- Lentils: 9g protein per 100g
- Quinoa: 4.4g protein per 100g

**Benefits:**
- Muscle building and repair
- Satiety and weight management
- Immune system support

Do you have specific protein goals or dietary restrictions I should consider?"""

    elif any(word in message_lower for word in ["vitamin", "vitamins", "minerals"]):
        return """💊 **Essential Vitamins & Minerals:**

**Key vitamins:**
- **Vitamin C**: Immune support (citrus, berries, bell peppers)
- **Vitamin D**: Bone health (sunlight, fatty fish, fortified foods)
- **B vitamins**: Energy metabolism (whole grains, leafy greens)
- **Vitamin A**: Eye health (carrots, sweet potatoes, spinach)

**Important minerals:**
- **Iron**: Oxygen transport (red meat, spinach, lentils)
- **Calcium**: Bone health (dairy, leafy greens, almonds)
- **Potassium**: Heart health (bananas, potatoes, avocados)
- **Magnesium**: Muscle function (nuts, seeds, whole grains)

🍎 **Best approach:** Eat a variety of colorful fruits and vegetables!

Which vitamins or minerals are you particularly interested in?"""

    elif any(word in message_lower for word in ["weight", "lose", "gain", "diet"]):
        response = "🎯 **Weight Management Guidance:**\n\n"

        if profile:
            if profile.activity_level == "sedentary":
                response += "**For your sedentary lifestyle:**\n- Focus on portion control\n- Add light physical activity\n- Choose nutrient-dense, lower-calorie foods\n\n"
            elif profile.activity_level == "very_active":
                response += "**For your active lifestyle:**\n- Ensure adequate calorie intake\n- Focus on post-workout nutrition\n- Balance carbs and protein\n\n"

        response += """**General principles:**
- Create a moderate calorie deficit for weight loss (300-500 cal/day)
- Include protein at each meal (helps with satiety)
- Stay hydrated (often thirst is mistaken for hunger)
- Focus on whole foods over processed foods

**Sustainable approach:**
- Make gradual changes
- Don't eliminate entire food groups
- Include foods you enjoy in moderation

Would you like specific meal planning advice or information about healthy weight loss rates?"""
        return response

    elif any(word in message_lower for word in ["allergy", "allergies", "allergic"]):
        response = "🚨 **Food Allergy Information:**\n\n"

        if profile and profile.allergies:
            response += f"**I see you have allergies to:** {', '.join([a.allergy for a in profile.allergies])}\n\n"
            response += "**Safety tips:**\n- Always read food labels carefully\n- Inform restaurants about your allergies\n- Consider carrying emergency medication if prescribed\n\n"

        response += """**Common food allergens:**
- Milk, eggs, fish, shellfish
- Tree nuts, peanuts, wheat, soy
- Sesame (becoming more recognized)

**Cross-contamination prevention:**
- Use separate cooking utensils
- Clean surfaces thoroughly
- Be aware of shared manufacturing facilities

**Alternative options:**
- Many substitutes available for common allergens
- Focus on naturally allergen-free whole foods

Do you need help finding alternatives for specific allergens?"""
        return response

    elif any(word in message_lower for word in ["meal", "recipe", "cook", "prepare"]):
        response = "👨‍🍳 **Meal Planning & Preparation:**\n\n"

        if profile:
            if profile.dietary_preferences == "vegetarian":
                response += "**Vegetarian meal ideas:**\n- Lentil curry with quinoa\n- Vegetable stir-fry with tofu\n- Chickpea salad with avocado\n- Black bean and sweet potato bowl\n\n"
            elif profile.dietary_preferences == "vegan":
                response += "**Vegan meal ideas:**\n- Buddha bowl with tahini dressing\n- Lentil bolognese with pasta\n- Chickpea curry with brown rice\n- Quinoa stuffed bell peppers\n\n"

        response += """**Meal prep tips:**
- Plan weekly menus in advance
- Batch cook grains and proteins
- Pre-cut vegetables for easy cooking
- Use proper food storage containers

**Quick & healthy meals:**
- Sheet pan dinners (vegetables + protein)
- One-pot meals (stews, curries)
- Salad bowls with various toppings
- Smoothie bowls for breakfast

Would you like specific recipes or meal planning strategies?"""
        return response

    elif any(word in message_lower for word in ["water", "hydration", "drink"]):
        return """💧 **Hydration Guidelines:**

**Daily water needs:**
- Generally 8 cups (64 oz) per day
- More if you're active or in hot weather
- Listen to your body's thirst cues

**Signs of good hydration:**
- Light yellow urine
- Moist lips and mouth
- Good energy levels
- Elastic skin

**Hydrating foods:**
- Watermelon (92% water)
- Cucumber (96% water)
- Tomatoes (94% water)
- Lettuce (95% water)

**Tips for better hydration:**
- Start your day with a glass of water
- Keep a water bottle nearby
- Flavor water with lemon, cucumber, or mint
- Eat water-rich fruits and vegetables

Are you having trouble staying hydrated, or do you have specific questions about fluid intake?"""

    else:
        # Generic nutritional guidance
        response = "🥗 **Welcome to your Nutritional AI Assistant!**\n\n"

        if profile:
            response += f"I see you're {profile.age} years old"
            if profile.culture:
                response += f" and have {profile.culture} cultural preferences"
            response += ". I'll keep this in mind for my recommendations!\n\n"

        response += """I can help you with:
- **Nutrition information** (calories, proteins, vitamins)
- **Meal planning** and healthy recipes
- **Dietary guidance** for specific needs
- **Food allergies** and alternatives
- **Weight management** advice
- **Hydration** recommendations

**Popular topics:**
- "Tell me about protein sources"
- "How many calories should I eat?"
- "What are good sources of vitamin C?"
- "Help me plan a healthy meal"
- "I have a peanut allergy, what are alternatives?"

What would you like to know about nutrition today?"""
        return response


def generate_suggestions(message: str) -> List[str]:
    """Generate follow-up suggestions based on the message"""
    message_lower = message.lower()

    if any(word in message_lower for word in ["calorie", "calories"]):
        return [
            "How many calories should I eat per day?",
            "What are low-calorie snack options?",
            "Tell me about calorie-dense vs nutrient-dense foods",
        ]
    elif any(word in message_lower for word in ["protein"]):
        return [
            "What are the best plant-based proteins?",
            "How much protein do I need daily?",
            "Show me high-protein meal ideas",
        ]
    elif any(word in message_lower for word in ["vitamin", "mineral"]):
        return [
            "What foods are rich in vitamin C?",
            "Tell me about iron deficiency",
            "How can I get enough calcium?",
        ]
    else:
        return [
            "Tell me about balanced nutrition",
            "Help me plan healthy meals",
            "What are good snack options?",
            "How much water should I drink?",
        ]
