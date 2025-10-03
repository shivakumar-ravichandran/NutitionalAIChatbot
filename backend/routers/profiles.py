"""
Profile management API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import uuid

from database.database import get_db
from database.models import UserProfile, UserAllergy, UserHealthCondition
from schemas import (
    ProfileCreate,
    ProfileUpdate,
    ProfileResponse,
    AllergyCreate,
    HealthConditionCreate,
)

router = APIRouter()


@router.post("/", response_model=ProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_profile(profile_data: ProfileCreate, db: Session = Depends(get_db)):
    """Create a new user profile"""
    try:
        # Create profile
        db_profile = UserProfile(
            profile_uuid=str(uuid.uuid4()),
            age=profile_data.age,
            gender=profile_data.gender,
            location=profile_data.location,
            culture=profile_data.culture,
            dietary_preferences=profile_data.dietary_preferences,
            activity_level=profile_data.activity_level,
            response_style=profile_data.response_style,
            language=profile_data.language,
        )

        db.add(db_profile)
        db.flush()  # Get the ID

        # Add allergies
        for allergy_data in profile_data.allergies:
            db_allergy = UserAllergy(
                profile_id=db_profile.id,
                allergy=allergy_data.allergy,
                severity=allergy_data.severity,
            )
            db.add(db_allergy)

        # Add health conditions
        for condition_data in profile_data.health_conditions:
            db_condition = UserHealthCondition(
                profile_id=db_profile.id,
                condition_name=condition_data.condition_name,
                severity=condition_data.severity,
                notes=condition_data.notes,
            )
            db.add(db_condition)

        db.commit()
        db.refresh(db_profile)

        return db_profile

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating profile: {str(e)}",
        )


@router.get("/{profile_uuid}", response_model=ProfileResponse)
async def get_profile(profile_uuid: str, db: Session = Depends(get_db)):
    """Get user profile by UUID"""
    db_profile = (
        db.query(UserProfile).filter(UserProfile.profile_uuid == profile_uuid).first()
    )

    if not db_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found"
        )

    return db_profile


@router.put("/{profile_uuid}", response_model=ProfileResponse)
async def update_profile(
    profile_uuid: str, profile_update: ProfileUpdate, db: Session = Depends(get_db)
):
    """Update user profile"""
    db_profile = (
        db.query(UserProfile).filter(UserProfile.profile_uuid == profile_uuid).first()
    )

    if not db_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found"
        )

    try:
        # Update profile fields
        update_data = profile_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_profile, field, value)

        db.commit()
        db.refresh(db_profile)

        return db_profile

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating profile: {str(e)}",
        )


@router.post("/{profile_uuid}/allergies", status_code=status.HTTP_201_CREATED)
async def add_allergy(
    profile_uuid: str, allergy_data: AllergyCreate, db: Session = Depends(get_db)
):
    """Add allergy to user profile"""
    db_profile = (
        db.query(UserProfile).filter(UserProfile.profile_uuid == profile_uuid).first()
    )

    if not db_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found"
        )

    try:
        db_allergy = UserAllergy(
            profile_id=db_profile.id,
            allergy=allergy_data.allergy,
            severity=allergy_data.severity,
        )

        db.add(db_allergy)
        db.commit()

        return {"message": "Allergy added successfully"}

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding allergy: {str(e)}",
        )


@router.post("/{profile_uuid}/health-conditions", status_code=status.HTTP_201_CREATED)
async def add_health_condition(
    profile_uuid: str,
    condition_data: HealthConditionCreate,
    db: Session = Depends(get_db),
):
    """Add health condition to user profile"""
    db_profile = (
        db.query(UserProfile).filter(UserProfile.profile_uuid == profile_uuid).first()
    )

    if not db_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found"
        )

    try:
        db_condition = UserHealthCondition(
            profile_id=db_profile.id,
            condition_name=condition_data.condition_name,
            severity=condition_data.severity,
            notes=condition_data.notes,
        )

        db.add(db_condition)
        db.commit()

        return {"message": "Health condition added successfully"}

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding health condition: {str(e)}",
        )


@router.delete("/{profile_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile(profile_uuid: str, db: Session = Depends(get_db)):
    """Delete user profile"""
    db_profile = (
        db.query(UserProfile).filter(UserProfile.profile_uuid == profile_uuid).first()
    )

    if not db_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found"
        )

    try:
        db.delete(db_profile)
        db.commit()

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting profile: {str(e)}",
        )


@router.get("/", response_model=List[ProfileResponse])
async def list_profiles(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all profiles (for testing purposes)"""
    profiles = db.query(UserProfile).offset(skip).limit(limit).all()
    return profiles
