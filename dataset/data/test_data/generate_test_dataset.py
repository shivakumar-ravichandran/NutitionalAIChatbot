import json
import csv
import random
from datetime import datetime

# Define the comprehensive test dataset
test_queries = []

# Children (Ages 3-12) - 100 queries
children_queries = [
    # Toddlers (3-5 years)
    {
        "query_id": 1,
        "user_query": "My 4-year-old doesn't like vegetables. What Tamil food can I make that's healthy and tasty?",
        "user_profile": {
            "age": 4,
            "age_group": "children",
            "culture": "Tamil",
            "location": "Tamil Nadu",
            "dietary_preference": "vegetarian",
            "health_status": "healthy",
            "goal": "increase vegetable intake",
            "constraints": ["picky_eater"],
        },
        "scenario_type": "picky_eating",
        "complexity": "medium",
        "expected_elements": [
            "playful_tone",
            "tamil_dishes",
            "hidden_vegetables",
            "fun_preparation",
        ],
    },
    {
        "query_id": 2,
        "user_query": "What Punjabi breakfast can I give my 3-year-old for energy?",
        "user_profile": {
            "age": 3,
            "age_group": "children",
            "culture": "Punjabi",
            "location": "Punjab",
            "dietary_preference": "vegetarian",
            "health_status": "active",
            "goal": "energy_rich_foods",
            "constraints": ["small_portions"],
        },
        "scenario_type": "energy_nutrition",
        "complexity": "low",
        "expected_elements": [
            "child_friendly_language",
            "punjabi_breakfast",
            "energy_foods",
            "portion_guidance",
        ],
    },
    {
        "query_id": 3,
        "user_query": "My toddler is constipated. What Bengali foods help with digestion?",
        "user_profile": {
            "age": 5,
            "age_group": "children",
            "culture": "Bengali",
            "location": "West Bengal",
            "dietary_preference": "non_vegetarian",
            "health_status": "digestive_issues",
            "goal": "improve_digestion",
            "constraints": ["medical_concern"],
        },
        "scenario_type": "health_issue",
        "complexity": "high",
        "expected_elements": [
            "health_focused",
            "bengali_foods",
            "fiber_rich",
            "gentle_suggestions",
        ],
    },
    {
        "query_id": 4,
        "user_query": "What Gujarati snacks are good for my 4-year-old's school tiffin?",
        "user_profile": {
            "age": 4,
            "age_group": "children",
            "culture": "Gujarati",
            "location": "Gujarat",
            "dietary_preference": "vegetarian",
            "health_status": "healthy",
            "goal": "nutritious_snacks",
            "constraints": ["school_appropriate", "no_refrigeration"],
        },
        "scenario_type": "school_nutrition",
        "complexity": "medium",
        "expected_elements": [
            "gujarati_snacks",
            "portable_food",
            "child_appeal",
            "nutritional_balance",
        ],
    },
    {
        "query_id": 5,
        "user_query": "My 5-year-old loves sweets. What healthy Maharashtrian sweet options can I give?",
        "user_profile": {
            "age": 5,
            "age_group": "children",
            "culture": "Maharashtrian",
            "location": "Maharashtra",
            "dietary_preference": "vegetarian",
            "health_status": "healthy",
            "goal": "healthier_sweets",
            "constraints": ["sweet_tooth"],
        },
        "scenario_type": "healthy_alternatives",
        "complexity": "medium",
        "expected_elements": [
            "healthy_sweets",
            "maharashtrian_options",
            "natural_sweeteners",
            "moderation_advice",
        ],
    },
    # School age children (6-12 years) - continuing with more diverse scenarios
    {
        "query_id": 6,
        "user_query": "What should I pack for my 8-year-old's lunch that gives energy for sports?",
        "user_profile": {
            "age": 8,
            "age_group": "children",
            "culture": "Punjabi",
            "location": "Punjab",
            "dietary_preference": "vegetarian",
            "health_status": "athletic",
            "goal": "sports_nutrition",
            "constraints": ["school_lunch", "energy_needs"],
        },
        "scenario_type": "athletic_nutrition",
        "complexity": "medium",
        "expected_elements": [
            "energy_foods",
            "portable_lunch",
            "sports_nutrition",
            "punjabi_cuisine",
        ],
    },
    {
        "query_id": 7,
        "user_query": "My 10-year-old daughter is underweight. What Kerala foods can help her gain healthy weight?",
        "user_profile": {
            "age": 10,
            "age_group": "children",
            "culture": "Malayalam",
            "location": "Kerala",
            "dietary_preference": "non_vegetarian",
            "health_status": "underweight",
            "goal": "healthy_weight_gain",
            "constraints": ["medical_supervision"],
        },
        "scenario_type": "weight_management",
        "complexity": "high",
        "expected_elements": [
            "weight_gain_foods",
            "kerala_cuisine",
            "calorie_dense",
            "balanced_nutrition",
        ],
    },
    {
        "query_id": 8,
        "user_query": "What iron-rich Rajasthani foods can I give my 9-year-old who has low hemoglobin?",
        "user_profile": {
            "age": 9,
            "age_group": "children",
            "culture": "Rajasthani",
            "location": "Rajasthan",
            "dietary_preference": "vegetarian",
            "health_status": "anemic",
            "goal": "increase_iron_levels",
            "constraints": ["medical_condition"],
        },
        "scenario_type": "nutritional_deficiency",
        "complexity": "high",
        "expected_elements": [
            "iron_rich_foods",
            "rajasthani_cuisine",
            "absorption_enhancers",
            "medical_awareness",
        ],
    },
    {
        "query_id": 9,
        "user_query": "My 7-year-old is allergic to nuts. What safe Assamese snacks can I make?",
        "user_profile": {
            "age": 7,
            "age_group": "children",
            "culture": "Assamese",
            "location": "Assam",
            "dietary_preference": "non_vegetarian",
            "health_status": "food_allergies",
            "goal": "safe_nutrition",
            "constraints": ["nut_allergy", "allergy_safety"],
        },
        "scenario_type": "food_allergies",
        "complexity": "high",
        "expected_elements": [
            "allergy_safe",
            "assamese_snacks",
            "nut_free",
            "safety_emphasis",
        ],
    },
    {
        "query_id": 10,
        "user_query": "What finger foods from Odia cuisine are good for my 6-year-old's birthday party?",
        "user_profile": {
            "age": 6,
            "age_group": "children",
            "culture": "Odia",
            "location": "Odisha",
            "dietary_preference": "vegetarian",
            "health_status": "healthy",
            "goal": "party_food",
            "constraints": ["finger_foods", "kid_friendly", "party_appropriate"],
        },
        "scenario_type": "special_occasion",
        "complexity": "medium",
        "expected_elements": [
            "odia_cuisine",
            "finger_foods",
            "party_appeal",
            "nutritional_balance",
        ],
    },
]

# Continue with more children queries (90 more to reach 100 total)
for i in range(11, 101):
    age = random.randint(3, 12)
    cultures = [
        "Tamil",
        "Punjabi",
        "Bengali",
        "Gujarati",
        "Maharashtrian",
        "Malayalam",
        "Assamese",
        "Odia",
        "Telugu",
        "Kannada",
    ]
    culture = random.choice(cultures)

    scenarios = [
        "breakfast_ideas",
        "lunch_planning",
        "dinner_options",
        "snack_suggestions",
        "growth_nutrition",
        "immunity_boosting",
        "brain_development",
        "bone_health",
        "digestive_health",
        "seasonal_eating",
        "festival_foods",
        "travel_nutrition",
    ]

    query_templates = {
        "breakfast_ideas": f"What {culture} breakfast is good for my {age}-year-old's brain development?",
        "lunch_planning": f"Help me plan a week's lunch for my {age}-year-old with {culture} food",
        "dinner_options": f"My {age}-year-old gets hungry at night. What {culture} dinner keeps them full?",
        "snack_suggestions": f"What healthy {culture} snacks can I give my {age}-year-old after school?",
        "growth_nutrition": f"My {age}-year-old is shorter than peers. What {culture} foods support growth?",
        "immunity_boosting": f"My {age}-year-old gets sick often. What {culture} foods boost immunity?",
        "brain_development": f"What {culture} foods are best for my {age}-year-old's concentration and memory?",
        "bone_health": f"How can I strengthen my {age}-year-old's bones with {culture} foods?",
        "digestive_health": f"My {age}-year-old has stomach issues. What gentle {culture} foods can help?",
        "seasonal_eating": f"What {culture} foods are best for my {age}-year-old during winter/summer?",
        "festival_foods": f"What healthy {culture} festival foods can I make for my {age}-year-old?",
        "travel_nutrition": f"We're traveling. What {culture} foods travel well for my {age}-year-old?",
    }

    scenario = random.choice(scenarios)
    query = query_templates[scenario]

    children_queries.append(
        {
            "query_id": i,
            "user_query": query,
            "user_profile": {
                "age": age,
                "age_group": "children",
                "culture": culture,
                "location": f"Sample_{culture}_location",
                "dietary_preference": random.choice(["vegetarian", "non_vegetarian"]),
                "health_status": random.choice(
                    ["healthy", "growth_concerns", "picky_eater", "active"]
                ),
                "goal": scenario.replace("_", " "),
                "constraints": random.sample(
                    [
                        "budget_conscious",
                        "time_limited",
                        "picky_eater",
                        "simple_cooking",
                    ],
                    2,
                ),
            },
            "scenario_type": scenario,
            "complexity": random.choice(["low", "medium", "high"]),
            "expected_elements": [
                f"{culture.lower()}_cuisine",
                "child_friendly",
                "nutritional_focus",
                "age_appropriate",
            ],
        }
    )

# Teenagers (Ages 13-17) - 75 queries
teenager_queries = []
for i in range(101, 176):
    age = random.randint(13, 17)
    cultures = [
        "Tamil",
        "Punjabi",
        "Bengali",
        "Gujarati",
        "Maharashtrian",
        "Malayalam",
        "Assamese",
        "Odia",
        "Telugu",
        "Kannada",
    ]
    culture = random.choice(cultures)

    scenarios = [
        "body_image",
        "acne_nutrition",
        "sports_performance",
        "exam_stress",
        "growth_spurt",
        "peer_pressure",
        "independence",
        "busy_schedule",
        "junk_food_alternatives",
        "dating_confidence",
    ]

    query_templates = {
        "body_image": f"I'm {age} and want to look fit. What {culture} foods help with a healthy body?",
        "acne_nutrition": f"I'm {age} with acne problems. What {culture} foods are good for clear skin?",
        "sports_performance": f"I'm {age} and play cricket/football. What {culture} foods boost my performance?",
        "exam_stress": f"I'm {age} with board exams. What {culture} brain foods help with studying?",
        "growth_spurt": f"I'm {age} and growing fast. What {culture} foods support my growth?",
        "peer_pressure": f"I'm {age} and friends eat junk. What cool {culture} alternatives can I choose?",
        "independence": f"I'm {age} and want to cook my own {culture} meals. What's easy and healthy?",
        "busy_schedule": f"I'm {age} with classes and coaching. What quick {culture} meals work?",
        "junk_food_alternatives": f"I'm {age} and love pizza/burgers. What {culture} versions are healthier?",
        "dating_confidence": f"I'm {age} and want to feel confident. What {culture} foods boost energy and mood?",
    }

    scenario = random.choice(scenarios)
    query = query_templates[scenario]

    teenager_queries.append(
        {
            "query_id": i,
            "user_query": query,
            "user_profile": {
                "age": age,
                "age_group": "teenagers",
                "culture": culture,
                "location": f"Sample_{culture}_location",
                "dietary_preference": random.choice(
                    ["vegetarian", "non_vegetarian", "flexitarian"]
                ),
                "health_status": random.choice(
                    ["healthy", "acne", "athletic", "stressed", "growing"]
                ),
                "goal": scenario.replace("_", " "),
                "constraints": random.sample(
                    [
                        "peer_pressure",
                        "limited_budget",
                        "busy_schedule",
                        "independence",
                    ],
                    2,
                ),
            },
            "scenario_type": scenario,
            "complexity": random.choice(["medium", "high"]),
            "expected_elements": [
                f"{culture.lower()}_cuisine",
                "teen_friendly",
                "motivational",
                "peer_aware",
            ],
        }
    )

# Adults (Ages 18-59) - 200 queries
adult_queries = []

# Detailed adult scenarios with specific examples
adult_detailed_queries = [
    {
        "query_id": 176,
        "user_query": "I'm 32 and trying to lose weight with a busy work schedule. Can you suggest quick Maharashtrian meals that are healthy?",
        "user_profile": {
            "age": 32,
            "age_group": "adults",
            "culture": "Maharashtrian",
            "location": "Maharashtra",
            "dietary_preference": "non_vegetarian",
            "health_status": "overweight",
            "goal": "weight_loss",
            "constraints": ["busy_schedule", "limited_cooking_time"],
        },
        "scenario_type": "weight_management_professional",
        "complexity": "high",
        "expected_elements": [
            "maharashtrian_cuisine",
            "low_calorie",
            "quick_recipes",
            "professional_tone",
        ],
    },
    {
        "query_id": 177,
        "user_query": "I have diabetes and want to control blood sugar while enjoying Kerala food. What should I eat?",
        "user_profile": {
            "age": 45,
            "age_group": "adults",
            "culture": "Malayalam",
            "location": "Kerala",
            "dietary_preference": "vegetarian",
            "health_status": "diabetic_type2",
            "goal": "blood_sugar_control",
            "constraints": ["traditional_preferences", "medical_supervision"],
        },
        "scenario_type": "diabetes_management",
        "complexity": "high",
        "expected_elements": [
            "kerala_cuisine",
            "low_glycemic",
            "portion_control",
            "medical_awareness",
        ],
    },
    {
        "query_id": 178,
        "user_query": "I'm pregnant (2nd trimester) and need Bengali foods that are nutritious but won't trigger nausea.",
        "user_profile": {
            "age": 28,
            "age_group": "adults",
            "culture": "Bengali",
            "location": "West Bengal",
            "dietary_preference": "pescatarian",
            "health_status": "pregnant_2nd_trimester",
            "goal": "pregnancy_nutrition",
            "constraints": ["morning_sickness", "food_aversions"],
        },
        "scenario_type": "pregnancy_nutrition",
        "complexity": "high",
        "expected_elements": [
            "bengali_cuisine",
            "pregnancy_safe",
            "nausea_friendly",
            "nutritional_density",
        ],
    },
    {
        "query_id": 179,
        "user_query": "I'm training for a marathon and need Maharashtrian foods for performance and recovery.",
        "user_profile": {
            "age": 24,
            "age_group": "adults",
            "culture": "Marathi",
            "location": "Maharashtra",
            "dietary_preference": "non_vegetarian",
            "health_status": "athletic_endurance",
            "goal": "sports_performance",
            "constraints": ["training_schedule", "recovery_needs"],
        },
        "scenario_type": "athletic_performance",
        "complexity": "high",
        "expected_elements": [
            "maharashtrian_cuisine",
            "performance_nutrition",
            "recovery_foods",
            "timing_guidance",
        ],
    },
    {
        "query_id": 180,
        "user_query": "I work night shifts and need Punjabi meal ideas that suit my reversed schedule.",
        "user_profile": {
            "age": 29,
            "age_group": "adults",
            "culture": "Punjabi",
            "location": "Punjab",
            "dietary_preference": "vegetarian",
            "health_status": "shift_worker",
            "goal": "schedule_adaptation",
            "constraints": ["night_shifts", "irregular_sleep", "digestive_concerns"],
        },
        "scenario_type": "shift_work_nutrition",
        "complexity": "high",
        "expected_elements": [
            "punjabi_cuisine",
            "shift_appropriate",
            "energy_management",
            "digestive_friendly",
        ],
    },
]

adult_queries.extend(adult_detailed_queries)

# Generate remaining adult queries (195 more)
for i in range(181, 376):
    age = random.randint(18, 59)
    cultures = [
        "Tamil",
        "Punjabi",
        "Bengali",
        "Gujarati",
        "Maharashtrian",
        "Malayalam",
        "Assamese",
        "Odia",
        "Telugu",
        "Kannada",
        "Marathi",
        "Rajasthani",
    ]
    culture = random.choice(cultures)

    scenarios = [
        "weight_loss",
        "muscle_building",
        "diabetes_management",
        "hypertension",
        "cholesterol_control",
        "pregnancy_nutrition",
        "breastfeeding",
        "PCOS_management",
        "thyroid_issues",
        "IBS_management",
        "busy_professional",
        "budget_meals",
        "meal_prep",
        "family_cooking",
        "entertaining_guests",
        "travel_nutrition",
        "festival_preparation",
        "seasonal_eating",
        "immunity_boosting",
        "stress_management",
    ]

    query_templates = {
        "weight_loss": f"I'm {age} and need to lose 15kg. What {culture} meal plan works for sustainable weight loss?",
        "muscle_building": f"I'm {age} and want to build muscle. What {culture} protein-rich foods should I eat?",
        "diabetes_management": f"I'm {age} with diabetes. How can I enjoy {culture} food while controlling blood sugar?",
        "hypertension": f"I'm {age} with high BP. What low-sodium {culture} recipes can I try?",
        "cholesterol_control": f"I'm {age} with high cholesterol. What heart-healthy {culture} foods should I choose?",
        "pregnancy_nutrition": f"I'm {age} and pregnant. What {culture} foods ensure proper nutrition for my baby?",
        "breastfeeding": f"I'm {age} and breastfeeding. What {culture} foods increase milk production and quality?",
        "PCOS_management": f"I'm {age} with PCOS. What {culture} foods help manage my symptoms?",
        "thyroid_issues": f"I'm {age} with thyroid problems. What {culture} foods support thyroid health?",
        "IBS_management": f"I'm {age} with IBS. What gentle {culture} foods won't trigger my symptoms?",
        "busy_professional": f"I'm {age} with a demanding job. What quick {culture} meals fit my hectic schedule?",
        "budget_meals": f"I'm {age} on a tight budget. What affordable {culture} meals are still nutritious?",
        "meal_prep": f"I'm {age} and want to meal prep. What {culture} dishes store and reheat well?",
        "family_cooking": f"I'm {age} cooking for my family. What {culture} meals satisfy everyone's tastes?",
        "entertaining_guests": f"I'm {age} hosting friends. What impressive yet healthy {culture} dishes can I make?",
        "travel_nutrition": f"I'm {age} and travel frequently. What {culture} foods travel well and stay nutritious?",
        "festival_preparation": f"I'm {age} preparing for festivals. What healthier versions of {culture} traditional sweets exist?",
        "seasonal_eating": f"I'm {age} and want to eat seasonally. What {culture} foods are best this season?",
        "immunity_boosting": f"I'm {age} and get sick often. What {culture} foods naturally boost immunity?",
        "stress_management": f"I'm {age} under work stress. What {culture} foods help manage stress and improve mood?",
    }

    scenario = random.choice(scenarios)
    query = query_templates[scenario]

    adult_queries.append(
        {
            "query_id": i,
            "user_query": query,
            "user_profile": {
                "age": age,
                "age_group": "adults",
                "culture": culture,
                "location": f"Sample_{culture}_location",
                "dietary_preference": random.choice(
                    ["vegetarian", "non_vegetarian", "pescatarian", "flexitarian"]
                ),
                "health_status": random.choice(
                    [
                        "healthy",
                        "overweight",
                        "underweight",
                        "diabetic",
                        "hypertensive",
                        "pregnant",
                        "stressed",
                    ]
                ),
                "goal": scenario.replace("_", " "),
                "constraints": random.sample(
                    [
                        "time_limited",
                        "budget_conscious",
                        "family_considerations",
                        "health_restrictions",
                        "skill_level",
                    ],
                    2,
                ),
            },
            "scenario_type": scenario,
            "complexity": random.choice(["medium", "high"]),
            "expected_elements": [
                f"{culture.lower()}_cuisine",
                "professional_tone",
                "evidence_based",
                "practical_advice",
            ],
        }
    )

# Elderly (Ages 60+) - 125 queries
elderly_queries = []

# Detailed elderly scenarios
elderly_detailed_queries = [
    {
        "query_id": 376,
        "user_query": "I'm 68 with heart problems. What low-sodium Gujarati foods can I still enjoy?",
        "user_profile": {
            "age": 68,
            "age_group": "elderly",
            "culture": "Gujarati",
            "location": "Gujarat",
            "dietary_preference": "vegetarian",
            "health_status": "cardiovascular_disease",
            "goal": "heart_health",
            "constraints": ["low_sodium", "traditional_preferences"],
        },
        "scenario_type": "cardiovascular_management",
        "complexity": "high",
        "expected_elements": [
            "gujarati_cuisine",
            "heart_healthy",
            "low_sodium",
            "respectful_tone",
        ],
    },
    {
        "query_id": 377,
        "user_query": "I'm 72 with digestive issues. What soft Tamil foods are easy on the stomach but nutritious?",
        "user_profile": {
            "age": 72,
            "age_group": "elderly",
            "culture": "Tamil",
            "location": "Tamil Nadu",
            "dietary_preference": "vegetarian",
            "health_status": "digestive_problems",
            "goal": "digestive_comfort",
            "constraints": ["soft_foods", "easy_digestion", "reduced_appetite"],
        },
        "scenario_type": "digestive_health_elderly",
        "complexity": "high",
        "expected_elements": [
            "tamil_cuisine",
            "easily_digestible",
            "nutritional_density",
            "gentle_preparation",
        ],
    },
]

elderly_queries.extend(elderly_detailed_queries)

# Generate remaining elderly queries (123 more)
for i in range(378, 501):
    age = random.randint(60, 90)
    cultures = [
        "Tamil",
        "Punjabi",
        "Bengali",
        "Gujarati",
        "Maharashtrian",
        "Malayalam",
        "Assamese",
        "Odia",
        "Telugu",
        "Kannada",
    ]
    culture = random.choice(cultures)

    scenarios = [
        "bone_health",
        "heart_health",
        "diabetes_elderly",
        "blood_pressure",
        "arthritis_nutrition",
        "memory_support",
        "digestive_comfort",
        "medication_interactions",
        "appetite_loss",
        "swallowing_difficulties",
        "loneliness_eating",
        "budget_fixed_income",
        "easy_cooking",
        "nutrient_absorption",
        "hydration_needs",
    ]

    query_templates = {
        "bone_health": f"I'm {age} with osteoporosis. What {culture} foods strengthen bones at my age?",
        "heart_health": f"I'm {age} with heart conditions. What gentle {culture} foods support heart health?",
        "diabetes_elderly": f"I'm {age} with diabetes for 20 years. What {culture} foods keep my sugar stable?",
        "blood_pressure": f"I'm {age} with fluctuating BP. What {culture} foods help maintain steady pressure?",
        "arthritis_nutrition": f"I'm {age} with painful joints. What anti-inflammatory {culture} foods help?",
        "memory_support": f"I'm {age} and forgetting things. What {culture} brain foods support memory?",
        "digestive_comfort": f"I'm {age} with sensitive digestion. What gentle {culture} foods agree with me?",
        "medication_interactions": f"I'm {age} on multiple medicines. What {culture} foods are safe to eat?",
        "appetite_loss": f"I'm {age} and don't feel hungry. What appealing {culture} foods encourage eating?",
        "swallowing_difficulties": f"I'm {age} with swallowing problems. What {culture} foods are safe and easy?",
        "loneliness_eating": f"I'm {age} and eat alone. What simple {culture} meals motivate me to cook?",
        "budget_fixed_income": f"I'm {age} on pension. What affordable {culture} meals are still healthy?",
        "easy_cooking": f"I'm {age} and cooking is hard now. What simple {culture} recipes work?",
        "nutrient_absorption": f"I'm {age} with absorption issues. What {culture} foods give maximum nutrition?",
        "hydration_needs": f"I'm {age} and forget to drink water. What {culture} foods help with hydration?",
    }

    scenario = random.choice(scenarios)
    query = query_templates[scenario]

    elderly_queries.append(
        {
            "query_id": i,
            "user_query": query,
            "user_profile": {
                "age": age,
                "age_group": "elderly",
                "culture": culture,
                "location": f"Sample_{culture}_location",
                "dietary_preference": random.choice(["vegetarian", "non_vegetarian"]),
                "health_status": random.choice(
                    [
                        "multiple_conditions",
                        "frail",
                        "active_senior",
                        "chronic_disease",
                        "medication_dependent",
                    ]
                ),
                "goal": scenario.replace("_", " "),
                "constraints": random.sample(
                    [
                        "physical_limitations",
                        "fixed_income",
                        "medication_interactions",
                        "reduced_appetite",
                        "cooking_difficulties",
                    ],
                    3,
                ),
            },
            "scenario_type": scenario,
            "complexity": random.choice(["medium", "high"]),
            "expected_elements": [
                f"{culture.lower()}_cuisine",
                "respectful_tone",
                "health_focused",
                "simple_explanations",
            ],
        }
    )

# Combine all queries
all_queries = children_queries + teenager_queries + adult_queries + elderly_queries

print(f"Generated {len(all_queries)} queries")
print(f"Children: {len(children_queries)}")
print(f"Teenagers: {len(teenager_queries)}")
print(f"Adults: {len(adult_queries)}")
print(f"Elderly: {len(elderly_queries)}")

# Save as JSON
with open("comprehensive_test_queries.json", "w", encoding="utf-8") as f:
    json.dump(all_queries, f, indent=2, ensure_ascii=False)

# Save as CSV for easy analysis
csv_data = []
for query in all_queries:
    row = {
        "query_id": query["query_id"],
        "user_query": query["user_query"],
        "age": query["user_profile"]["age"],
        "age_group": query["user_profile"]["age_group"],
        "culture": query["user_profile"]["culture"],
        "location": query["user_profile"]["location"],
        "dietary_preference": query["user_profile"]["dietary_preference"],
        "health_status": query["user_profile"]["health_status"],
        "goal": query["user_profile"]["goal"],
        "constraints": ", ".join(query["user_profile"]["constraints"]),
        "scenario_type": query["scenario_type"],
        "complexity": query["complexity"],
        "expected_elements": ", ".join(query["expected_elements"]),
    }
    csv_data.append(row)

with open("comprehensive_test_queries.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
    writer.writeheader()
    writer.writerows(csv_data)

print("Dataset files created successfully!")
