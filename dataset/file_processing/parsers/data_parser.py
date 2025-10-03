"""
Data Parser for Age-Specific Nutritional Information Extraction

This module handles CSV and JSON files to extract age-specific nutritional information including:
- Age-group dietary requirements
- Nutritional needs by life stage
- Growth and development nutrition
- Age-specific portion sizes
- Developmental milestones nutrition
- Senior citizen dietary needs
- Pediatric nutrition guidelines
- Adult nutrition recommendations
"""

import logging
import csv
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgeSpecificData:
    """Data class for structured age-specific nutritional information"""

    age_group: str = ""
    min_age_months: Optional[int] = None
    max_age_months: Optional[int] = None
    gender: str = "all"  # all, male, female

    # Nutritional requirements
    calories_per_day: Optional[float] = None
    protein_grams_per_day: Optional[float] = None
    carbs_grams_per_day: Optional[float] = None
    fat_grams_per_day: Optional[float] = None
    fiber_grams_per_day: Optional[float] = None

    # Vitamins (daily requirements)
    vitamins: Dict[str, float] = None

    # Minerals (daily requirements)
    minerals: Dict[str, float] = None

    # Specific nutritional guidelines
    recommended_foods: List[str] = None
    foods_to_avoid: List[str] = None
    portion_sizes: Dict[str, str] = None  # food_category -> portion_description

    # Age-specific considerations
    developmental_needs: List[str] = None
    special_considerations: List[str] = None
    meal_frequency: Optional[int] = None  # meals per day
    snack_frequency: Optional[int] = None  # snacks per day

    # Hydration
    water_intake_ml: Optional[float] = None

    # Activity level considerations
    sedentary_calories: Optional[float] = None
    moderate_activity_calories: Optional[float] = None
    high_activity_calories: Optional[float] = None

    # Health conditions common to age group
    common_deficiencies: List[str] = None
    health_risks: List[str] = None

    # Sources and metadata
    source_document: str = ""
    data_source: str = ""  # WHO, FDA, etc.
    last_updated: Optional[str] = None
    extraction_confidence: float = 0.0

    def __post_init__(self):
        """Initialize mutable default values"""
        if self.vitamins is None:
            self.vitamins = {}
        if self.minerals is None:
            self.minerals = {}
        if self.recommended_foods is None:
            self.recommended_foods = []
        if self.foods_to_avoid is None:
            self.foods_to_avoid = []
        if self.portion_sizes is None:
            self.portion_sizes = {}
        if self.developmental_needs is None:
            self.developmental_needs = []
        if self.special_considerations is None:
            self.special_considerations = []
        if self.common_deficiencies is None:
            self.common_deficiencies = []
        if self.health_risks is None:
            self.health_risks = []


class DataAgeSpecificParser:
    """Main class for parsing age-specific nutritional information from data files"""

    def __init__(self):
        self.age_group_mappings = self._initialize_age_groups()
        self.nutrient_mappings = self._initialize_nutrient_mappings()
        self.standard_columns = self._initialize_standard_columns()

    def _initialize_age_groups(self) -> Dict[str, Dict[str, int]]:
        """Initialize standard age group definitions"""
        return {
            "infant_0_6m": {"min_months": 0, "max_months": 6},
            "infant_6_12m": {"min_months": 6, "max_months": 12},
            "toddler_1_3y": {"min_months": 12, "max_months": 36},
            "preschool_3_6y": {"min_months": 36, "max_months": 72},
            "school_6_12y": {"min_months": 72, "max_months": 144},
            "teen_12_18y": {"min_months": 144, "max_months": 216},
            "adult_18_65y": {"min_months": 216, "max_months": 780},
            "senior_65plus": {"min_months": 780, "max_months": None},
            "pregnant": {"min_months": 216, "max_months": 540},  # 18-45 typical
            "lactating": {"min_months": 216, "max_months": 540},
        }

    def _initialize_nutrient_mappings(self) -> Dict[str, str]:
        """Initialize mappings for nutrient names to standardized names"""
        return {
            # Energy
            "energy": "calories_per_day",
            "calories": "calories_per_day",
            "kcal": "calories_per_day",
            "cal": "calories_per_day",
            # Macronutrients
            "protein": "protein_grams_per_day",
            "carbs": "carbs_grams_per_day",
            "carbohydrates": "carbs_grams_per_day",
            "fat": "fat_grams_per_day",
            "lipids": "fat_grams_per_day",
            "fiber": "fiber_grams_per_day",
            "fibre": "fiber_grams_per_day",
            # Hydration
            "water": "water_intake_ml",
            "fluids": "water_intake_ml",
            "hydration": "water_intake_ml",
            # Activity levels
            "sedentary": "sedentary_calories",
            "moderate": "moderate_activity_calories",
            "active": "high_activity_calories",
        }

    def _initialize_standard_columns(self) -> Dict[str, List[str]]:
        """Initialize expected column names for different data types"""
        return {
            "age_identifiers": [
                "age",
                "age_group",
                "age_range",
                "life_stage",
                "min_age",
                "max_age",
                "age_months",
                "age_years",
            ],
            "gender_identifiers": ["gender", "sex", "male", "female", "all"],
            "nutrient_columns": [
                "calories",
                "protein",
                "carbs",
                "fat",
                "fiber",
                "vitamin_a",
                "vitamin_c",
                "vitamin_d",
                "calcium",
                "iron",
            ],
            "recommendation_columns": [
                "recommended_foods",
                "avoid_foods",
                "portion_size",
                "meal_frequency",
                "considerations",
                "notes",
            ],
        }

    def parse_csv(self, file_path: str) -> List[AgeSpecificData]:
        """
        Parse a CSV file and extract age-specific nutritional information

        Args:
            file_path: Path to the CSV file

        Returns:
            List of AgeSpecificData objects containing extracted information
        """
        logger.info(f"Parsing CSV file: {file_path}")

        try:
            # Try different encodings
            encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                logger.error(f"Could not read CSV file with any encoding: {file_path}")
                return []

            logger.info(f"CSV contains {len(df)} rows and {len(df.columns)} columns")

            # Analyze and parse the CSV structure
            age_data = self._parse_dataframe(df, file_path)

            logger.info(f"Extracted {len(age_data)} age-specific entries from CSV")
            return age_data

        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {str(e)}")
            return []

    def parse_json(self, file_path: str) -> List[AgeSpecificData]:
        """
        Parse a JSON file and extract age-specific nutritional information

        Args:
            file_path: Path to the JSON file

        Returns:
            List of AgeSpecificData objects containing extracted information
        """
        logger.info(f"Parsing JSON file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            age_data = []

            if isinstance(json_data, list):
                # Array of objects
                for item in json_data:
                    if isinstance(item, dict):
                        data = self._parse_json_object(item, file_path)
                        if data:
                            age_data.append(data)

            elif isinstance(json_data, dict):
                # Single object or nested structure
                if self._is_single_age_record(json_data):
                    data = self._parse_json_object(json_data, file_path)
                    if data:
                        age_data.append(data)
                else:
                    # Nested structure - iterate through keys
                    for key, value in json_data.items():
                        if isinstance(value, dict):
                            data = self._parse_json_object(value, file_path)
                            if data:
                                # Use key as age group if not specified in data
                                if not data.age_group:
                                    data.age_group = key
                                age_data.append(data)
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    data = self._parse_json_object(item, file_path)
                                    if data:
                                        age_data.append(data)

            logger.info(f"Extracted {len(age_data)} age-specific entries from JSON")
            return age_data

        except Exception as e:
            logger.error(f"Error parsing JSON file {file_path}: {str(e)}")
            return []

    def _parse_dataframe(
        self, df: pd.DataFrame, source_file: str
    ) -> List[AgeSpecificData]:
        """Parse pandas DataFrame and extract age-specific data"""
        age_data = []

        # Normalize column names
        df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

        # Identify key columns
        age_col = self._find_column(df, self.standard_columns["age_identifiers"])
        gender_col = self._find_column(df, self.standard_columns["gender_identifiers"])

        logger.info(f"Identified age column: {age_col}, gender column: {gender_col}")

        for index, row in df.iterrows():
            try:
                data = AgeSpecificData(source_document=source_file)
                extraction_count = 0

                # Extract age information
                if age_col:
                    age_value = str(row[age_col]).strip()
                    data.age_group, data.min_age_months, data.max_age_months = (
                        self._parse_age_value(age_value)
                    )
                    if data.age_group:
                        extraction_count += 1

                # Extract gender information
                if gender_col:
                    gender_value = str(row[gender_col]).strip().lower()
                    data.gender = self._normalize_gender(gender_value)
                    if data.gender != "all":
                        extraction_count += 1

                # Extract nutritional values
                for col in df.columns:
                    col_lower = col.lower()
                    value = row[col]

                    if pd.isna(value) or value == "":
                        continue

                    # Map column to nutrient
                    if col_lower in self.nutrient_mappings:
                        attr_name = self.nutrient_mappings[col_lower]
                        try:
                            numeric_value = self._extract_numeric_value(str(value))
                            if numeric_value is not None:
                                setattr(data, attr_name, numeric_value)
                                extraction_count += 1
                        except (ValueError, TypeError):
                            continue

                    # Handle vitamins and minerals
                    elif "vitamin" in col_lower or "mineral" in col_lower:
                        try:
                            numeric_value = self._extract_numeric_value(str(value))
                            if numeric_value is not None:
                                if "vitamin" in col_lower:
                                    vitamin_name = col_lower.replace("_", " ").title()
                                    data.vitamins[vitamin_name] = numeric_value
                                else:
                                    mineral_name = col_lower.replace("_", " ").title()
                                    data.minerals[mineral_name] = numeric_value
                                extraction_count += 1
                        except (ValueError, TypeError):
                            continue

                    # Handle recommendations and lists
                    elif any(
                        keyword in col_lower
                        for keyword in ["recommend", "avoid", "food"]
                    ):
                        text_value = str(value).strip()
                        if text_value and text_value.lower() != "nan":
                            if "recommend" in col_lower:
                                data.recommended_foods = self._parse_food_list(
                                    text_value
                                )
                            elif "avoid" in col_lower:
                                data.foods_to_avoid = self._parse_food_list(text_value)
                            extraction_count += 1

                    # Handle special considerations
                    elif any(
                        keyword in col_lower
                        for keyword in ["consideration", "note", "special"]
                    ):
                        text_value = str(value).strip()
                        if text_value and text_value.lower() != "nan":
                            data.special_considerations = (
                                self._parse_consideration_list(text_value)
                            )
                            extraction_count += 1

                # Set extraction confidence
                data.extraction_confidence = min(extraction_count / 10.0, 1.0)

                if self._is_valid_age_data(data):
                    age_data.append(data)

            except Exception as e:
                logger.warning(f"Error processing row {index}: {str(e)}")
                continue

        return age_data

    def _parse_json_object(
        self, obj: Dict[str, Any], source_file: str
    ) -> Optional[AgeSpecificData]:
        """Parse a single JSON object into AgeSpecificData"""
        data = AgeSpecificData(source_document=source_file)
        extraction_count = 0

        for key, value in obj.items():
            if value is None or value == "":
                continue

            key_lower = key.lower().strip().replace(" ", "_")

            # Direct attribute mapping
            if hasattr(data, key_lower):
                try:
                    if isinstance(value, (int, float)):
                        setattr(data, key_lower, value)
                        extraction_count += 1
                    elif isinstance(value, str) and key_lower in [
                        "age_group",
                        "gender",
                        "source_document",
                    ]:
                        setattr(data, key_lower, value)
                        extraction_count += 1
                except (ValueError, TypeError):
                    continue

            # Age parsing
            elif "age" in key_lower:
                if isinstance(value, str):
                    age_group, min_age, max_age = self._parse_age_value(value)
                    data.age_group = age_group or data.age_group
                    data.min_age_months = min_age or data.min_age_months
                    data.max_age_months = max_age or data.max_age_months
                    extraction_count += 1
                elif isinstance(value, dict):
                    # Nested age object
                    if "min" in value and "max" in value:
                        data.min_age_months = self._convert_age_to_months(value["min"])
                        data.max_age_months = self._convert_age_to_months(value["max"])
                        extraction_count += 1

            # Nutrient mapping
            elif key_lower in self.nutrient_mappings:
                attr_name = self.nutrient_mappings[key_lower]
                try:
                    numeric_value = self._extract_numeric_value(str(value))
                    if numeric_value is not None:
                        setattr(data, attr_name, numeric_value)
                        extraction_count += 1
                except (ValueError, TypeError):
                    continue

            # Vitamins and minerals
            elif isinstance(value, dict):
                if "vitamin" in key_lower or key_lower == "vitamins":
                    data.vitamins.update(self._parse_nutrient_dict(value))
                    extraction_count += 1
                elif "mineral" in key_lower or key_lower == "minerals":
                    data.minerals.update(self._parse_nutrient_dict(value))
                    extraction_count += 1
                elif "portion" in key_lower:
                    data.portion_sizes.update(value)
                    extraction_count += 1

            # Lists
            elif isinstance(value, list):
                if any(keyword in key_lower for keyword in ["recommend", "food"]):
                    data.recommended_foods = value
                    extraction_count += 1
                elif "avoid" in key_lower:
                    data.foods_to_avoid = value
                    extraction_count += 1
                elif any(
                    keyword in key_lower
                    for keyword in ["consideration", "special", "need"]
                ):
                    data.special_considerations = value
                    extraction_count += 1
                elif "deficien" in key_lower:
                    data.common_deficiencies = value
                    extraction_count += 1

            # String lists (comma-separated)
            elif isinstance(value, str):
                if any(keyword in key_lower for keyword in ["recommend", "avoid"]):
                    food_list = self._parse_food_list(value)
                    if "recommend" in key_lower:
                        data.recommended_foods = food_list
                    else:
                        data.foods_to_avoid = food_list
                    extraction_count += 1

        # Set confidence
        data.extraction_confidence = min(extraction_count / 8.0, 1.0)

        return data if self._is_valid_age_data(data) else None

    def _find_column(
        self, df: pd.DataFrame, possible_names: List[str]
    ) -> Optional[str]:
        """Find a column that matches possible names"""
        df_columns_lower = [col.lower() for col in df.columns]

        for name in possible_names:
            if name in df_columns_lower:
                return df.columns[df_columns_lower.index(name)]

            # Partial match
            for col in df.columns:
                if name in col.lower():
                    return col

        return None

    def _parse_age_value(
        self, age_str: str
    ) -> tuple[Optional[str], Optional[int], Optional[int]]:
        """Parse age string and return age_group, min_months, max_months"""
        age_str = age_str.lower().strip()

        # Check predefined age groups
        for group, ranges in self.age_group_mappings.items():
            if any(keyword in age_str for keyword in group.split("_")):
                return group, ranges["min_months"], ranges["max_months"]

        # Parse ranges like "6-12 months" or "2-3 years"
        range_match = re.match(r"(\d+)[-–](\d+)\s*(month|year|y|m)", age_str)
        if range_match:
            min_val, max_val, unit = range_match.groups()
            multiplier = 12 if unit.startswith("y") else 1
            return age_str, int(min_val) * multiplier, int(max_val) * multiplier

        # Parse single values like "6 months" or "2 years"
        single_match = re.match(r"(\d+)\s*(month|year|y|m)", age_str)
        if single_match:
            val, unit = single_match.groups()
            multiplier = 12 if unit.startswith("y") else 1
            months = int(val) * multiplier
            return age_str, months, months

        return age_str, None, None

    def _convert_age_to_months(
        self, age_value: Union[str, int, float]
    ) -> Optional[int]:
        """Convert age value to months"""
        if isinstance(age_value, (int, float)):
            return int(age_value)  # Assume already in months

        if isinstance(age_value, str):
            age_str = age_value.lower().strip()

            # Extract number and unit
            match = re.match(r"(\d+(?:\.\d+)?)\s*(month|year|y|m)", age_str)
            if match:
                val, unit = match.groups()
                multiplier = 12 if unit.startswith("y") else 1
                return int(float(val) * multiplier)

        return None

    def _normalize_gender(self, gender_str: str) -> str:
        """Normalize gender string"""
        gender_lower = gender_str.lower()

        if gender_lower in ["m", "male", "men", "boy"]:
            return "male"
        elif gender_lower in ["f", "female", "women", "girl"]:
            return "female"
        else:
            return "all"

    def _extract_numeric_value(self, value_str: str) -> Optional[float]:
        """Extract numeric value from string"""
        # Remove common non-numeric characters
        cleaned = re.sub(r"[^\d.-]", "", value_str.strip())

        try:
            return float(cleaned)
        except ValueError:
            return None

    def _parse_food_list(self, text: str) -> List[str]:
        """Parse comma-separated food list"""
        if not text or text.lower() == "nan":
            return []

        foods = [food.strip() for food in text.split(",")]
        return [food for food in foods if food and len(food) > 1]

    def _parse_consideration_list(self, text: str) -> List[str]:
        """Parse consideration list from text"""
        if not text or text.lower() == "nan":
            return []

        # Try to split by common delimiters
        for delimiter in [";", ",", "•", "-"]:
            if delimiter in text:
                items = [item.strip() for item in text.split(delimiter)]
                return [item for item in items if item and len(item) > 3]

        # Return as single item if no delimiters found
        return [text.strip()]

    def _parse_nutrient_dict(self, nutrient_data: Dict[str, Any]) -> Dict[str, float]:
        """Parse nutrient dictionary and extract numeric values"""
        parsed = {}

        for key, value in nutrient_data.items():
            numeric_value = self._extract_numeric_value(str(value))
            if numeric_value is not None:
                # Clean up key name
                clean_key = key.replace("_", " ").title()
                parsed[clean_key] = numeric_value

        return parsed

    def _is_single_age_record(self, data: Dict[str, Any]) -> bool:
        """Check if JSON object represents a single age record"""
        age_indicators = ["age", "age_group", "life_stage", "calories", "protein"]
        return any(key.lower() in " ".join(age_indicators) for key in data.keys())

    def _is_valid_age_data(self, data: AgeSpecificData) -> bool:
        """Check if the extracted data contains meaningful age-specific information"""
        has_age_info = bool(
            data.age_group or data.min_age_months or data.max_age_months
        )
        has_nutrition_data = bool(
            data.calories_per_day
            or data.protein_grams_per_day
            or data.vitamins
            or data.minerals
            or data.recommended_foods
        )

        return has_age_info and has_nutrition_data

    def save_results(self, age_data: List[AgeSpecificData], output_path: str) -> bool:
        """
        Save extracted age-specific data to JSON file

        Args:
            age_data: List of AgeSpecificData objects
            output_path: Path where to save the results

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Convert dataclass objects to dictionaries
            data_dicts = [asdict(data) for data in age_data]

            # Add metadata
            output_data = {
                "metadata": {
                    "extraction_date": datetime.now().isoformat(),
                    "total_entries": len(data_dicts),
                    "parser_version": "1.0.0",
                },
                "age_specific_data": data_dicts,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully saved {len(age_data)} entries to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    parser = DataAgeSpecificParser()

    # Test with sample files
    sample_csv = "sample_age_nutrition.csv"
    sample_json = "sample_age_nutrition.json"

    all_results = []

    # Parse CSV file
    if Path(sample_csv).exists():
        csv_results = parser.parse_csv(sample_csv)
        all_results.extend(csv_results)
        print(f"Extracted {len(csv_results)} age-specific entries from CSV")

    # Parse JSON file
    if Path(sample_json).exists():
        json_results = parser.parse_json(sample_json)
        all_results.extend(json_results)
        print(f"Extracted {len(json_results)} age-specific entries from JSON")

    if all_results:
        print(f"\nTotal extracted: {len(all_results)} age-specific entries")
        for i, data in enumerate(all_results, 1):
            print(f"\n--- Entry {i} ---")
            print(f"Age Group: {data.age_group}")
            print(f"Age Range: {data.min_age_months}-{data.max_age_months} months")
            print(f"Gender: {data.gender}")
            print(f"Calories/day: {data.calories_per_day}")
            print(f"Protein/day: {data.protein_grams_per_day}g")
            print(f"Recommended Foods: {data.recommended_foods[:3]}")  # Show first 3
            print(f"Confidence: {data.extraction_confidence:.2f}")

        # Save results
        output_file = "extracted_age_specific_data.json"
        parser.save_results(all_results, output_file)
    else:
        print("No sample files found. Please provide valid CSV or JSON files.")


import re
