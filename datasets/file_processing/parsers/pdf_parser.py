"""
PDF Parser for Nutritional Information Extraction

This module handles PDF files and extracts nutritional information including:
- Nutrient content (proteins, carbs, fats, vitamins, minerals)
- Food composition data
- Caloric information
- Serving sizes and portions
- Dietary guidelines and recommendations
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pdfplumber
import PyPDF2
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NutritionalData:
    """Data class for structured nutritional information"""

    food_name: str = ""
    calories_per_100g: Optional[float] = None
    macronutrients: Dict[str, float] = None
    micronutrients: Dict[str, float] = None
    vitamins: Dict[str, float] = None
    minerals: Dict[str, float] = None
    serving_size: str = ""
    dietary_fiber: Optional[float] = None
    sugar_content: Optional[float] = None
    sodium_content: Optional[float] = None
    cholesterol: Optional[float] = None
    allergens: List[str] = None
    dietary_tags: List[str] = None  # vegetarian, vegan, gluten-free, etc.
    source_document: str = ""
    extraction_confidence: float = 0.0

    def __post_init__(self):
        """Initialize mutable default values"""
        if self.macronutrients is None:
            self.macronutrients = {}
        if self.micronutrients is None:
            self.micronutrients = {}
        if self.vitamins is None:
            self.vitamins = {}
        if self.minerals is None:
            self.minerals = {}
        if self.allergens is None:
            self.allergens = []
        if self.dietary_tags is None:
            self.dietary_tags = []


class PDFNutritionalParser:
    """Main class for parsing nutritional information from PDF files"""

    def __init__(self):
        self.nutritional_patterns = self._initialize_patterns()
        self.unit_conversions = self._initialize_unit_conversions()

    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for nutritional data extraction"""
        patterns = {
            # Macronutrients
            "protein": re.compile(
                r"protein[:\s]*(\d+(?:\.\d+)?)\s*([gmμ%]+)", re.IGNORECASE
            ),
            "carbs": re.compile(
                r"carbohydrate[s]?[:\s]*(\d+(?:\.\d+)?)\s*([gmμ%]+)", re.IGNORECASE
            ),
            "fat": re.compile(
                r"(?:total\s+)?fat[:\s]*(\d+(?:\.\d+)?)\s*([gmμ%]+)", re.IGNORECASE
            ),
            "calories": re.compile(
                r"calor(?:ies?|ic)[:\s]*(\d+(?:\.\d+)?)\s*(?:kcal|cal)?", re.IGNORECASE
            ),
            # Micronutrients
            "fiber": re.compile(
                r"(?:dietary\s+)?fiber[:\s]*(\d+(?:\.\d+)?)\s*([gmμ%]+)", re.IGNORECASE
            ),
            "sugar": re.compile(
                r"sugar[s]?[:\s]*(\d+(?:\.\d+)?)\s*([gmμ%]+)", re.IGNORECASE
            ),
            "sodium": re.compile(
                r"sodium[:\s]*(\d+(?:\.\d+)?)\s*([mgμ%]+)", re.IGNORECASE
            ),
            "cholesterol": re.compile(
                r"cholesterol[:\s]*(\d+(?:\.\d+)?)\s*([mgμ%]+)", re.IGNORECASE
            ),
            # Vitamins
            "vitamin_a": re.compile(
                r"vitamin\s+a[:\s]*(\d+(?:\.\d+)?)\s*([iuμg%]+)", re.IGNORECASE
            ),
            "vitamin_c": re.compile(
                r"vitamin\s+c[:\s]*(\d+(?:\.\d+)?)\s*([mgμg%]+)", re.IGNORECASE
            ),
            "vitamin_d": re.compile(
                r"vitamin\s+d[:\s]*(\d+(?:\.\d+)?)\s*([iuμg%]+)", re.IGNORECASE
            ),
            "vitamin_b12": re.compile(
                r"vitamin\s+b12[:\s]*(\d+(?:\.\d+)?)\s*([μgmg%]+)", re.IGNORECASE
            ),
            # Minerals
            "calcium": re.compile(
                r"calcium[:\s]*(\d+(?:\.\d+)?)\s*([mgμg%]+)", re.IGNORECASE
            ),
            "iron": re.compile(
                r"iron[:\s]*(\d+(?:\.\d+)?)\s*([mgμg%]+)", re.IGNORECASE
            ),
            "potassium": re.compile(
                r"potassium[:\s]*(\d+(?:\.\d+)?)\s*([mgμg%]+)", re.IGNORECASE
            ),
            "zinc": re.compile(
                r"zinc[:\s]*(\d+(?:\.\d+)?)\s*([mgμg%]+)", re.IGNORECASE
            ),
            # Food identification
            "food_name": re.compile(
                r"(?:food|item|product)[:\s]*([a-zA-Z\s,]+)", re.IGNORECASE
            ),
            "serving_size": re.compile(
                r"serving\s+size[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z\s]+)", re.IGNORECASE
            ),
            # Allergens
            "allergens": re.compile(
                r"(?:contains|allergens?)[:\s]*([a-zA-Z\s,]+)", re.IGNORECASE
            ),
        }
        return patterns

    def _initialize_unit_conversions(self) -> Dict[str, float]:
        """Initialize unit conversion factors to standardize measurements"""
        return {
            # Weight conversions to grams
            "kg": 1000.0,
            "g": 1.0,
            "mg": 0.001,
            "μg": 0.000001,
            "mcg": 0.000001,
            # Volume conversions to ml
            "l": 1000.0,
            "ml": 1.0,
            "cup": 240.0,
            "tbsp": 15.0,
            "tsp": 5.0,
            # Energy
            "kcal": 1.0,
            "cal": 0.001,
            "kj": 0.239,  # kilojoule to kcal
        }

    def parse_pdf(self, file_path: str) -> List[NutritionalData]:
        """
        Parse a PDF file and extract nutritional information

        Args:
            file_path: Path to the PDF file

        Returns:
            List of NutritionalData objects containing extracted information
        """
        logger.info(f"Starting PDF parsing for: {file_path}")

        try:
            # Try pdfplumber first (better for structured data)
            nutritional_data = self._parse_with_pdfplumber(file_path)

            if not nutritional_data:
                # Fallback to PyPDF2
                logger.info("Pdfplumber extraction failed, trying PyPDF2...")
                nutritional_data = self._parse_with_pypdf2(file_path)

            logger.info(f"Extracted {len(nutritional_data)} nutritional entries")
            return nutritional_data

        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            return []

    def _parse_with_pdfplumber(self, file_path: str) -> List[NutritionalData]:
        """Parse PDF using pdfplumber for better table extraction"""
        nutritional_data = []

        try:
            with pdfplumber.open(file_path) as pdf:
                all_text = ""
                tables_data = []

                for page in pdf.pages:
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        all_text += page_text + "\n"

                    # Extract tables
                    tables = page.extract_tables()
                    tables_data.extend(tables)

                # Process text-based extraction
                text_data = self._extract_from_text(all_text, file_path)
                nutritional_data.extend(text_data)

                # Process table-based extraction
                table_data = self._extract_from_tables(tables_data, file_path)
                nutritional_data.extend(table_data)

        except Exception as e:
            logger.error(f"PDFPlumber parsing error: {str(e)}")

        return nutritional_data

    def _parse_with_pypdf2(self, file_path: str) -> List[NutritionalData]:
        """Parse PDF using PyPDF2 as fallback"""
        nutritional_data = []

        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                all_text = ""

                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        all_text += page_text + "\n"

                # Extract nutritional information from text
                text_data = self._extract_from_text(all_text, file_path)
                nutritional_data.extend(text_data)

        except Exception as e:
            logger.error(f"PyPDF2 parsing error: {str(e)}")

        return nutritional_data

    def _extract_from_text(self, text: str, source_file: str) -> List[NutritionalData]:
        """Extract nutritional information from plain text"""
        logger.info("Extracting nutritional data from text content")

        # Split text into potential food entries
        sections = self._split_into_sections(text)
        nutritional_data = []

        for section in sections:
            data = self._extract_nutritional_values(section, source_file)
            if data and self._is_valid_nutritional_data(data):
                nutritional_data.append(data)

        return nutritional_data

    def _extract_from_tables(
        self, tables: List[List[List[str]]], source_file: str
    ) -> List[NutritionalData]:
        """Extract nutritional information from table structures"""
        logger.info(f"Processing {len(tables)} tables for nutritional data")

        nutritional_data = []

        for table in tables:
            if not table or len(table) < 2:  # Need at least header and one data row
                continue

            # Try to identify nutritional facts tables
            header_row = [cell.lower() if cell else "" for cell in table[0]]

            # Check if this looks like a nutrition facts table
            if self._is_nutrition_table(header_row):
                table_data = self._parse_nutrition_table(table, source_file)
                nutritional_data.extend(table_data)

        return nutritional_data

    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into sections that might contain individual food items"""
        # Split by common delimiters in nutritional documents
        patterns = [
            r"\n\s*\n",  # Double newlines
            r"(?=\bNutrition\s+Facts\b)",  # Before "Nutrition Facts"
            r"(?=\bServing\s+Size\b)",  # Before "Serving Size"
            r"(?=\bCalories\b)",  # Before "Calories"
        ]

        sections = [text]
        for pattern in patterns:
            new_sections = []
            for section in sections:
                new_sections.extend(re.split(pattern, section, flags=re.IGNORECASE))
            sections = [s.strip() for s in new_sections if s.strip()]

        return sections

    def _extract_nutritional_values(
        self, text: str, source_file: str
    ) -> Optional[NutritionalData]:
        """Extract nutritional values from a text section"""
        data = NutritionalData(source_document=source_file)
        extraction_count = 0

        # Extract each type of nutritional information
        for nutrient_type, pattern in self.nutritional_patterns.items():
            matches = pattern.findall(text)

            if matches:
                extraction_count += 1
                value, unit = (
                    matches[0] if isinstance(matches[0], tuple) else (matches[0], "")
                )

                try:
                    numeric_value = float(value)
                    standardized_value = self._standardize_unit(
                        numeric_value, unit.lower()
                    )

                    # Categorize the nutrient
                    if nutrient_type in ["protein", "carbs", "fat"]:
                        data.macronutrients[nutrient_type] = standardized_value
                    elif nutrient_type == "calories":
                        data.calories_per_100g = standardized_value
                    elif nutrient_type in ["fiber", "sugar", "sodium", "cholesterol"]:
                        setattr(
                            data,
                            (
                                f"{nutrient_type}_content"
                                if nutrient_type != "fiber"
                                else "dietary_fiber"
                            ),
                            standardized_value,
                        )
                    elif nutrient_type.startswith("vitamin_"):
                        vitamin_name = nutrient_type.replace("_", " ").title()
                        data.vitamins[vitamin_name] = standardized_value
                    elif nutrient_type in ["calcium", "iron", "potassium", "zinc"]:
                        data.minerals[nutrient_type.title()] = standardized_value
                    elif nutrient_type == "food_name":
                        data.food_name = value.strip()
                    elif nutrient_type == "serving_size":
                        data.serving_size = f"{value} {unit}".strip()
                    elif nutrient_type == "allergens":
                        data.allergens = [
                            allergen.strip() for allergen in value.split(",")
                        ]

                except ValueError:
                    logger.warning(
                        f"Could not convert {value} to numeric value for {nutrient_type}"
                    )

        # Calculate extraction confidence
        data.extraction_confidence = min(
            extraction_count / 10.0, 1.0
        )  # Max confidence at 10+ extractions

        return data if extraction_count > 0 else None

    def _is_nutrition_table(self, header_row: List[str]) -> bool:
        """Check if a table appears to contain nutritional information"""
        nutrition_indicators = [
            "nutrient",
            "calories",
            "protein",
            "fat",
            "carbohydrate",
            "vitamin",
            "mineral",
            "serving",
            "amount",
            "value",
        ]

        header_text = " ".join(header_row).lower()
        return any(indicator in header_text for indicator in nutrition_indicators)

    def _parse_nutrition_table(
        self, table: List[List[str]], source_file: str
    ) -> List[NutritionalData]:
        """Parse a nutrition facts table"""
        nutritional_data = []

        if len(table) < 2:
            return nutritional_data

        headers = [cell.lower().strip() if cell else "" for cell in table[0]]

        # Try to identify column indices
        nutrient_col = self._find_column_index(
            headers, ["nutrient", "component", "item"]
        )
        value_col = self._find_column_index(headers, ["amount", "value", "quantity"])
        unit_col = self._find_column_index(headers, ["unit", "units"])

        if nutrient_col is None or value_col is None:
            return nutritional_data

        # Process each data row
        for row in table[1:]:
            if len(row) <= max(nutrient_col, value_col):
                continue

            nutrient = row[nutrient_col].strip() if row[nutrient_col] else ""
            value_str = row[value_col].strip() if row[value_col] else ""
            unit = (
                row[unit_col].strip()
                if unit_col is not None and len(row) > unit_col and row[unit_col]
                else ""
            )

            if not nutrient or not value_str:
                continue

            try:
                value = float(re.sub(r"[^\d.]", "", value_str))

                # Create or update nutritional data entry
                data = NutritionalData(
                    food_name=f"Item from {Path(source_file).stem}",
                    source_document=source_file,
                    extraction_confidence=0.8,  # High confidence for table data
                )

                # Categorize the nutrient
                nutrient_lower = nutrient.lower()
                if "protein" in nutrient_lower:
                    data.macronutrients["protein"] = self._standardize_unit(value, unit)
                elif "carb" in nutrient_lower:
                    data.macronutrients["carbs"] = self._standardize_unit(value, unit)
                elif "fat" in nutrient_lower:
                    data.macronutrients["fat"] = self._standardize_unit(value, unit)
                elif "calor" in nutrient_lower:
                    data.calories_per_100g = value
                # Add more categorizations as needed

                if self._is_valid_nutritional_data(data):
                    nutritional_data.append(data)

            except ValueError:
                logger.warning(
                    f"Could not parse value '{value_str}' for nutrient '{nutrient}'"
                )

        return nutritional_data

    def _find_column_index(
        self, headers: List[str], possible_names: List[str]
    ) -> Optional[int]:
        """Find the index of a column based on possible header names"""
        for i, header in enumerate(headers):
            for name in possible_names:
                if name in header:
                    return i
        return None

    def _standardize_unit(self, value: float, unit: str) -> float:
        """Standardize units to common measurements"""
        unit_clean = re.sub(r"[^\w]", "", unit.lower())

        if unit_clean in self.unit_conversions:
            return value * self.unit_conversions[unit_clean]

        return value  # Return as-is if unit not recognized

    def _is_valid_nutritional_data(self, data: NutritionalData) -> bool:
        """Check if the extracted data contains meaningful nutritional information"""
        has_macronutrients = bool(data.macronutrients)
        has_calories = data.calories_per_100g is not None
        has_micronutrients = bool(data.vitamins or data.minerals)
        has_basic_info = bool(
            data.dietary_fiber or data.sugar_content or data.sodium_content
        )

        return (
            has_macronutrients or has_calories or has_micronutrients or has_basic_info
        )

    def save_results(
        self, nutritional_data: List[NutritionalData], output_path: str
    ) -> bool:
        """
        Save extracted nutritional data to JSON file

        Args:
            nutritional_data: List of NutritionalData objects
            output_path: Path where to save the results

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Convert dataclass objects to dictionaries
            data_dicts = [asdict(data) for data in nutritional_data]

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data_dicts, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Successfully saved {len(nutritional_data)} entries to {output_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    parser = PDFNutritionalParser()

    # Test with a sample PDF file
    sample_pdf = "sample_nutrition_facts.pdf"

    if Path(sample_pdf).exists():
        results = parser.parse_pdf(sample_pdf)

        print(f"Extracted {len(results)} nutritional entries:")
        for i, data in enumerate(results, 1):
            print(f"\n--- Entry {i} ---")
            print(f"Food: {data.food_name}")
            print(f"Calories: {data.calories_per_100g}")
            print(f"Macronutrients: {data.macronutrients}")
            print(f"Vitamins: {data.vitamins}")
            print(f"Minerals: {data.minerals}")
            print(f"Confidence: {data.extraction_confidence:.2f}")

        # Save results
        output_file = "extracted_nutritional_data.json"
        parser.save_results(results, output_file)
    else:
        print(f"Sample file {sample_pdf} not found. Please provide a valid PDF file.")
