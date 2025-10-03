"""
Utility Functions for File Processing Module

This module contains common utility functions used across all parsers including:
- Text cleaning and normalization
- File validation and handling
- Data type conversions
- Logging helpers
- Configuration management
"""

import re
import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import unicodedata


def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def clean_text(
    text: str, remove_extra_whitespace: bool = True, normalize_unicode: bool = True
) -> str:
    """
    Clean and normalize text content

    Args:
        text: Input text to clean
        remove_extra_whitespace: Whether to remove extra whitespace
        normalize_unicode: Whether to normalize unicode characters

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Normalize unicode if requested
    if normalize_unicode:
        text = unicodedata.normalize("NFKD", text)

    # Remove control characters except newlines and tabs
    text = "".join(
        char
        for char in text
        if unicodedata.category(char)[0] != "C" or char in "\n\t\r"
    )

    # Remove extra whitespace if requested
    if remove_extra_whitespace:
        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        text = text.strip()

    return text


def extract_numbers_from_text(text: str) -> List[float]:
    """
    Extract all numeric values from text

    Args:
        text: Input text

    Returns:
        List of extracted numeric values
    """
    # Pattern to match numbers (including decimals, negatives, and scientific notation)
    number_pattern = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    matches = re.findall(number_pattern, text)

    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue

    return numbers


def normalize_unit(unit: str) -> str:
    """
    Normalize measurement units to standard forms

    Args:
        unit: Input unit string

    Returns:
        Normalized unit string
    """
    if not unit:
        return ""

    unit_mappings = {
        # Weight/mass
        "grams": "g",
        "gram": "g",
        "milligrams": "mg",
        "milligram": "mg",
        "kilograms": "kg",
        "kilogram": "kg",
        "micrograms": "μg",
        "microgram": "μg",
        "mcg": "μg",
        # Volume
        "liters": "l",
        "liter": "l",
        "litres": "l",
        "litre": "l",
        "milliliters": "ml",
        "milliliter": "ml",
        "millilitres": "ml",
        "millilitre": "ml",
        "cups": "cup",
        "tablespoons": "tbsp",
        "tablespoon": "tbsp",
        "teaspoons": "tsp",
        "teaspoon": "tsp",
        # Energy
        "calories": "cal",
        "calorie": "cal",
        "kilocalories": "kcal",
        "kilocalorie": "kcal",
        "kilojoules": "kj",
        "kilojoule": "kj",
        # Other units
        "international units": "iu",
        "international unit": "iu",
        "percent": "%",
        "percentage": "%",
    }

    unit_clean = unit.lower().strip()
    return unit_mappings.get(unit_clean, unit_clean)


def validate_file_path(
    file_path: Union[str, Path], expected_extensions: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Validate file path and check if file exists and has expected extension

    Args:
        file_path: Path to validate
        expected_extensions: List of expected file extensions (e.g., ['.pdf', '.csv'])

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            return False, f"File does not exist: {file_path}"

        # Check if it's a file (not directory)
        if not path.is_file():
            return False, f"Path is not a file: {file_path}"

        # Check file extension if specified
        if expected_extensions:
            file_ext = path.suffix.lower()
            if file_ext not in [ext.lower() for ext in expected_extensions]:
                return (
                    False,
                    f"Unexpected file extension: {file_ext}. Expected: {expected_extensions}",
                )

        # Check if file is readable
        if not os.access(path, os.R_OK):
            return False, f"File is not readable: {file_path}"

        # Check if file is not empty
        if path.stat().st_size == 0:
            return False, f"File is empty: {file_path}"

        return True, ""

    except Exception as e:
        return False, f"Error validating file: {str(e)}"


def safe_cast_to_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Safely cast a value to float, returning default if conversion fails

    Args:
        value: Value to convert
        default: Default value to return if conversion fails

    Returns:
        Float value or default
    """
    if value is None or value == "":
        return default

    try:
        # Handle string values
        if isinstance(value, str):
            # Remove common non-numeric characters
            cleaned = re.sub(r"[^\d.-]", "", value.strip())
            if not cleaned:
                return default
            return float(cleaned)

        # Handle numeric values
        return float(value)

    except (ValueError, TypeError):
        return default


def parse_age_range(age_text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse age range text and return min/max ages in months

    Args:
        age_text: Text containing age information

    Returns:
        Tuple of (min_age_months, max_age_months)
    """
    if not age_text:
        return None, None

    age_text = age_text.lower().strip()

    # Handle specific patterns
    patterns = [
        # Range patterns like "6-12 months", "2-3 years"
        (
            r"(\d+)[-–](\d+)\s*(month|year|m|y)",
            lambda m: (
                int(m.group(1)) * (12 if m.group(3).startswith("y") else 1),
                int(m.group(2)) * (12 if m.group(3).startswith("y") else 1),
            ),
        ),
        # Single age patterns like "6 months", "2 years"
        (
            r"(\d+)\s*(month|year|m|y)",
            lambda m: (
                int(m.group(1)) * (12 if m.group(2).startswith("y") else 1),
                int(m.group(1)) * (12 if m.group(2).startswith("y") else 1),
            ),
        ),
        # Plus patterns like "65+"
        (
            r"(\d+)\+",
            lambda m: (int(m.group(1)) * 12, None),  # Assume years for plus notation
        ),
    ]

    for pattern, converter in patterns:
        match = re.search(pattern, age_text)
        if match:
            return converter(match)

    return None, None


def split_list_string(text: str, delimiters: Optional[List[str]] = None) -> List[str]:
    """
    Split a string into a list using various delimiters

    Args:
        text: Input text to split
        delimiters: List of delimiters to use for splitting

    Returns:
        List of cleaned string items
    """
    if not text or text.lower() == "nan":
        return []

    if delimiters is None:
        delimiters = [",", ";", "|", "\n", "•", "-"]

    # Try each delimiter
    for delimiter in delimiters:
        if delimiter in text:
            items = [item.strip() for item in text.split(delimiter)]
            return [item for item in items if item and len(item) > 1]

    # If no delimiters found, return as single item if meaningful
    text = text.strip()
    if text and len(text) > 2:
        return [text]

    return []


def create_output_directory(output_path: Union[str, Path]) -> Path:
    """
    Ensure output directory exists and return Path object

    Args:
        output_path: Output directory path

    Returns:
        Path object for the output directory
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get detailed information about a file

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file information
    """
    path = Path(file_path)

    if not path.exists():
        return {"exists": False, "error": "File does not exist"}

    try:
        stat = path.stat()
        return {
            "exists": True,
            "name": path.name,
            "stem": path.stem,
            "suffix": path.suffix,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": stat.st_mtime,
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
            "parent": str(path.parent),
            "absolute_path": str(path.absolute()),
        }
    except Exception as e:
        return {"exists": True, "error": f"Error getting file info: {str(e)}"}


def calculate_confidence_score(
    extracted_fields: int,
    total_expected_fields: int,
    quality_indicators: Optional[Dict[str, bool]] = None,
) -> float:
    """
    Calculate extraction confidence score based on various factors

    Args:
        extracted_fields: Number of successfully extracted fields
        total_expected_fields: Total number of expected fields
        quality_indicators: Optional dictionary of quality indicators

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if total_expected_fields == 0:
        return 0.0

    # Base score from field extraction ratio
    base_score = min(extracted_fields / total_expected_fields, 1.0)

    # Adjust based on quality indicators
    if quality_indicators:
        adjustments = 0.0
        indicator_count = len(quality_indicators)

        for indicator, is_good in quality_indicators.items():
            if is_good:
                adjustments += 0.1  # Small positive adjustment
            else:
                adjustments -= 0.05  # Small negative adjustment

        # Average the adjustments
        if indicator_count > 0:
            avg_adjustment = adjustments / indicator_count
            base_score = max(0.0, min(1.0, base_score + avg_adjustment))

    return round(base_score, 3)


def merge_dictionaries(
    *dicts: Dict[str, Any], conflict_resolution: str = "last"
) -> Dict[str, Any]:
    """
    Merge multiple dictionaries with conflict resolution

    Args:
        *dicts: Variable number of dictionaries to merge
        conflict_resolution: How to handle conflicts ('first', 'last', 'merge')

    Returns:
        Merged dictionary
    """
    if not dicts:
        return {}

    if len(dicts) == 1:
        return dicts[0].copy()

    result = {}

    for d in dicts:
        if not isinstance(d, dict):
            continue

        for key, value in d.items():
            if key not in result:
                result[key] = value
            else:
                if conflict_resolution == "first":
                    # Keep first value
                    continue
                elif conflict_resolution == "last":
                    # Use last value
                    result[key] = value
                elif conflict_resolution == "merge":
                    # Try to merge if both are dicts or lists
                    existing = result[key]
                    if isinstance(existing, dict) and isinstance(value, dict):
                        result[key] = merge_dictionaries(
                            existing, value, conflict_resolution
                        )
                    elif isinstance(existing, list) and isinstance(value, list):
                        result[key] = existing + value
                    else:
                        result[key] = value

    return result


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math

    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def deduplicate_list(
    items: List[Any], key_func: Optional[callable] = None
) -> List[Any]:
    """
    Remove duplicates from list while preserving order

    Args:
        items: List of items to deduplicate
        key_func: Optional function to extract comparison key from items

    Returns:
        List with duplicates removed
    """
    if not items:
        return []

    seen = set()
    result = []

    for item in items:
        key = key_func(item) if key_func else item

        # Handle unhashable types
        try:
            if key not in seen:
                seen.add(key)
                result.append(item)
        except TypeError:
            # For unhashable types, do linear search
            if key not in [key_func(x) if key_func else x for x in result]:
                result.append(item)

    return result


# Constants for common patterns and mappings
COMMON_FOOD_UNITS = {
    "g",
    "grams",
    "gram",
    "mg",
    "milligrams",
    "milligram",
    "kg",
    "kilograms",
    "kilogram",
    "oz",
    "ounce",
    "ounces",
    "lb",
    "pound",
    "pounds",
    "cup",
    "cups",
    "tbsp",
    "tablespoon",
    "tablespoons",
    "tsp",
    "teaspoon",
    "teaspoons",
    "ml",
    "milliliter",
    "milliliters",
    "l",
    "liter",
    "liters",
    "fl oz",
    "fluid ounce",
}

NUTRIENT_CATEGORIES = {
    "macronutrients": ["protein", "carbohydrates", "carbs", "fat", "lipids"],
    "vitamins": [
        "vitamin a",
        "vitamin c",
        "vitamin d",
        "vitamin e",
        "vitamin k",
        "vitamin b1",
        "vitamin b2",
        "vitamin b3",
        "vitamin b6",
        "vitamin b12",
        "thiamine",
        "riboflavin",
        "niacin",
        "folate",
        "folic acid",
    ],
    "minerals": [
        "calcium",
        "iron",
        "magnesium",
        "phosphorus",
        "potassium",
        "sodium",
        "zinc",
        "copper",
        "manganese",
        "selenium",
    ],
    "other": ["fiber", "sugar", "cholesterol", "caffeine", "alcohol"],
}

AGE_GROUP_KEYWORDS = {
    "infant": ["infant", "baby", "newborn", "0-6 months", "6-12 months"],
    "toddler": ["toddler", "1-3 years", "12-36 months"],
    "preschool": ["preschool", "preschooler", "3-6 years"],
    "school": ["school age", "school-age", "6-12 years"],
    "teen": ["teen", "teenager", "adolescent", "12-18 years"],
    "adult": ["adult", "18-65 years", "grown-up"],
    "senior": ["senior", "elderly", "65+", "older adult"],
}
