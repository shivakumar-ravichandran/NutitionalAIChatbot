"""
Configuration Module for File Processing

This module contains configuration settings, constants, and default values
used across the file processing system.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    """Configuration class for file processing settings"""

    # Directory settings
    output_dir: str = "outputs"
    temp_dir: str = "temp"
    log_dir: str = "logs"

    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    console_logging: bool = True

    # Processing settings
    max_file_size_mb: int = 100
    parallel_processing: bool = False
    max_workers: int = 4

    # Parser-specific settings
    pdf_extraction_method: str = "auto"  # "auto", "pdfplumber", "pypdf2"
    text_encoding: str = "utf-8"
    csv_delimiter: str = "auto"  # "auto", ",", ";", "\t"

    # Quality thresholds
    min_confidence_score: float = 0.3
    min_extraction_fields: int = 2

    # Output settings
    save_individual_files: bool = True
    save_processing_stats: bool = True
    output_format: str = "json"  # "json", "csv", "excel"

    # Error handling
    continue_on_error: bool = True
    max_retries: int = 2

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate_config()

    def validate_config(self):
        """Validate configuration values"""
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")

        if self.min_confidence_score < 0 or self.min_confidence_score > 1:
            raise ValueError("min_confidence_score must be between 0 and 1")

        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")

        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid log_level")

        if self.output_format not in ["json", "csv", "excel"]:
            raise ValueError("Invalid output_format")


# Default configuration instance
DEFAULT_CONFIG = ProcessingConfig()


# File type configurations
FILE_TYPE_CONFIG = {
    "pdf": {
        "max_size_mb": 50,
        "supported_versions": ["1.4", "1.5", "1.6", "1.7", "2.0"],
        "extraction_methods": ["pdfplumber", "pypdf2"],
        "timeout_seconds": 60,
    },
    "docx": {
        "max_size_mb": 25,
        "supported_formats": [".docx", ".doc"],
        "timeout_seconds": 30,
    },
    "markdown": {
        "max_size_mb": 10,
        "supported_formats": [".md", ".markdown"],
        "encoding": "utf-8",
        "timeout_seconds": 15,
    },
    "csv": {
        "max_size_mb": 100,
        "supported_delimiters": [",", ";", "\t", "|"],
        "encoding_fallbacks": ["utf-8", "utf-8-sig", "latin-1", "cp1252"],
        "timeout_seconds": 45,
    },
    "json": {
        "max_size_mb": 50,
        "max_nesting_depth": 10,
        "encoding": "utf-8",
        "timeout_seconds": 30,
    },
}


# Nutritional data extraction patterns
NUTRITIONAL_PATTERNS = {
    "nutrients": {
        "protein": [
            r"protein[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)",
            r"total protein[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)",
        ],
        "carbohydrates": [
            r"carbohydrate[s]?[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)",
            r"total carb[s]?[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)",
        ],
        "fat": [
            r"(?:total\s+)?fat[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)",
            r"lipid[s]?[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)",
        ],
        "calories": [
            r"calor(?:ies?|ic)[:\s]*(\d+(?:\.\d+)?)",
            r"energy[:\s]*(\d+(?:\.\d+)?)\s*(?:kcal|cal)",
        ],
        "fiber": [
            r"(?:dietary\s+)?fiber[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)",
            r"fibre[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)",
        ],
    },
    "vitamins": {
        "vitamin_a": [r"vitamin\s+a[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)"],
        "vitamin_c": [r"vitamin\s+c[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)"],
        "vitamin_d": [r"vitamin\s+d[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)"],
        "vitamin_b12": [r"vitamin\s+b12[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)"],
    },
    "minerals": {
        "calcium": [r"calcium[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)"],
        "iron": [r"iron[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)"],
        "sodium": [r"sodium[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)"],
        "potassium": [r"potassium[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z%]*)"],
    },
}


# Cultural data extraction patterns
CULTURAL_PATTERNS = {
    "identifiers": {
        "culture_name": [
            r"(?:culture|tradition|community|people)[:\s]*([A-Za-z\s,]+)",
            r"([A-Za-z]+)\s+(?:culture|tradition|community)",
        ],
        "region": [
            r"(?:region|area|country|state|province)[:\s]*([A-Za-z\s,]+)",
            r"from\s+([A-Za-z\s,]+)\s+region",
        ],
    },
    "foods": {
        "traditional_foods": [
            r"(?:traditional|staple|native|indigenous)\s+(?:food|dish|meal)[s]?[:\s]*([^.!?]*)",
            r"(?:eat|consume)\s+([^.!?]*?)\s+(?:traditionally|customarily)",
        ],
        "festival_foods": [
            r"(?:festival|celebration|ceremony)[:\s]*([^.!?]*)",
            r"during\s+([A-Za-z\s]+)\s+(?:festival|celebration)[,\s]*(?:eat|serve|prepare)\s+([^.!?]*)",
        ],
    },
    "practices": {
        "dietary_restrictions": [
            r"(?:forbidden|prohibited|taboo|avoid|restrict)[s]?[:\s]*([^.!?]*)",
            r"(?:cannot|must not|should not)\s+(?:eat|consume)\s+([^.!?]*)",
        ],
        "cooking_methods": [
            r"(?:cook|prepare|method|technique)[s]?[:\s]*([^.!?]*)",
            r"(?:traditionally|usually)\s+(?:cooked|prepared)\s+(?:by|using|with)\s+([^.!?]*)",
        ],
    },
}


# Age-specific data patterns
AGE_SPECIFIC_PATTERNS = {
    "age_ranges": {
        "infant": {
            "patterns": [
                r"infant[s]?",
                r"baby",
                r"newborn",
                r"0[-–]?6\s*months?",
                r"6[-–]?12\s*months?",
            ],
            "min_months": 0,
            "max_months": 12,
        },
        "toddler": {
            "patterns": [r"toddler[s]?", r"1[-–]?3\s*years?", r"12[-–]?36\s*months?"],
            "min_months": 12,
            "max_months": 36,
        },
        "preschool": {
            "patterns": [r"preschool(?:er)?[s]?", r"3[-–]?6\s*years?"],
            "min_months": 36,
            "max_months": 72,
        },
        "school_age": {
            "patterns": [r"school\s*age", r"6[-–]?12\s*years?"],
            "min_months": 72,
            "max_months": 144,
        },
        "teenager": {
            "patterns": [
                r"teen(?:ager)?[s]?",
                r"adolescent[s]?",
                r"12[-–]?18\s*years?",
            ],
            "min_months": 144,
            "max_months": 216,
        },
        "adult": {
            "patterns": [r"adult[s]?", r"18[-–]?65\s*years?", r"grown[-\s]?up[s]?"],
            "min_months": 216,
            "max_months": 780,
        },
        "senior": {
            "patterns": [r"senior[s]?", r"elderly", r"65\+", r"older\s+adult[s]?"],
            "min_months": 780,
            "max_months": None,
        },
    },
    "nutritional_requirements": {
        "energy_patterns": [
            r"(?:energy|calor(?:ies?|ic))[:\s]*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:kcal|cal)(?:ories?)?",
        ],
        "protein_patterns": [
            r"protein[:\s]*(\d+(?:\.\d+)?)\s*(?:g|grams?)?",
            r"(\d+(?:\.\d+)?)\s*g(?:rams?)?\s+protein",
        ],
        "recommendation_patterns": [
            r"recommend(?:ed)?[:\s]*([^.!?]*)",
            r"should\s+(?:eat|consume|have)[:\s]*([^.!?]*)",
        ],
    },
}


# Unit conversion mappings
UNIT_CONVERSIONS = {
    "weight": {
        "kg": 1000.0,  # to grams
        "g": 1.0,
        "mg": 0.001,
        "μg": 0.000001,
        "mcg": 0.000001,
        "oz": 28.3495,
        "lb": 453.592,
    },
    "volume": {
        "l": 1000.0,  # to ml
        "ml": 1.0,
        "cup": 240.0,
        "tbsp": 15.0,
        "tsp": 5.0,
        "fl oz": 29.5735,
        "pt": 473.176,
        "qt": 946.353,
        "gal": 3785.41,
    },
    "energy": {"kcal": 1.0, "cal": 0.001, "kj": 0.239006, "j": 0.000239006},  # to kcal
}


# Output templates
OUTPUT_TEMPLATES = {
    "integrated_data": {
        "metadata": {
            "version": "1.0.0",
            "processing_date": None,  # Will be filled at runtime
            "description": "Integrated nutritional knowledge base",
            "data_sources": {
                "nutritional_data": "PDF files - nutrition facts and food composition",
                "cultural_data": "MD/DOCX files - cultural dietary practices and traditions",
                "age_specific_data": "CSV/JSON files - age-specific nutritional requirements",
            },
        },
        "processing_summary": {},
        "nutritional_data": [],
        "cultural_data": [],
        "age_specific_data": [],
        "processing_errors": [],
    },
    "processing_summary": {
        "total_files": 0,
        "processed_successfully": 0,
        "failed_files": 0,
        "nutritional_entries": 0,
        "cultural_entries": 0,
        "age_specific_entries": 0,
        "processing_date": None,
        "supported_formats": [],
    },
}


# Quality assessment criteria
QUALITY_CRITERIA = {
    "nutritional_data": {
        "required_fields": ["food_name", "calories_per_100g"],
        "optional_fields": ["macronutrients", "vitamins", "minerals"],
        "min_confidence": 0.4,
        "quality_indicators": {
            "has_serving_size": True,
            "has_multiple_nutrients": True,
            "has_source_info": True,
        },
    },
    "cultural_data": {
        "required_fields": ["culture_name", "traditional_foods"],
        "optional_fields": ["dietary_practices", "festive_foods", "cooking_methods"],
        "min_confidence": 0.3,
        "quality_indicators": {
            "has_region_info": True,
            "has_practices": True,
            "has_detailed_descriptions": True,
        },
    },
    "age_specific_data": {
        "required_fields": ["age_group", "calories_per_day"],
        "optional_fields": ["protein_grams_per_day", "vitamins", "minerals"],
        "min_confidence": 0.5,
        "quality_indicators": {
            "has_age_range": True,
            "has_multiple_nutrients": True,
            "has_recommendations": True,
        },
    },
}


# Environment variables with defaults
def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables with defaults"""
    return {
        "OUTPUT_DIR": os.getenv("FILE_PROCESSING_OUTPUT_DIR", "outputs"),
        "LOG_LEVEL": os.getenv("FILE_PROCESSING_LOG_LEVEL", "INFO"),
        "MAX_FILE_SIZE_MB": int(os.getenv("FILE_PROCESSING_MAX_FILE_SIZE_MB", "100")),
        "PARALLEL_PROCESSING": os.getenv("FILE_PROCESSING_PARALLEL", "false").lower()
        == "true",
        "MAX_WORKERS": int(os.getenv("FILE_PROCESSING_MAX_WORKERS", "4")),
        "CONTINUE_ON_ERROR": os.getenv(
            "FILE_PROCESSING_CONTINUE_ON_ERROR", "true"
        ).lower()
        == "true",
        "MIN_CONFIDENCE_SCORE": float(
            os.getenv("FILE_PROCESSING_MIN_CONFIDENCE", "0.3")
        ),
    }


def create_config_from_env() -> ProcessingConfig:
    """Create ProcessingConfig from environment variables"""
    env_config = get_env_config()

    return ProcessingConfig(
        output_dir=env_config["OUTPUT_DIR"],
        log_level=env_config["LOG_LEVEL"],
        max_file_size_mb=env_config["MAX_FILE_SIZE_MB"],
        parallel_processing=env_config["PARALLEL_PROCESSING"],
        max_workers=env_config["MAX_WORKERS"],
        continue_on_error=env_config["CONTINUE_ON_ERROR"],
        min_confidence_score=env_config["MIN_CONFIDENCE_SCORE"],
    )


def save_config_to_file(config: ProcessingConfig, file_path: str) -> bool:
    """Save configuration to JSON file"""
    try:
        import json
        from dataclasses import asdict

        config_dict = asdict(config)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        return True
    except Exception:
        return False


def load_config_from_file(file_path: str) -> Optional[ProcessingConfig]:
    """Load configuration from JSON file"""
    try:
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        return ProcessingConfig(**config_dict)
    except Exception:
        return None


# Application constants
APP_NAME = "Nutritional AI File Processing Module"
VERSION = "1.0.0"
AUTHOR = "Nutritional AI Team"
DESCRIPTION = "Comprehensive file processing module for extracting nutritional, cultural, and age-specific dietary information"

# Supported file extensions and MIME types
SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
}

# Default file patterns for directory processing
DEFAULT_FILE_PATTERNS = ["*.pdf", "*.docx", "*.md", "*.csv", "*.json"]
