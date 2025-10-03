# File Processing Module - Package Initialization

from .main import FileProcessingOrchestrator
from .parsers.pdf_parser import PDFNutritionalParser, NutritionalData
from .parsers.text_parser import TextCulturalParser, CulturalData
from .parsers.data_parser import DataAgeSpecificParser, AgeSpecificData
from .utils.config import ProcessingConfig, DEFAULT_CONFIG
from .utils.utils import setup_logging, clean_text, validate_file_path

__version__ = "1.0.0"
__author__ = "Nutritional AI Team"
__description__ = "Comprehensive file processing module for extracting nutritional, cultural, and age-specific dietary information"

# Package metadata
__all__ = [
    "FileProcessingOrchestrator",
    "PDFNutritionalParser",
    "TextCulturalParser",
    "DataAgeSpecificParser",
    "NutritionalData",
    "CulturalData",
    "AgeSpecificData",
    "ProcessingConfig",
    "DEFAULT_CONFIG",
    "setup_logging",
    "clean_text",
    "validate_file_path",
]
