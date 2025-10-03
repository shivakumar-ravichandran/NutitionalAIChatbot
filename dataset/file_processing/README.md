# Nutritional AI File Processing Module

A comprehensive Python module for extracting structured nutritional information from various file formats including PDF, Markdown, DOCX, CSV, and JSON files.

## Overview

This module is designed to process different types of files and extract three main categories of nutritional information:

- **Nutritional Data** (from PDF files): Nutrition facts, food composition, caloric content, vitamins, minerals
- **Cultural Data** (from MD/DOCX files): Traditional foods, cultural dietary practices, religious restrictions, festival foods
- **Age-Specific Data** (from CSV/JSON files): Age-group nutritional requirements, developmental needs, portion sizes

## Features

- ✅ **Multi-format Support**: PDF, Markdown, DOCX, CSV, JSON
- ✅ **Intelligent Parsing**: Uses regex patterns and machine learning approaches
- ✅ **Structured Output**: Consistent JSON output format for all data types
- ✅ **Quality Assessment**: Confidence scoring for extracted data
- ✅ **Batch Processing**: Process single files or entire directories
- ✅ **Error Handling**: Robust error handling with detailed logging
- ✅ **Extensible**: Easy to add new parsers and data types

## Installation

1. Clone or download the module to your project directory
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Core Dependencies

- `pandas>=1.5.0` - Data manipulation and analysis
- `PyPDF2>=3.0.0` - PDF text extraction
- `pdfplumber>=0.7.0` - Advanced PDF parsing
- `python-docx>=0.8.11` - DOCX file processing
- `markdown>=3.4.0` - Markdown parsing

## Quick Start

### Basic Usage

```python
from file_processing.main import FileProcessingOrchestrator

# Initialize the orchestrator
processor = FileProcessingOrchestrator(output_dir="my_outputs")

# Process a single file
result = processor.process_single_file("nutrition_facts.pdf")

# Process multiple files
files = ["nutrition.pdf", "cultural_info.md", "age_data.csv"]
results = processor.process_files(files)

# Process entire directory
results = processor.process_directory("./input_files", recursive=True)
```

### Command Line Usage

```bash
# Process specific files
python main.py --files nutrition.pdf cultural_info.md age_data.csv

# Process directory
python main.py --directory ./input_files --recursive

# Custom output
python main.py --files *.pdf --output my_nutrition_data.json

# Validate files only
python main.py --files *.pdf --validate-only
```

## Module Structure

```
file_processing/
├── main.py                 # Main orchestrator module
├── parsers/
│   ├── pdf_parser.py       # PDF nutritional data extraction
│   ├── text_parser.py      # MD/DOCX cultural data extraction
│   └── data_parser.py      # CSV/JSON age-specific data extraction
├── utils/
│   ├── utils.py           # Common utility functions
│   └── config.py          # Configuration settings
├── outputs/               # Generated output files
├── examples/              # Example input files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Supported File Types

### PDF Files (Nutritional Data)

- **Purpose**: Extract nutrition facts, food composition data
- **Output**: Food name, calories, macronutrients, vitamins, minerals, serving sizes
- **Example Use**: Nutrition labels, food composition databases, dietary guides

### Markdown Files (Cultural Data)

- **Purpose**: Extract cultural dietary practices and traditional foods
- **Output**: Culture name, traditional foods, dietary practices, festival foods, cooking methods
- **Example Use**: Cultural food guides, traditional recipe collections, dietary restriction documents

### DOCX Files (Cultural Data)

- **Purpose**: Same as Markdown but for Word documents
- **Output**: Same structured cultural information
- **Example Use**: Academic papers, cultural food studies, dietary tradition documents

### CSV Files (Age-Specific Data)

- **Purpose**: Extract age-group nutritional requirements
- **Output**: Age groups, daily nutritional needs, recommended foods, portion sizes
- **Example Use**: Dietary guidelines, pediatric nutrition tables, senior care nutrition data

### JSON Files (Age-Specific Data)

- **Purpose**: Same as CSV but for structured JSON data
- **Output**: Same age-specific nutritional information
- **Example Use**: API responses, structured nutrition databases, healthcare applications

## Data Models

### NutritionalData (PDF Output)

```python
@dataclass
class NutritionalData:
    food_name: str
    calories_per_100g: Optional[float]
    macronutrients: Dict[str, float]  # protein, carbs, fat
    vitamins: Dict[str, float]        # vitamin A, C, D, etc.
    minerals: Dict[str, float]        # calcium, iron, etc.
    serving_size: str
    allergens: List[str]
    dietary_tags: List[str]           # vegetarian, vegan, etc.
    extraction_confidence: float
```

### CulturalData (MD/DOCX Output)

```python
@dataclass
class CulturalData:
    culture_name: str
    region: str
    traditional_foods: List[str]
    dietary_practices: List[str]
    religious_restrictions: List[str]
    festive_foods: Dict[str, List[str]]  # occasion -> foods
    cooking_methods: List[str]
    medicinal_foods: Dict[str, str]      # food -> use
    cultural_beliefs: List[str]
    extraction_confidence: float
```

### AgeSpecificData (CSV/JSON Output)

```python
@dataclass
class AgeSpecificData:
    age_group: str
    min_age_months: Optional[int]
    max_age_months: Optional[int]
    calories_per_day: Optional[float]
    protein_grams_per_day: Optional[float]
    vitamins: Dict[str, float]
    minerals: Dict[str, float]
    recommended_foods: List[str]
    foods_to_avoid: List[str]
    portion_sizes: Dict[str, str]
    extraction_confidence: float
```

## Configuration

The module can be configured through the `ProcessingConfig` class or environment variables:

```python
from file_processing.utils.config import ProcessingConfig

config = ProcessingConfig(
    output_dir="custom_outputs",
    log_level="DEBUG",
    max_file_size_mb=50,
    min_confidence_score=0.4,
    parallel_processing=True
)

processor = FileProcessingOrchestrator()
# Apply custom configuration
```

### Environment Variables

```bash
export FILE_PROCESSING_OUTPUT_DIR="./outputs"
export FILE_PROCESSING_LOG_LEVEL="INFO"
export FILE_PROCESSING_MAX_FILE_SIZE_MB="100"
export FILE_PROCESSING_MIN_CONFIDENCE="0.3"
```

## Output Format

The module generates integrated JSON output with the following structure:

```json
{
  "metadata": {
    "version": "1.0.0",
    "processing_date": "2024-01-01T12:00:00",
    "description": "Integrated nutritional knowledge base"
  },
  "processing_summary": {
    "total_files": 10,
    "processed_successfully": 9,
    "nutritional_entries": 25,
    "cultural_entries": 8,
    "age_specific_entries": 12
  },
  "nutritional_data": [
    {
      "food_name": "Brown Rice",
      "calories_per_100g": 111,
      "macronutrients": {
        "protein": 2.6,
        "carbs": 22.0,
        "fat": 0.9
      },
      "extraction_confidence": 0.85
    }
  ],
  "cultural_data": [
    {
      "culture_name": "Indian",
      "traditional_foods": ["Rice", "Dal", "Curry"],
      "festive_foods": {
        "Diwali": ["Sweets", "Nuts"]
      },
      "extraction_confidence": 0.75
    }
  ],
  "age_specific_data": [
    {
      "age_group": "toddler_1_3y",
      "calories_per_day": 1000,
      "recommended_foods": ["Milk", "Fruits", "Vegetables"],
      "extraction_confidence": 0.9
    }
  ]
}
```

## Advanced Usage

### Custom Parser Development

To add support for a new file type:

1. Create a new parser class in the `parsers/` directory
2. Implement the required parsing methods
3. Add the parser to the main orchestrator
4. Update the configuration

```python
class CustomParser:
    def parse_file(self, file_path: str) -> List[CustomData]:
        # Implement your parsing logic
        pass
```

### Batch Processing with Filters

```python
# Process only PDF files in directory
results = processor.process_directory(
    "./documents",
    file_patterns=["*.pdf"]
)

# Process with quality filters
high_quality_data = [
    item for item in results['nutritional_data']
    if item.extraction_confidence > 0.7
]
```

### Integration with Databases

```python
# Save results to database
def save_to_database(results):
    for item in results['nutritional_data']:
        # Save to your database
        db.save_nutritional_data(item)
```

## Examples

See the `examples/` directory for sample input files and usage examples:

- `sample_nutrition_facts.pdf` - Example PDF with nutrition labels
- `sample_cultural_info.md` - Example Markdown with cultural food information
- `sample_age_nutrition.csv` - Example CSV with age-specific requirements

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=file_processing
```

## Troubleshooting

### Common Issues

1. **PDF Parsing Errors**

   - Ensure PDF is not password-protected
   - Try different extraction methods: `pdfplumber` vs `PyPDF2`
   - Check if PDF contains text (not just images)

2. **DOCX Files Not Processing**

   - Install `python-docx`: `pip install python-docx`
   - Ensure DOCX file is not corrupted

3. **Low Confidence Scores**

   - Review extraction patterns in configuration
   - Check if input files contain expected data formats
   - Adjust minimum confidence threshold

4. **Memory Issues with Large Files**
   - Reduce `max_file_size_mb` in configuration
   - Process files individually instead of batch processing
   - Enable chunked processing for large datasets

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or configure file logging
processor = FileProcessingOrchestrator()
processor.config.log_file = "processing.log"
processor.config.log_level = "DEBUG"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This module is part of the Nutritional AI Chatbot project and follows the same licensing terms.

## Support

For issues, questions, or contributions:

1. Check the troubleshooting section
2. Review existing issues in the project repository
3. Create a new issue with detailed information about your problem
4. Include sample files and error logs when possible

## Changelog

### Version 1.0.0 (Current)

- Initial release
- Support for PDF, MD, DOCX, CSV, JSON files
- Integrated processing pipeline
- Quality assessment and confidence scoring
- Command-line interface
- Comprehensive documentation

### Future Enhancements

- OCR support for scanned documents
- Machine learning-based classification
- Web scraping capabilities
- Database integration
- REST API interface
- Multi-language support
