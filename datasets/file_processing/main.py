"""
File Processing Main Module

This is the main orchestrator module that coordinates all file parsers to process
different input file types and generate structured nutritional knowledge output.

Supported file types:
- PDF files: Extract nutritional information (nutrition facts, food composition data)
- Markdown files: Extract cultural-specific dietary information
- DOCX files: Extract cultural-specific dietary information
- CSV files: Extract age-specific nutritional requirements
- JSON files: Extract age-specific nutritional requirements

Output: Structured JSON with integrated nutritional, cultural, and age-specific data
"""

import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import asdict

# Import the specialized parsers
from parsers.pdf_parser import PDFNutritionalParser, NutritionalData
from parsers.text_parser import TextCulturalParser, CulturalData
from parsers.data_parser import DataAgeSpecificParser, AgeSpecificData

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FileProcessingOrchestrator:
    """
    Main orchestrator class that coordinates all file processing operations
    """

    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the orchestrator

        Args:
            output_dir: Directory where output files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize specialized parsers
        self.pdf_parser = PDFNutritionalParser()
        self.text_parser = TextCulturalParser()
        self.data_parser = DataAgeSpecificParser()

        # Supported file extensions
        self.supported_extensions = {
            ".pdf": "pdf",
            ".md": "markdown",
            ".docx": "docx",
            ".csv": "csv",
            ".json": "json",
        }

        # Processing results storage
        self.processing_results = {
            "nutritional_data": [],
            "cultural_data": [],
            "age_specific_data": [],
            "processing_summary": {},
            "errors": [],
        }

    def process_files(
        self, input_paths: Union[str, List[str]], output_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process multiple files and generate integrated output

        Args:
            input_paths: Single file path or list of file paths to process
            output_filename: Optional custom output filename

        Returns:
            Dictionary containing processing results and summary
        """
        logger.info("Starting file processing orchestration")

        # Normalize input paths
        if isinstance(input_paths, str):
            input_paths = [input_paths]

        # Reset results
        self.processing_results = {
            "nutritional_data": [],
            "cultural_data": [],
            "age_specific_data": [],
            "processing_summary": {},
            "errors": [],
        }

        processed_files = 0
        total_files = len(input_paths)

        # Process each file
        for file_path in input_paths:
            try:
                result = self.process_single_file(file_path)
                if result["success"]:
                    processed_files += 1
                    logger.info(f"Successfully processed: {file_path}")
                else:
                    logger.error(
                        f"Failed to process: {file_path} - {result.get('error', 'Unknown error')}"
                    )
                    self.processing_results["errors"].append(
                        {
                            "file": file_path,
                            "error": result.get("error", "Unknown error"),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
            except Exception as e:
                logger.error(f"Exception processing {file_path}: {str(e)}")
                self.processing_results["errors"].append(
                    {
                        "file": file_path,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Generate processing summary
        self.processing_results["processing_summary"] = {
            "total_files": total_files,
            "processed_successfully": processed_files,
            "failed_files": total_files - processed_files,
            "nutritional_entries": len(self.processing_results["nutritional_data"]),
            "cultural_entries": len(self.processing_results["cultural_data"]),
            "age_specific_entries": len(self.processing_results["age_specific_data"]),
            "processing_date": datetime.now().isoformat(),
            "supported_formats": list(self.supported_extensions.keys()),
        }

        # Save integrated results
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"integrated_nutrition_data_{timestamp}.json"

        output_path = self.output_dir / output_filename
        self.save_integrated_results(output_path)

        logger.info(f"File processing completed. Results saved to: {output_path}")
        return self.processing_results

    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file based on its type

        Args:
            file_path: Path to the file to process

        Returns:
            Dictionary with processing result
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "file_type": None,
            }

        file_extension = file_path.suffix.lower()

        if file_extension not in self.supported_extensions:
            return {
                "success": False,
                "error": f"Unsupported file type: {file_extension}",
                "file_type": file_extension,
                "supported_types": list(self.supported_extensions.keys()),
            }

        file_type = self.supported_extensions[file_extension]
        logger.info(f"Processing {file_type} file: {file_path}")

        try:
            if file_type == "pdf":
                return self.process_pdf_file(str(file_path))
            elif file_type == "markdown":
                return self.process_markdown_file(str(file_path))
            elif file_type == "docx":
                return self.process_docx_file(str(file_path))
            elif file_type == "csv":
                return self.process_csv_file(str(file_path))
            elif file_type == "json":
                return self.process_json_file(str(file_path))
            else:
                return {
                    "success": False,
                    "error": f"No processor available for file type: {file_type}",
                    "file_type": file_type,
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Processing error: {str(e)}",
                "file_type": file_type,
            }

    def process_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """Process PDF file for nutritional information"""
        try:
            nutritional_data = self.pdf_parser.parse_pdf(file_path)

            if nutritional_data:
                self.processing_results["nutritional_data"].extend(nutritional_data)
                return {
                    "success": True,
                    "file_type": "pdf",
                    "entries_extracted": len(nutritional_data),
                    "data_type": "nutritional",
                }
            else:
                return {
                    "success": False,
                    "error": "No nutritional data extracted from PDF",
                    "file_type": "pdf",
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"PDF processing error: {str(e)}",
                "file_type": "pdf",
            }

    def process_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """Process Markdown file for cultural information"""
        try:
            cultural_data = self.text_parser.parse_markdown(file_path)

            if cultural_data:
                self.processing_results["cultural_data"].extend(cultural_data)
                return {
                    "success": True,
                    "file_type": "markdown",
                    "entries_extracted": len(cultural_data),
                    "data_type": "cultural",
                }
            else:
                return {
                    "success": False,
                    "error": "No cultural data extracted from Markdown",
                    "file_type": "markdown",
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Markdown processing error: {str(e)}",
                "file_type": "markdown",
            }

    def process_docx_file(self, file_path: str) -> Dict[str, Any]:
        """Process DOCX file for cultural information"""
        try:
            cultural_data = self.text_parser.parse_docx(file_path)

            if cultural_data:
                self.processing_results["cultural_data"].extend(cultural_data)
                return {
                    "success": True,
                    "file_type": "docx",
                    "entries_extracted": len(cultural_data),
                    "data_type": "cultural",
                }
            else:
                return {
                    "success": False,
                    "error": "No cultural data extracted from DOCX",
                    "file_type": "docx",
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"DOCX processing error: {str(e)}",
                "file_type": "docx",
            }

    def process_csv_file(self, file_path: str) -> Dict[str, Any]:
        """Process CSV file for age-specific information"""
        try:
            age_data = self.data_parser.parse_csv(file_path)

            if age_data:
                self.processing_results["age_specific_data"].extend(age_data)
                return {
                    "success": True,
                    "file_type": "csv",
                    "entries_extracted": len(age_data),
                    "data_type": "age_specific",
                }
            else:
                return {
                    "success": False,
                    "error": "No age-specific data extracted from CSV",
                    "file_type": "csv",
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"CSV processing error: {str(e)}",
                "file_type": "csv",
            }

    def process_json_file(self, file_path: str) -> Dict[str, Any]:
        """Process JSON file for age-specific information"""
        try:
            age_data = self.data_parser.parse_json(file_path)

            if age_data:
                self.processing_results["age_specific_data"].extend(age_data)
                return {
                    "success": True,
                    "file_type": "json",
                    "entries_extracted": len(age_data),
                    "data_type": "age_specific",
                }
            else:
                return {
                    "success": False,
                    "error": "No age-specific data extracted from JSON",
                    "file_type": "json",
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"JSON processing error: {str(e)}",
                "file_type": "json",
            }

    def save_integrated_results(self, output_path: Path) -> bool:
        """
        Save integrated processing results to JSON file

        Args:
            output_path: Path where to save the integrated results

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Convert dataclass objects to dictionaries
            integrated_data = {
                "metadata": {
                    "version": "1.0.0",
                    "processing_date": datetime.now().isoformat(),
                    "description": "Integrated nutritional knowledge base",
                    "data_sources": {
                        "nutritional_data": "PDF files - nutrition facts and food composition",
                        "cultural_data": "MD/DOCX files - cultural dietary practices and traditions",
                        "age_specific_data": "CSV/JSON files - age-specific nutritional requirements",
                    },
                },
                "processing_summary": self.processing_results["processing_summary"],
                "nutritional_data": [
                    asdict(data) for data in self.processing_results["nutritional_data"]
                ],
                "cultural_data": [
                    asdict(data) for data in self.processing_results["cultural_data"]
                ],
                "age_specific_data": [
                    asdict(data)
                    for data in self.processing_results["age_specific_data"]
                ],
                "processing_errors": self.processing_results["errors"],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(integrated_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully saved integrated results to {output_path}")

            # Also save individual category files for easier access
            self.save_category_files(output_path.parent)

            return True

        except Exception as e:
            logger.error(f"Error saving integrated results to {output_path}: {str(e)}")
            return False

    def save_category_files(self, output_dir: Path) -> None:
        """Save individual category files for easier access"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save nutritional data
        if self.processing_results["nutritional_data"]:
            nutritional_file = output_dir / f"nutritional_data_{timestamp}.json"
            nutritional_dict = [
                asdict(data) for data in self.processing_results["nutritional_data"]
            ]
            with open(nutritional_file, "w", encoding="utf-8") as f:
                json.dump(nutritional_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Nutritional data saved to {nutritional_file}")

        # Save cultural data
        if self.processing_results["cultural_data"]:
            cultural_file = output_dir / f"cultural_data_{timestamp}.json"
            cultural_dict = [
                asdict(data) for data in self.processing_results["cultural_data"]
            ]
            with open(cultural_file, "w", encoding="utf-8") as f:
                json.dump(cultural_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Cultural data saved to {cultural_file}")

        # Save age-specific data
        if self.processing_results["age_specific_data"]:
            age_file = output_dir / f"age_specific_data_{timestamp}.json"
            age_dict = [
                asdict(data) for data in self.processing_results["age_specific_data"]
            ]
            with open(age_file, "w", encoding="utf-8") as f:
                json.dump(age_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Age-specific data saved to {age_file}")

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        stats = {
            "overview": self.processing_results["processing_summary"],
            "data_breakdown": {
                "nutritional_entries": len(self.processing_results["nutritional_data"]),
                "cultural_entries": len(self.processing_results["cultural_data"]),
                "age_specific_entries": len(
                    self.processing_results["age_specific_data"]
                ),
            },
            "quality_metrics": {
                "nutritional_confidence_avg": 0.0,
                "cultural_confidence_avg": 0.0,
                "age_specific_confidence_avg": 0.0,
            },
            "errors": len(self.processing_results["errors"]),
        }

        # Calculate average confidence scores
        if self.processing_results["nutritional_data"]:
            nutritional_confidences = [
                data.extraction_confidence
                for data in self.processing_results["nutritional_data"]
            ]
            stats["quality_metrics"]["nutritional_confidence_avg"] = sum(
                nutritional_confidences
            ) / len(nutritional_confidences)

        if self.processing_results["cultural_data"]:
            cultural_confidences = [
                data.extraction_confidence
                for data in self.processing_results["cultural_data"]
            ]
            stats["quality_metrics"]["cultural_confidence_avg"] = sum(
                cultural_confidences
            ) / len(cultural_confidences)

        if self.processing_results["age_specific_data"]:
            age_confidences = [
                data.extraction_confidence
                for data in self.processing_results["age_specific_data"]
            ]
            stats["quality_metrics"]["age_specific_confidence_avg"] = sum(
                age_confidences
            ) / len(age_confidences)

        return stats

    def process_directory(
        self,
        directory_path: str,
        recursive: bool = False,
        file_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process all supported files in a directory

        Args:
            directory_path: Path to directory containing files to process
            recursive: Whether to search subdirectories recursively
            file_patterns: Optional list of file patterns to match (e.g., ['*.pdf', '*.csv'])

        Returns:
            Dictionary containing processing results and summary
        """
        logger.info(f"Processing directory: {directory_path}")

        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(
                f"Directory not found or not a directory: {directory_path}"
            )

        # Find all supported files
        supported_files = []

        if recursive:
            for ext in self.supported_extensions.keys():
                pattern = f"**/*{ext}"
                if file_patterns is None or any(
                    ext in pattern for pattern in file_patterns
                ):
                    supported_files.extend(directory.glob(pattern))
        else:
            for ext in self.supported_extensions.keys():
                pattern = f"*{ext}"
                if file_patterns is None or any(
                    ext in pattern for pattern in file_patterns
                ):
                    supported_files.extend(directory.glob(pattern))

        logger.info(f"Found {len(supported_files)} supported files to process")

        # Process all found files
        file_paths = [str(file_path) for file_path in supported_files]
        return self.process_files(file_paths)

    def validate_input_files(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Validate input files before processing

        Args:
            file_paths: List of file paths to validate

        Returns:
            Dictionary with 'valid' and 'invalid' file lists
        """
        valid_files = []
        invalid_files = []

        for file_path in file_paths:
            path = Path(file_path)

            if not path.exists():
                invalid_files.append(f"{file_path} - File not found")
            elif path.suffix.lower() not in self.supported_extensions:
                invalid_files.append(f"{file_path} - Unsupported file type")
            elif path.stat().st_size == 0:
                invalid_files.append(f"{file_path} - Empty file")
            else:
                valid_files.append(file_path)

        return {
            "valid": valid_files,
            "invalid": invalid_files,
            "summary": {
                "total": len(file_paths),
                "valid": len(valid_files),
                "invalid": len(invalid_files),
            },
        }


def main():
    """
    Example usage and command-line interface
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="File Processing Module for Nutritional AI Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --files nutrition.pdf cultural_info.md age_data.csv
  python main.py --directory ./input_files --recursive
  python main.py --files *.pdf --output custom_output.json
        """,
    )

    parser.add_argument("--files", nargs="+", help="List of files to process")
    parser.add_argument("--directory", help="Directory containing files to process")
    parser.add_argument(
        "--recursive", action="store_true", help="Process directory recursively"
    )
    parser.add_argument("--output", help="Custom output filename")
    parser.add_argument(
        "--output-dir", default="outputs", help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate files without processing",
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = FileProcessingOrchestrator(output_dir=args.output_dir)

    try:
        if args.files:
            # Process specific files
            if args.validate_only:
                validation_result = orchestrator.validate_input_files(args.files)
                print(f"Validation results: {validation_result}")
                return

            results = orchestrator.process_files(args.files, args.output)

        elif args.directory:
            # Process directory
            results = orchestrator.process_directory(
                args.directory, recursive=args.recursive
            )
        else:
            parser.print_help()
            return

        # Print processing statistics
        stats = orchestrator.get_processing_statistics()
        print("\n=== Processing Statistics ===")
        print(
            f"Total files processed: {stats['overview']['processed_successfully']}/{stats['overview']['total_files']}"
        )
        print(f"Nutritional entries: {stats['data_breakdown']['nutritional_entries']}")
        print(f"Cultural entries: {stats['data_breakdown']['cultural_entries']}")
        print(
            f"Age-specific entries: {stats['data_breakdown']['age_specific_entries']}"
        )
        print(f"Processing errors: {stats['errors']}")

        if stats["errors"] > 0:
            print("\nErrors encountered:")
            for error in results["errors"]:
                print(f"  - {error['file']}: {error['error']}")

        print(f"\nProcessing completed successfully!")

    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
