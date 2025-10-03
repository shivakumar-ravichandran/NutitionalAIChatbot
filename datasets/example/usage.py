"""
Example Usage Script for File Processing Module

This script demonstrates various ways to use the file processing module
for extracting nutritional, cultural, and age-specific data from files.
"""

import os
import sys
from pathlib import Path

# Add the file_processing module to the path
sys.path.append(str(Path(__file__).parent))

from main import FileProcessingOrchestrator
from utils.config import ProcessingConfig


def example_single_file_processing():
    """Example: Process a single file"""
    print("=== Single File Processing Example ===")

    # Initialize the orchestrator
    processor = FileProcessingOrchestrator(output_dir="example_outputs")

    # Process the sample cultural information file
    sample_file = "examples/sample_cultural_info.md"

    if Path(sample_file).exists():
        result = processor.process_single_file(sample_file)

        print(f"Processing result: {result}")

        if result["success"]:
            print(f"✅ Successfully processed {sample_file}")
            print(f"   - File type: {result['file_type']}")
            print(f"   - Entries extracted: {result['entries_extracted']}")
            print(f"   - Data type: {result['data_type']}")
        else:
            print(f"❌ Failed to process {sample_file}: {result['error']}")
    else:
        print(f"❌ Sample file not found: {sample_file}")


def example_multiple_files_processing():
    """Example: Process multiple files at once"""
    print("\n=== Multiple Files Processing Example ===")

    processor = FileProcessingOrchestrator(output_dir="example_outputs")

    # List of sample files to process
    sample_files = [
        "examples/sample_cultural_info.md",
        "examples/sample_age_nutrition.csv",
        "examples/sample_age_nutrition.json",
    ]

    # Filter existing files
    existing_files = [f for f in sample_files if Path(f).exists()]

    if existing_files:
        print(f"Processing {len(existing_files)} files...")
        results = processor.process_files(
            existing_files, "example_integrated_output.json"
        )

        # Print processing statistics
        stats = processor.get_processing_statistics()
        print(f"\n📊 Processing Statistics:")
        print(f"   - Total files: {stats['overview']['total_files']}")
        print(
            f"   - Successfully processed: {stats['overview']['processed_successfully']}"
        )
        print(f"   - Cultural entries: {stats['data_breakdown']['cultural_entries']}")
        print(
            f"   - Age-specific entries: {stats['data_breakdown']['age_specific_entries']}"
        )
        print(
            f"   - Average confidence: {stats['quality_metrics']['cultural_confidence_avg']:.2f}"
        )

        if results["errors"]:
            print(f"\n⚠️ Errors encountered:")
            for error in results["errors"]:
                print(f"   - {error['file']}: {error['error']}")
    else:
        print("❌ No sample files found to process")


def example_directory_processing():
    """Example: Process all files in a directory"""
    print("\n=== Directory Processing Example ===")

    processor = FileProcessingOrchestrator(output_dir="example_outputs")

    examples_dir = "examples"

    if Path(examples_dir).exists():
        print(f"Processing all supported files in {examples_dir}/...")
        results = processor.process_directory(examples_dir)

        print(f"📂 Directory processing completed:")
        print(
            f"   - Files found and processed: {results['processing_summary']['processed_successfully']}"
        )
        print(
            f"   - Total entries extracted: {sum([
            results['processing_summary']['nutritional_entries'],
            results['processing_summary']['cultural_entries'],
            results['processing_summary']['age_specific_entries']
        ])}"
        )
    else:
        print(f"❌ Examples directory not found: {examples_dir}")


def example_custom_configuration():
    """Example: Using custom configuration"""
    print("\n=== Custom Configuration Example ===")

    # Create custom configuration
    custom_config = ProcessingConfig(
        output_dir="custom_outputs",
        log_level="DEBUG",
        min_confidence_score=0.5,  # Higher confidence threshold
        save_individual_files=True,
        continue_on_error=True,
    )

    print(f"🔧 Using custom configuration:")
    print(f"   - Output directory: {custom_config.output_dir}")
    print(f"   - Log level: {custom_config.log_level}")
    print(f"   - Min confidence: {custom_config.min_confidence_score}")

    # Initialize with custom config
    processor = FileProcessingOrchestrator(output_dir=custom_config.output_dir)

    # Process with custom settings
    sample_file = "examples/sample_cultural_info.md"
    if Path(sample_file).exists():
        result = processor.process_single_file(sample_file)
        print(
            f"   - Processing result: {'✅ Success' if result['success'] else '❌ Failed'}"
        )


def example_validation_only():
    """Example: Validate files without processing"""
    print("\n=== File Validation Example ===")

    processor = FileProcessingOrchestrator()

    # List of files to validate
    test_files = [
        "examples/sample_cultural_info.md",
        "examples/sample_age_nutrition.csv",
        "examples/sample_age_nutrition.json",
        "examples/nonexistent_file.pdf",  # This will fail validation
        "README.md",  # This exists but might not have expected content
    ]

    validation_result = processor.validate_input_files(test_files)

    print(f"📋 Validation Results:")
    print(f"   - Valid files: {validation_result['summary']['valid']}")
    print(f"   - Invalid files: {validation_result['summary']['invalid']}")

    if validation_result["valid"]:
        print(f"   ✅ Valid files:")
        for file in validation_result["valid"]:
            print(f"      - {file}")

    if validation_result["invalid"]:
        print(f"   ❌ Invalid files:")
        for issue in validation_result["invalid"]:
            print(f"      - {issue}")


def example_data_analysis():
    """Example: Analyze extracted data"""
    print("\n=== Data Analysis Example ===")

    processor = FileProcessingOrchestrator(output_dir="example_outputs")

    # Process sample files
    sample_files = [
        f
        for f in [
            "examples/sample_cultural_info.md",
            "examples/sample_age_nutrition.csv",
        ]
        if Path(f).exists()
    ]

    if sample_files:
        results = processor.process_files(sample_files)

        # Analyze cultural data
        cultural_data = processor.processing_results["cultural_data"]
        if cultural_data:
            print(f"🍽️ Cultural Data Analysis:")
            for data in cultural_data:
                print(f"   - Culture: {data.culture_name}")
                print(f"     • Traditional foods: {len(data.traditional_foods)} items")
                print(f"     • Cooking methods: {len(data.cooking_methods)} methods")
                print(f"     • Festival foods: {len(data.festive_foods)} occasions")
                print(f"     • Confidence: {data.extraction_confidence:.2f}")

        # Analyze age-specific data
        age_data = processor.processing_results["age_specific_data"]
        if age_data:
            print(f"\n👶 Age-Specific Data Analysis:")
            for data in age_data:
                print(f"   - Age group: {data.age_group}")
                print(f"     • Calories/day: {data.calories_per_day}")
                print(f"     • Protein/day: {data.protein_grams_per_day}g")
                print(f"     • Recommended foods: {len(data.recommended_foods)} items")
                print(f"     • Confidence: {data.extraction_confidence:.2f}")


def create_sample_output_analysis():
    """Example: Create and analyze sample output"""
    print("\n=== Sample Output Analysis ===")

    processor = FileProcessingOrchestrator(output_dir="example_outputs")

    # Check if we have any output files
    output_dir = Path("example_outputs")
    if output_dir.exists():
        json_files = list(output_dir.glob("*.json"))
        if json_files:
            print(f"📄 Found {len(json_files)} output files:")
            for file in json_files:
                file_size = file.stat().st_size
                print(f"   - {file.name} ({file_size:,} bytes)")
        else:
            print("   No output files found. Run other examples first.")
    else:
        print("   Output directory doesn't exist. Run other examples first.")


def main():
    """Run all examples"""
    print("🚀 File Processing Module - Usage Examples")
    print("=" * 50)

    try:
        # Run all examples
        example_single_file_processing()
        example_multiple_files_processing()
        example_directory_processing()
        example_custom_configuration()
        example_validation_only()
        example_data_analysis()
        create_sample_output_analysis()

        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("📁 Check the 'example_outputs' directory for generated files.")

    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
