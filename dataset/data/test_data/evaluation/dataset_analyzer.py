import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import statistics


class TestDatasetAnalyzer:
    """
    Comprehensive analyzer for the nutritional AI chatbot test dataset.
    Provides statistical analysis, demographic distribution, and evaluation insights.
    """

    def __init__(self, dataset_path: str):
        """Initialize analyzer with dataset path."""
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset()

    def load_dataset(self) -> List[Dict]:
        """Load the test dataset from JSON file."""
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Dataset file not found: {self.dataset_path}")
            return []

    def get_basic_statistics(self) -> Dict:
        """Get basic statistics about the dataset."""
        if not self.dataset:
            return {}

        stats = {
            "total_queries": len(self.dataset),
            "age_distribution": self.analyze_age_distribution(),
            "culture_distribution": self.analyze_culture_distribution(),
            "scenario_distribution": self.analyze_scenario_distribution(),
            "complexity_distribution": self.analyze_complexity_distribution(),
            "dietary_preference_distribution": self.analyze_dietary_preferences(),
        }

        return stats

    def analyze_age_distribution(self) -> Dict:
        """Analyze age group distribution in the dataset."""
        age_groups = {}
        ages = []

        for query in self.dataset:
            age_group = query["user_profile"]["age_group"]
            age = query["user_profile"]["age"]

            age_groups[age_group] = age_groups.get(age_group, 0) + 1
            ages.append(age)

        return {
            "by_group": age_groups,
            "age_stats": {
                "min_age": min(ages),
                "max_age": max(ages),
                "mean_age": statistics.mean(ages),
                "median_age": statistics.median(ages),
            },
        }

    def analyze_culture_distribution(self) -> Dict:
        """Analyze cultural representation in the dataset."""
        cultures = {}
        for query in self.dataset:
            culture = query["user_profile"]["culture"]
            cultures[culture] = cultures.get(culture, 0) + 1

        # Calculate cultural diversity metrics
        total_queries = len(self.dataset)
        culture_percentages = {
            culture: (count / total_queries) * 100
            for culture, count in cultures.items()
        }

        return {
            "counts": cultures,
            "percentages": culture_percentages,
            "diversity_score": len(
                cultures
            ),  # Number of different cultures represented
        }

    def analyze_scenario_distribution(self) -> Dict:
        """Analyze scenario type coverage."""
        scenarios = {}
        for query in self.dataset:
            scenario = query["scenario_type"]
            scenarios[scenario] = scenarios.get(scenario, 0) + 1

        return {"counts": scenarios, "unique_scenarios": len(scenarios)}

    def analyze_complexity_distribution(self) -> Dict:
        """Analyze query complexity distribution."""
        complexity = {}
        for query in self.dataset:
            level = query["complexity"]
            complexity[level] = complexity.get(level, 0) + 1

        return complexity

    def analyze_dietary_preferences(self) -> Dict:
        """Analyze dietary preference representation."""
        preferences = {}
        for query in self.dataset:
            pref = query["user_profile"]["dietary_preference"]
            preferences[pref] = preferences.get(pref, 0) + 1

        return preferences

    def analyze_health_conditions(self) -> Dict:
        """Analyze health status representation."""
        health_statuses = {}
        for query in self.dataset:
            status = query["user_profile"]["health_status"]
            health_statuses[status] = health_statuses.get(status, 0) + 1

        return health_statuses

    def generate_demographic_cross_analysis(self) -> Dict:
        """Generate cross-analysis of demographics."""
        cross_analysis = {
            "age_culture_matrix": {},
            "age_dietary_matrix": {},
            "culture_health_matrix": {},
        }

        # Age-Culture matrix
        for query in self.dataset:
            age_group = query["user_profile"]["age_group"]
            culture = query["user_profile"]["culture"]

            if age_group not in cross_analysis["age_culture_matrix"]:
                cross_analysis["age_culture_matrix"][age_group] = {}

            culture_dict = cross_analysis["age_culture_matrix"][age_group]
            culture_dict[culture] = culture_dict.get(culture, 0) + 1

        return cross_analysis

    def identify_coverage_gaps(self) -> Dict:
        """Identify potential gaps in dataset coverage."""
        gaps = {
            "underrepresented_cultures": [],
            "underrepresented_scenarios": [],
            "missing_combinations": [],
        }

        # Identify cultures with low representation
        culture_dist = self.analyze_culture_distribution()
        total_queries = len(self.dataset)
        min_expected_per_culture = total_queries / 20  # Expect at least 5% per culture

        for culture, count in culture_dist["counts"].items():
            if count < min_expected_per_culture:
                gaps["underrepresented_cultures"].append(
                    {
                        "culture": culture,
                        "count": count,
                        "percentage": (count / total_queries) * 100,
                    }
                )

        return gaps

    def export_analysis_report(self, output_path: str):
        """Export comprehensive analysis report."""
        analysis = {
            "dataset_info": {
                "total_queries": len(self.dataset),
                "analysis_date": datetime.now().isoformat(),
                "dataset_path": self.dataset_path,
            },
            "basic_statistics": self.get_basic_statistics(),
            "health_conditions": self.analyze_health_conditions(),
            "demographic_cross_analysis": self.generate_demographic_cross_analysis(),
            "coverage_gaps": self.identify_coverage_gaps(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        print(f"Analysis report exported to: {output_path}")

    def create_evaluation_batch(
        self, batch_size: int = 50, output_dir: str = "./evaluation_batches/"
    ) -> List[str]:
        """Create evaluation batches for human assessment."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        batch_files = []
        total_queries = len(self.dataset)

        for i in range(0, total_queries, batch_size):
            batch_data = self.dataset[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            batch_file = os.path.join(output_dir, f"evaluation_batch_{batch_num}.json")

            with open(batch_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "batch_info": {
                            "batch_number": batch_num,
                            "queries_count": len(batch_data),
                            "query_range": f"{i + 1}-{min(i + batch_size, total_queries)}",
                        },
                        "queries": batch_data,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            batch_files.append(batch_file)

        print(f"Created {len(batch_files)} evaluation batches in {output_dir}")
        return batch_files


# Usage example and analysis execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TestDatasetAnalyzer(
        "data/test_data/queries/comprehensive_test_dataset.json"
    )

    # Generate comprehensive analysis
    print("=== Test Dataset Analysis ===")
    print(f"Total queries: {len(analyzer.dataset)}")

    # Basic statistics
    stats = analyzer.get_basic_statistics()
    print("\n=== Age Distribution ===")
    for group, count in stats["age_distribution"]["by_group"].items():
        print(f"{group}: {count}")

    print("\n=== Culture Distribution ===")
    for culture, count in stats["culture_distribution"]["counts"].items():
        percentage = stats["culture_distribution"]["percentages"][culture]
        print(f"{culture}: {count} ({percentage:.1f}%)")

    print("\n=== Scenario Types ===")
    for scenario, count in stats["scenario_distribution"]["counts"].items():
        print(f"{scenario}: {count}")

    print("\n=== Complexity Distribution ===")
    for level, count in stats["complexity_distribution"].items():
        print(f"{level}: {count}")

    # Export comprehensive report
    analyzer.export_analysis_report(
        "data/test_data/evaluation/dataset_analysis_report.json"
    )

    # Create evaluation batches
    analyzer.create_evaluation_batch(
        batch_size=25, output_dir="data/test_data/evaluation/batches/"
    )

    print("\nAnalysis complete! Check the evaluation directory for detailed reports.")
