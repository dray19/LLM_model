from test_basemodel import BaseModelPredictor
import json
import csv
from difflib import SequenceMatcher

def calculate_similarity(expected, generated):
    """Calculate similarity between expected and generated strings using SequenceMatcher."""
    return SequenceMatcher(None, expected, generated).ratio()


def remove_outer_parentheses(code: str) -> str:
    code = code.strip()
    if code.startswith("(") and code.endswith(")"):
        return code[1:-1].strip()
    return code

def remove_comment_lines(code: str) -> str:
    return "\n".join(
        line for line in code.splitlines()
        if "#" not in line
    )

def collapse_to_one_line(code: str) -> str:
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    return "".join(lines)


def evaluate_basemodel(test_file, output_csv):
    """
    Load test cases from a JSONL file, evaluate using the base model, and save predictions and metrics to a CSV file.

    Args:
        test_file (str): Path to the test.jsonl file containing the test cases.
        output_csv (str): Path to save the CSV results.
    """
    # Initialize the base model predictor
    predictor = BaseModelPredictor()

    # Load test cases from the JSONL file
    test_cases = []
    with open(test_file, "r") as f:
        for line in f:
            test_cases.append(json.loads(line))
            

    # Prepare results for CSV
    results = []
    for i, test in enumerate(test_cases):
        print(f"Evaluating Test Case {i + 1}...")
        try:
            # Generate response
            response = predictor.generate_response(test["messages"])
            response = response.replace("```python", "").replace("```", "").strip()
            response = response.replace("result =", "").strip()
            response = response.replace("import pandas as pd", "").strip()
            response = remove_comment_lines(response)
            response = remove_outer_parentheses(response)
            response = collapse_to_one_line(response)
            import pandas as pd
            print(f"Generated: {response}")
            print(f"Expected: {test['expected']}")

            # Calculate similarity as a metric
            similarity_score = calculate_similarity(test["expected"], response)

            # Determine if the prediction is correct
            # Here we define a correct prediction as having similarity >= 0.95
            is_correct = similarity_score >= 0.90
            method_correct = similarity_score >= 0.70

            # Append the result to the results list
            results.append({
                "test_case_id": i + 1,
                "instruction": test["messages"][1]["content"],
                "expected_output": test["expected"],
                "generated_output": response,
                "similarity_score": round(similarity_score, 2),
                "is_correct": is_correct,
                "method_correct": method_correct
            })
        except Exception as e:
            print(f"Error during Test Case {i + 1}: {e}")
            results.append({
                "test_case_id": i + 1,
                "instruction": test["messages"][1]["content"],
                "expected_output": test["expected"],
                "generated_output": "",
                "similarity_score": 0.0,
                "is_correct": False,
                "method_correct": False,
                "error": str(e)
            })
        print("-" * 60)

    # Write results to a CSV file
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = [
            "test_case_id",
            "instruction",
            "expected_output",
            "generated_output",
            "similarity_score",
            "is_correct",
            "method_correct",
            "error"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results have been saved to {output_csv}.")

if __name__ == "__main__":
    # Path to the test cases file
    test_file_path = "data/test.jsonl"

    # Path to save the results CSV file
    output_csv_path = "Model_Results/basemodel_results.csv"

    # Run evaluation
    evaluate_basemodel(test_file_path, output_csv_path)
