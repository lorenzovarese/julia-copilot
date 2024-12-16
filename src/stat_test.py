import json
import argparse
from statsmodels.stats.contingency_tables import mcnemar

# Function to load 'passed' field from JSON file
def load_passed_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Extract 'passed' field (True/False)
    return [item['passed'] for item in data]

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Perform McNemar\'s test on two models.')
    parser.add_argument('file1', type=str, help='Path to the first JSON file.')
    parser.add_argument('file2', type=str, help='Path to the second JSON file.')

    args = parser.parse_args()
    file1 = args.file1
    file2 = args.file2

    # Load data for both models
    model1_results = load_passed_results(file1)
    model2_results = load_passed_results(file2)
    
    # Check if both lists are the same length
    if len(model1_results) != len(model2_results):
        raise ValueError("Both models must be evaluated on the same number of tests.")
    
    # Create contingency table for McNemar's test
    # a: Both correct, b: Model 1 correct & Model 2 wrong, c: Model 1 wrong & Model 2 correct, d: Both wrong
    a = sum(1 for m1, m2 in zip(model1_results, model2_results) if m1 and m2)
    b = sum(1 for m1, m2 in zip(model1_results, model2_results) if m1 and not m2)
    c = sum(1 for m1, m2 in zip(model1_results, model2_results) if not m1 and m2)
    d = sum(1 for m1, m2 in zip(model1_results, model2_results) if not m1 and not m2)
    
    # Perform McNemar's test
    contingency_table = [[a, b], [c, d]]
    result = mcnemar(contingency_table, exact=True)
    
    # Print the results
    print("\nMcNemar's Test Results:")
    print(f"p-value: {result.pvalue}")
    
    # Interpret results
    alpha = 0.05
    if result.pvalue < alpha:
        print("The difference between the models is statistically significant.")
    else:
        print("The difference between the models is NOT statistically significant.")
