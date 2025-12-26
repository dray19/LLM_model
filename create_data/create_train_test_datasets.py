import json
import random
from typing import List, Tuple

def split_data(data: List, train_ratio: float = 0.8, seed: int = None) -> Tuple[List, List]:
    """
    Splits data into training and testing datasets ensuring no overlapping examples.

    Args:
        data (List): The dataset to split.
        train_ratio (float): Proportion of data to include in the training set. Default is 0.8.
        seed (int): Random seed for reproducibility. Default is None.

    Returns:
        Tuple[List, List]: Training and testing datasets.
    """
    if seed is not None:
        random.seed(seed)

    data_shuffled = data[:]
    random.shuffle(data_shuffled)

    split_index = int(len(data) * train_ratio)
    train_set = data_shuffled[:split_index]
    test_set = data_shuffled[split_index:]

    return train_set, test_set

def create_train_examples(num_examples: int) -> List[dict]:
    """
    Create a list of random examples formatted for training.

    Args:
        num_examples (int): Number of examples to generate.

    Returns:
        List[dict]: A list of generated training examples.
    """
    templates = [
        # ---------- MULTI-AGGREGATIONS ----------
        (
            "Compute the mean and std of {col} grouped by {group}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df.groupby('{group}')['{col}'].agg(['mean','std'])"
        ),
        (
            "Compute count of rows grouped by {group}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df.groupby('{group}').size()"
        ),
        # ---------- DATE / TIME ----------
        (
            "Extract hour from INIT_DATE_TIME.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df['hour'] = df['INIT_DATE_TIME'].dt.hour"
        ),
        (
            "Filter rows for a specific hour {val}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df[df['INIT_DATE_TIME'].dt.hour == {val}]"
        ),
        # ---------- TOP / BOTTOM ----------
        (
            "Return top {val} rows by {col}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df.nlargest({val}, '{col}')"
        ),
        (
            "Return bottom {val} rows by {col}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df.nsmallest({val}, '{col}')"
        ),
        # ---------- DISTINCT / UNIQUE ----------
        (
            "Get unique values of {col}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df['{col}'].unique()"
        ),
        (
            "Count unique values of {col}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df['{col}'].nunique()"
        ),
        # ---------- BASIC AGGREGATIONS ----------
        (
            "Compute the mean of {col} grouped by {group}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df.groupby('{group}')['{col}'].mean()"
        ),
        (
            "Compute the sum of {col} grouped by {group}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df.groupby('{group}')['{col}'].sum()"
        ),
        (
            "Compute the min of {col} grouped by {group}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df.groupby('{group}')['{col}'].min()"
        ),
         (
            "Compute the max of {col} grouped by {group}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df.groupby('{group}')['{col}'].max()"
        ),
        # ---------- FILTERING ----------
        (
            "Filter rows where {col} is greater than {val}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df[df['{col}'] > {val}]"
        ),
        (
            "Filter rows where {col} is greater than or equal to{val}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df[df['{col}'] >= {val}]"
        ),
        (
            "Filter rows where {col} is less than or equal to{val}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df[df['{col}'] <= {val}]"
        ),
        (
            "Filter rows where {col} is less than {val}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df[df['{col}'] < {val}]"
        ),
         (
            "Filter rows where {col} is equal to{val}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df[df['{col}'] == {val}]"
        ),
        (
            "Filter rows where gain is negative.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df[df['gain'] < 0]"
        ),
        # ---------- SORTING ----------
        (
            "Sort the DataFrame by {col}.",
            "Columns: INIT_DATE_TIME, LZ, models, power_OBS, day_ahead, pred, gain",
            "df.sort_values('{col}')"
        )
        
    ]

    columns = ["pred", "gain", "power_OBS", "day_ahead"]
    groups = ["LZ", "models"]
    dates = ["INIT_DATE_TIME"]

    examples = []
    for _ in range(num_examples):
        t = random.choice(templates)
        record = {
            "instruction": t[0].format(
                col=random.choice(columns),
                group=random.choice(groups),
                dt=random.choice(dates),
                val=round(random.uniform(0, 200),0)
            ),
            "input": t[1].format(
                col=random.choice(columns),
                group=random.choice(groups),
                dt=random.choice(dates),
                val=round(random.uniform(0, 200),0)
            ),
            "output": t[2].format(
                col=random.choice(columns),
                group=random.choice(groups),
                dt=random.choice(dates),
                val=round(random.uniform(0, 200),0)
            )
        }
        examples.append(record)
    return examples

def create_test_examples(train_examples: List) -> List[dict]:
    """
    Convert the training examples to testing format with 'messages' and 'expected'.

    Args:
        train_examples (List): The training examples.

    Returns:
        List[dict]: A list of testing examples.
    """
    test_examples = []
    for example in train_examples:
        test_examples.append({
            "messages": [
                {"role": "system", "content": "You are a code generator. Respond with ONLY valid Python code. "
            "No explanations. No markdown. No imports. NO sample data.\n\n"
            "Rules:\n"
            "- You MUST assume a pandas DataFrame named df already exists in memory and is the ONLY input dataset.\n"
            "- Generate code that operates ONLY on df or intermediate objects derived directly from df.\n"
            "- Do NOT reference external variables, files, paths, configs, or objects not derived from df.\n"
            "- Do NOT read from or write to disk.\n"
            "- Do NOT make network calls.\n"
            "- Do NOT use randomness or non-deterministic behavior.\n"
            "- Do NOT use unsafe operations (eval, exec, compile, ast, subprocess, os, shell commands).\n"
            "- Do NOT mutate df unless explicitly requested; prefer creating new objects.\n"
            "- Avoid chained assignment; use .loc for assignments.\n"
            "- Do NOT assume column dtypes; handle numeric vs non-numeric safely.\n"
            "- For groupby aggregations, use numeric_only=True when appropriate.\n"
            "- Guard against missing columns: if required columns are missing, assign result to "
            "a clear error string like \"ERROR: missing columns: ['col1', 'col2']\".\n"
            "- Always assign the final output to a variable named result.\n"
            "- Do NOT print unless explicitly requested.\n"
            "- Keep the code minimal, deterministic, and directly executable."},
                {"role": "user", "content": example["instruction"] + " \n" + example["input"]}
            ],
            "expected": example["output"]
        })
    return test_examples

if __name__ == "__main__":
    # Number of total examples to create
    total_examples = 1000

    # Train-test split ratio
    train_ratio = 0.85

    # Create training examples
    train_examples = create_train_examples(total_examples)

    # Split into train and test data
    train_set, test_set_transform = split_data(train_examples, train_ratio, seed=42)

    # Convert the test set into the desired structure
    test_set = create_test_examples(test_set_transform)

    # Save training data
    with open("data/train.jsonl", "w") as train_file:
        for example in train_set:
            train_file.write(json.dumps(example) + "\n")

    # Save testing data
    with open("data/test.jsonl", "w") as test_file:
        for example in test_set:
            test_file.write(json.dumps(example) + "\n")

    print(f"Generated {len(train_set)} training examples in 'train.jsonl'.")
    print(f"Generated {len(test_set)} testing examples in 'test.jsonl'.")