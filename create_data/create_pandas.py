import json
import random

templates = [

    # ---------- BASIC AGGREGATIONS ----------
    (
        "Compute the mean of {col} grouped by {group}.",
        "Columns include {group} and {col}.",
        "df.groupby('{group}')['{col}'].mean()"
    ),
    (
        "Compute the median of {col} grouped by {group}.",
        "Columns include {group} and {col}.",
        "df.groupby('{group}')['{col}'].median()"
    ),
    (
        "Compute the sum of {col} grouped by {group}.",
        "Columns include {group} and {col}.",
        "df.groupby('{group}')['{col}'].sum()"
    ),

    # ---------- MULTI-METRIC AGG ----------
    (
        "Compute the mean and standard deviation of {col} grouped by {group}.",
        "Columns include {group} and {col}.",
        "df.groupby('{group}')['{col}'].agg(['mean', 'std'])"
    ),

    # ---------- FILTERING ----------
    (
        "Filter rows where {col} is greater than {val}.",
        "The column name is {col}.",
        "df[df['{col}'] > {val}]"
    ),
    (
        "Filter rows where {col} is less than or equal to {val}.",
        "The column name is {col}.",
        "df[df['{col}'] <= {val}]"
    ),

    # ---------- SORTING ----------
    (
        "Sort the DataFrame by {col} in descending order.",
        "The column name is {col}.",
        "df.sort_values('{col}', ascending=False)"
    ),
    (
        "Sort the DataFrame by {col} in ascending order.",
        "The column name is {col}.",
        "df.sort_values('{col}')"
    ),

    # ---------- DATETIME ----------
    (
        "Extract the hour from a datetime column.",
        "The datetime column is {dt}.",
        "df['hour'] = df['{dt}'].dt.hour"
    ),
    (
        "Extract the day of week from a datetime column.",
        "The datetime column is {dt}.",
        "df['dayofweek'] = df['{dt}'].dt.dayofweek"
    ),

    # ---------- RESAMPLING ----------
    (
        "Compute the daily mean of {col}.",
        "The datetime column is {dt}.",
        "df.resample('D', on='{dt}')['{col}'].mean()"
    ),

    # ---------- CLEANING ----------
    (
        "Replace missing values in {col} with the column mean.",
        "{col} contains missing values.",
        "df['{col}'] = df['{col}'].fillna(df['{col}'].mean())"
    ),
    (
        "Drop rows where {col} is missing.",
        "{col} contains missing values.",
        "df = df.dropna(subset=['{col}'])"
    ),

    # ---------- TYPE HANDLING ----------
    (
        "Convert {col} to numeric and drop invalid rows.",
        "{col} has non-numeric values.",
        "df = df[pd.to_numeric(df['{col}'], errors='coerce').notna()]"
    ),

    # ---------- DEBUGGING / COMMON ERRORS ----------
    (
        "Fix a pandas aggregation error when computing the mean.",
        "df.groupby(['{group}']).mean() raises agg function failed.",
        "df.groupby(['{group}']).mean(numeric_only=True)"
    ),
    (
        "Fix a SettingWithCopyWarning when assigning a new column.",
        "df_filtered = df[df['{group}'] == value]",
        "df_filtered = df[df['{group}'] == value].copy()"
    ),

    # ---------- PIVOT / RESHAPE ----------
    (
        "Create a pivot table of mean {col} by {group}.",
        "Columns include {group} and {col}.",
        "df.pivot_table(values='{col}', index='{group}', aggfunc='mean')"
    ),

    # ---------- ROLLING ----------
    (
        "Compute a rolling 3-period mean of {col}.",
        "The DataFrame is time-ordered.",
        "df['{col}'].rolling(3).mean()"
    )
]

columns = ["power_OBS", "gain", "price"]
groups = ["LZ", "models"]
dates = ["INIT_DATE_TIME", "DateTime"]

with open("/Users/drazenzack/Desktop/LLM_model/data/train.jsonl", "w") as f:
    for _ in range(500):
        t = random.choice(templates)

        record = {
            "instruction": t[0].format(
                col=random.choice(columns),
                group=random.choice(groups),
                dt=random.choice(dates),
                val=random.choice([100, 150, 200])
            ),
            "input": t[1].format(
                col=random.choice(columns),
                group=random.choice(groups),
                dt=random.choice(dates),
                val=random.choice([100, 150, 200])
            ),
            "output": t[2].format(
                col=random.choice(columns),
                group=random.choice(groups),
                dt=random.choice(dates),
                val=random.choice([100, 150, 200])
            )
        }

        f.write(json.dumps(record) + "\n")