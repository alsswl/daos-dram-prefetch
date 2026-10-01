import pandas as pd

# Replace 'data.csv' with your CSV file path
df = pd.read_csv('stack_output_1.1.csv')

# Optionally, filter out rows where generation_tokens are zero to avoid division by zero errors
df = df[df['generation_tokens'] != 0]

# Calculate the ratio: generation_time / generation_tokens for each row
df['ratio'] = df['generation_time'] / df['generation_tokens']

# Calculate the average of these ratios
average_ratio = df['ratio'].mean()

print("Average generation_time / generation_tokens:", average_ratio)
