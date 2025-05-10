import pandas as pd
import os

# Get the current directory where the script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define file paths for Parquet and CSV in the same directory
parquet_file_path = os.path.join(current_directory, "NF-UNSW-NB15-V2.parquet")
csv_file_path = os.path.join(current_directory, "NF-UNSW-NB15-V2.csv")

# Load the Parquet file
df = pd.read_parquet(parquet_file_path)

# Save as CSV in the same directory
df.to_csv(csv_file_path, index=False)

print("Parquet file converted to CSV successfully at:", csv_file_path)
