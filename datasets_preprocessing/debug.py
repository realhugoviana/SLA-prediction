import pandas as pd

# Load your original CSV file
df = pd.read_csv("datasets/best_performing_merge/Sliding_windows/Combined_3M.csv") 

print("--- Initial Data Types Check ---")
print(df.dtypes) 

print("\n--- Summary of Columns needing attention ---")
object_columns = df.select_dtypes(include=['object']).columns.tolist()
if object_columns:
    print(f"The following columns are currently flagged as 'object' type: {object_columns}")
else:
    print("No 'object' type columns found, suggesting the error might be deeper (e.g., data structure issue).")
