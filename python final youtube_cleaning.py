import pandas as pd

# ✅ Load the Excel file
file_path = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\final_cleaned_youtube_comments.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# ✅ Remove rows where 'translated_comment' is empty or NaN
df_cleaned = df.dropna(subset=['translated_comment'])

# ✅ Remove rows where 'translated_comment' contains only numbers
df_cleaned = df_cleaned[~df_cleaned['translated_comment'].astype(str).str.strip().str.isdigit()]

# ✅ Save the final cleaned dataset
output_file = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\best_cleaned_youtube_comments.xlsx"
df_cleaned.to_excel(output_file, index=False)

print(f"✅ Cleaning completed! Cleaned file saved as: {output_file}")
