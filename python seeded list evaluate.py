import pickle
import pandas as pd

# === STEP 1: Load Preprocessed Comments ===
file_path = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\texts_filtered.pkl"

with open(file_path, "rb") as f:
    texts = pickle.load(f)

# === STEP 2: Flatten Texts for Matching ===
flat_corpus = " ".join([" ".join(comment) for comment in texts])

# === STEP 3: Define DOI Seed Word Lists ===
doi_seed_words = {
    "Relative Advantage": [
        "better", "safer", "cheaper", "efficient", "convenient", "superior",
        "improved", "faster", "advanced", "innovative", "smart", "upgraded", "beneficial"
    ],
    "Compatibility": [
        "match", "suitable", "aligns", "lifestyle", "habit", "need", "fit",
        "accustomed", "consistent", "routine", "familiar", "relevant", "user-friendly"
    ],
    "Complexity": [
        "confusing", "complicated", "difficult", "hard", "complex", "unclear", "risky",
        "dangerous", "unreliable", "malfunction", "glitchy", "uncertain"
    ],
    "Trialability": [
        "try", "test", "demo", "experiment", "sample", "trial", "temporary",
        "pilot", "preview", "simulate", "experience", "first-time", "ride"
    ],
    "Observability": [
        "see", "watch", "observe", "visible", "show", "display", "public",
        "coverage", "proven", "noticeable", "share", "evidence", "social"
    ]
}

# === STEP 4: Perform Coverage Check ===
coverage_results = {}
for dimension, words in doi_seed_words.items():
    match_count = sum(1 for word in words if word in flat_corpus)
    coverage = match_count / len(words)
    coverage_results[dimension] = round(coverage * 100, 2)

# === STEP 5: Display Results ===
coverage_df = pd.DataFrame.from_dict(coverage_results, orient='index', columns=["Coverage (%)"])
print("\n📊 DOI Seed Word Coverage in Your Comments:")
print(coverage_df)
