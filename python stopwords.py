import pickle

# ✅ Save `texts` after preprocessing
texts_file = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\texts.pkl"

with open(texts_file, "wb") as f:
    pickle.dump(texts, f)

print(f"✅ 'texts' has been saved to {texts_file}!")
