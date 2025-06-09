import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and preprocess pq_matrix.xlsx
pq_matrix = pd.read_excel(r"C:\Users\HUAWEI\Desktop\25春\线代\project\pq_matrix.xlsx", header=None)
pq_matrix_processed = (pq_matrix - 3) / 2  # Normalize into [-1, 1] range

# Step 2: Load and preprocess tag_matrix.xlsx
tag_matrix_raw = pd.read_excel(r"C:\Users\HUAWEI\Desktop\25春\线代\project\similarity_matrix.xlsx", header=None)
tag_matrix = tag_matrix_raw.iloc[1:, 1:]  # Drop first row and first column
tag_matrix.columns = tag_matrix_raw.iloc[0, 1:]  # Reset column names
tag_matrix.index = tag_matrix_raw.iloc[1:, 0]    # Reset index names
tag_matrix = tag_matrix.astype(float)  # Ensure numeric

# Step 3: Check shape consistency
print("Shape of pq_matrix_processed:", pq_matrix_processed.shape)
print("Shape of tag_matrix:", tag_matrix.shape)
assert pq_matrix_processed.shape == tag_matrix.shape, "Matrices must be the same shape!"

# Convert to numpy arrays
pq_flat = pq_matrix_processed.to_numpy().flatten()
tag_flat = tag_matrix.to_numpy().flatten()

# Step 4: Compute similarity metrics
pearson_corr, _ = pearsonr(pq_flat, tag_flat)
cos_sim = cosine_similarity(pq_flat.reshape(1, -1), tag_flat.reshape(1, -1))[0][0]
mse = np.mean((pq_flat - tag_flat) ** 2)

# Step 5: Output results
print("===== Matrix Similarity Report =====")
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Cosine Similarity:   {cos_sim:.4f}")
print(f"Mean Squared Error:  {mse:.4f}")
print("----- Interpretation -----")

# Pearson
if pearson_corr > 0.85:
    print("✅ Very strong linear correlation between matrices.")
elif pearson_corr > 0.7:
    print("✅ Strong correlation between matrices.")
elif pearson_corr > 0.5:
    print("⚠️  Moderate correlation — consider refining the model.")
else:
    print("❌ Weak correlation — matrices may represent different structure.")

# Cosine
if cos_sim > 0.95:
    print("✅ Cosine similarity is excellent.")
elif cos_sim > 0.85:
    print("✅ Cosine similarity is strong.")
elif cos_sim > 0.7:
    print("⚠️  Cosine similarity is moderate.")
else:
    print("❌ Low cosine similarity — direction mismatch.")

# MSE
if mse < 0.05:
    print("✅ Very close in numeric values (low error).")
elif mse < 0.15:
    print("✅ Acceptable numeric deviation.")
else:
    print("❌ Large numeric difference.")


