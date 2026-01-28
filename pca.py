import numpy as np

# -------------------------------
# Step 1: Input Dataset
# -------------------------------
# Rows = Students, Columns = Attributes
X = np.array([
    [78, 85, 80, 82],
    [65, 70, 68, 72],
    [90, 92, 88, 91],
    [72, 75, 70, 74],
    [85, 88, 84, 86]
])

print("Original Dataset:\n", X)

# -------------------------------
# Step 2: Mean of each attribute
# -------------------------------
mean = np.mean(X, axis=0)
print("\nMean of each attribute:\n", mean)

# -------------------------------
# Step 3: Mean Centering
# -------------------------------
X_centered = X - mean
print("\nMean-centered data:\n", X_centered)

# -------------------------------
# Step 4: Covariance Matrix
# -------------------------------
cov_matrix = np.cov(X_centered.T)
print("\nCovariance Matrix:\n", cov_matrix)

# -------------------------------
# Step 5: Eigenvalues & Eigenvectors
# -------------------------------
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# -------------------------------
# Step 6: Sort Eigenvalues (Descending)
# -------------------------------
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nSorted Eigenvalues:\n", eigenvalues)
print("\nSorted Eigenvectors:\n", eigenvectors)

# -------------------------------
# Step 7: Percentage of Variance
# -------------------------------
variance_ratio = eigenvalues / np.sum(eigenvalues)
print("\nVariance Ratio:\n", variance_ratio)

# -------------------------------
# Step 8: Select First Principal Component
# -------------------------------
pc1 = eigenvectors[:, 0]
print("\nFirst Principal Component (PC1):\n", pc1)

# -------------------------------
# Step 9: Project Data onto PC1
# -------------------------------
reduced_data = np.dot(X_centered, pc1)
print("\nReduced 1-D Dataset (PC1 scores):\n", reduced_data)