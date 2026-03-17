# ==============================
# 1️⃣ Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA


# ==============================
# 2️⃣ Load Dataset
# ==============================
data = pd.read_csv("spam_dataset.csv")

print("Columns:", data.columns)

data = data[['label', 'text']]

# Convert labels if needed
if data['label'].dtype == 'object':
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# ==============================
# 3️⃣ Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    data['text'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# ==============================
# 4️⃣ TF-IDF Vectorization
# ==============================
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==============================
# 5️⃣ Train SVM Model
# ==============================
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# ==============================
# 6️⃣ Prediction & Evaluation
# ==============================
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==============================
# 7️⃣ Test Custom Email
# ==============================
sample = ["You have won $1000! Click here to claim your prize now"]
sample_tfidf = vectorizer.transform(sample)

prediction = model.predict(sample_tfidf)

if prediction[0] == 1:
    print("Spam 🚨")
else:
    print("Not Spam ✅")


# ==============================
# 8️⃣ Confusion Matrix Visualization
# ==============================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Ham", "Spam"])
plt.yticks([0, 1], ["Ham", "Spam"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()


# ==============================
# 9️⃣ Accuracy Bar Graph
# ==============================
accuracy = accuracy_score(y_test, y_pred)

plt.figure()
plt.bar(["Accuracy"], [accuracy])
plt.ylim(0, 1)
plt.title("Model Accuracy")
plt.ylabel("Accuracy Score")
plt.show()


# ==============================
# 🔟 SVM Decision Boundary Diagram (Using PCA)
# ==============================

# Reduce TF-IDF to 2D
pca = PCA(n_components=2)

X_train_dense = X_train_tfidf.toarray()
X_reduced = pca.fit_transform(X_train_dense)

# Train new SVM on reduced data
svm_2d = SVC(kernel='linear')
svm_2d.fit(X_reduced, y_train)

# Create mesh grid
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure()

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_train)

# Highlight support vectors
plt.scatter(svm_2d.support_vectors_[:, 0],
            svm_2d.support_vectors_[:, 1])

plt.title("SVM Decision Boundary (PCA Reduced Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()
