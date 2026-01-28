# Task 1: Naive Bayes from scratch (No pretrained models)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset
texts = [
    "I love this product",
    "Very bad experience",
    "Amazing quality",
    "Not worth the money",
    "Excellent performance",
    "Terrible support",
    "Happy with purchase",
    "not bad at all"
    "Waste of time",
    "Good value",
    "Disappointed"
]

labels = [
    "Positive",
    "Negative",
    "Positive",
    "Negative",
    "Positive",
    "Negative",
    "Positive",
    "Positive",
    "Negative",
    "Positive",
    "Negative"
]

# Convert text data into numerical form (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, labels)

# Test the model
test_sentence = ["not bad product"]
test_vector = vectorizer.transform(test_sentence)

prediction = model.predict(test_vector)
print("Prediction:", prediction[0])
