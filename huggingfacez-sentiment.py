from transformers import pipeline

# Load pretrained sentiment model
sentiment_analyzer = pipeline("sentiment-analysis")

# Test text
text = "The product is very bad and I am disappointed."

# Predict sentiment
result = sentiment_analyzer(text)

print(result)
