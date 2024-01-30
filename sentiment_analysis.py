from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (replace with your own dataset)
texts = ["I loved the movie, it was fantastic!",
         "The product was a complete disappointment.",
         "This book is amazing and inspiring.",
         "The service was terrible, never using it again."]

labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Example of making predictions on new text
new_text = "I'm really happy with my purchase!"
new_text_vectorized = vectorizer.transform([new_text])
prediction = classifier.predict(new_text_vectorized)
print("Sentiment Prediction for '{}': {}".format(new_text, "Positive" if prediction[0] == 1 else "Negative"))
