import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Documents\tripadvisor_hotel_reviews.csv")

# Check the columns (assume columns are 'Review' and 'Rating')
print(df.head())

# Create binary target based on rating
df['Label'] = df['Rating'].apply(lambda x: 'Good' if x >= 4 else 'Bad')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Label'], test_size=0.2, random_state=42)

# Create a pipeline for vectorization and classification
model = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Optional: evaluate the model
print("Model accuracy:", model.score(X_test, y_test))

# Function to classify new user input
def classify_review(review_text):
    prediction = model.predict([review_text])[0]
    if "not" in review_text:
         print(f"Classification: Bad")
    else:
        print(f"\nReview: {review_text}")
        print(f"Classification: {prediction}")

# Example usage
while True:
    user_input = input("\nEnter a hotel review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    classify_review(user_input)
