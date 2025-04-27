# Trip-Advisor-Hotel-Reviews
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

df = pd.read_csv(r"C:\Users\Dell\Desktop\chithra\trip.csv")
print(df.head())

df['Label'] = df['Rating'].apply(lambda x: 'Good' if x >= 4 else 'Bad')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Label'], test_size=0.2, random_state=42)

model = Pipeline([
    ('vectorizer',TfidfVectorizer(ngram_range=(1,2))),
    ('classifier', MultinomialNB())])

    model.fit(X_train, y_train)

    print("Model accuracy:", model.score(X_test, y_test))

    def classify_review(review_text):
    prediction = model.predict([review_text])[0]
    print(f"\nReview: {review_text}")
    print(f"Classification: {prediction}")

    while True:
    user_input = input("\nEnter a hotel review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    classify_review(user_input)# Trip-Advisor-Hotel-Reviews
