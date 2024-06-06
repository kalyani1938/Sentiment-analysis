import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample data
data = {
    'review': [
        "The food was great and the service was excellent!",
        "I hated the taste of the dish.",
        "The place was nice but the food was awful.",
        "Absolutely fantastic! Will come again.",
        "Not worth the money.",
        "The dessert was delicious, and the staff were friendly."
    ],
    'sentiment': ['positive', 'negative', 'negative', 'positive', 'negative', 'positive']
}

df = pd.DataFrame(data)

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize
    words = word_tokenize(text)
    # Lowercase
    words = [word.lower() for word in words]
    # Remove punctuation and stop words
    words = [word for word in words if word.isalpha() and word not in stop_words]
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

df['cleaned_review'] = df['review'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Predicting new data
new_reviews = ["The ambiance was wonderful and the food was pretty good too.", 
               "Terrible service. Will not come back.", 
               "Loved the pasta!"]

# Preprocess and vectorize new data
new_reviews_cleaned = [preprocess_text(review) for review in new_reviews]
new_reviews_vectorized = vectorizer.transform(new_reviews_cleaned)

# Predict sentiments
predictions = model.predict(new_reviews_vectorized)
print(predictions)
