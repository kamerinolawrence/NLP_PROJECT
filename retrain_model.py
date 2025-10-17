import pandas as pd
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load the CSV dataset
df = pd.read_csv("judge-1377884607_tweet_product_company.csv", encoding = 'latin1')

# Rename column for simplicity
df = df.rename(columns={
    'is_there_an_emotion_directed_at_a_brand_or_product': 'sentiment'
})

# Define text cleaner
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)

    # Handle negotiations
    negation_patterns = {
        r'\b(am|are) not a fan\b': 'dislike',
        r'\b(do|did|does) not like\b': 'dislike',
        r'\b(do|did|does|is) not good\b': 'bad',
        r'\b(do|did|does|is) not great\b': 'poor',
        r'\b(am|are) not happy\b': 'unhappy',
        r'\b(do|did|does) not worth\b': 'worthless',
        r'\b(do|did|does) not recommend\b': 'avoid',
        r'\b(is|was|are) not bad\b': 'good',
        r'\b(is|was|are) not terrible\b': 'good',
        r'\b(am|are|do|did|does) not unhappy\b': 'happy',
        r'\b(am|is|are) not poor\b': 'good',
        r'\b(do|did|does) not want\b': 'dislike',
        r'\b(do|did|does) not need\b': 'dislike',
        r'\b(do|did|does) not like\b': 'dislike',

        # Positive negations
        r'\b(is|are) not bad\b': 'good',
        r'\b(is|are) not terrible\b': 'good'
    }
    for pattern, replacement in negation_patterns.items():
        text = re.sub(pattern, replacement, text)

    # General negation
    text = re.sub(r'\b(do|did|does|am|is|are|was|were) not (\w+)\b', r'not_\2', text)

    
    #  normalize repeated letters (soooo -> soo)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Strip extra spaces
    text = text.strip()
    
    return text


# Apply cleaning
df = df.dropna(subset=['tweet_text']) 
df['cleaned_text'] = df['tweet_text'].astype(str).apply(clean_tweet)
df['label'] = df['sentiment']  #

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label'], test_size=0.2, random_state=42
)

# Rebuild and train the pipeline
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        preprocessor=None,
        max_features=5000,
        ngram_range=(1,2)
    )),
    ('svm', LinearSVC(class_weight='balanced', random_state=42))
])

svm_pipeline.fit(X_train, y_train)

# Save the retrained pipeline
joblib.dump(svm_pipeline, "sentiment_pipeline.pkl")

print("SVM Sentiment Model retrained and saved as sentiment_pipeline.pkl")