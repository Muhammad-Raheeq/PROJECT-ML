import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import re
#  Load Dataset
file_path = "News_Category_Dataset_v3.csv"
# Try robust loading
df = pd.read_csv(
    file_path,
    delimiter=",",              
    quoting=3,               
    on_bad_lines="skip",        
    engine="python",            
    encoding="utf-8"            
)
print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())
# Use 'headline' + 'short_description' as input text
df["text"] = df["headline"].fillna("") + " " + df["short_description"].fillna("")
X = df["text"]
y = df["category"]
# Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)  # remove punctuation/numbers
    return text
X = X.apply(clean_text)
# Remove rare categories 
category_counts = df['category'].value_counts()
valid_categories = category_counts[category_counts > 1].index
df = df[df['category'].isin(valid_categories)]
X = df["text"]
y = df["category"]
# Train/test split with stratify (now safe)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#  Vectorization 
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
#  Train Classifier
model = LogisticRegression(max_iter=300)
model.fit(X_train_tfidf, y_train)
#  Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#  User Input Prediction
def predict_category(article_text):
    cleaned = clean_text(article_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return pred
sample_article = input("Enter the titel of headline ")
print("Predicted Category:", predict_category(sample_article))
