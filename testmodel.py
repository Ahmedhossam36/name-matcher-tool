
import pandas as pd
import jellyfish
from metaphone import doublemetaphone
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

pairs_df = pd.read_csv("pairs_df.csv")
# Function to apply phonetic encoding
def phonetic_encoding(name):
    primary, secondary = doublemetaphone(name)
    return primary

# Apply phonetic encoding to both columns
pairs_df['Name1_Encoded'] = pairs_df['Name1'].apply(phonetic_encoding)
pairs_df['Name2_Encoded'] = pairs_df['Name2'].apply(phonetic_encoding)

# Combine encoded names into one column for vectorization
pairs_df['Combined_Encoded'] = pairs_df['Name1_Encoded'] + ' ' + pairs_df['Name2_Encoded']

# Convert the phonetic codes to a numerical representation
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
X = vectorizer.fit_transform(pairs_df['Combined_Encoded'])
y = pairs_df['Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
joblib.dump(model, 'name_matching_model.pkl')

# To save the vectorizer as well
joblib.dump(vectorizer, 'vectorizer.pkl')