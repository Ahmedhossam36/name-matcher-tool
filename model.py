import pandas as pd
import jellyfish
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import joblib

# Load the sample data
pairs_df = pd.read_csv("pairs_df.csv")

# Function to get Soundex encoding
def get_soundex(name):
    return jellyfish.soundex(name)

# Apply Soundex encoding to names
pairs_df['Soundex1'] = pairs_df['Name1'].apply(get_soundex)
pairs_df['Soundex2'] = pairs_df['Name2'].apply(get_soundex)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embeddings(names):
    inputs = tokenizer(names, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Batch processing for BERT embeddings
def batch_bert_embeddings(df, column_name):
    batch_size = 32  # Adjust batch size based on your GPU/CPU capabilities
    embeddings = []
    for i in range(0, len(df), batch_size):
        batch = df[column_name].iloc[i:i+batch_size].tolist()
        batch_embeddings = get_bert_embeddings(batch)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# Get BERT embeddings for both Name1 and Name2
bert_embeddings1 = batch_bert_embeddings(pairs_df, 'Name1')
bert_embeddings2 = batch_bert_embeddings(pairs_df, 'Name2')

# Combine Soundex and BERT embeddings
combined_embeddings = np.hstack([bert_embeddings1, bert_embeddings2])

# Convert combined embeddings to a list of arrays
X = combined_embeddings
y = pairs_df['Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model (Logistic Regression)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Save the trained model and tokenizer
joblib.dump(clf, 'name_matching_model.pkl')
joblib.dump(tokenizer, 'tokenizer.pkl')

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Load the trained model and tokenizer
loaded_model = joblib.load('name_matching_model.pkl')
loaded_tokenizer = joblib.load('tokenizer.pkl')

# Function to get BERT embeddings using the loaded tokenizer and model
def get_bert_embeddings_loaded(name):
    inputs = loaded_tokenizer(name, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Example usage of the loaded model and tokenizer
sample_names = ['Ahmed Hossam', 'Ahmd Hosam']
sample_embeddings = [get_bert_embeddings_loaded(name) for name in sample_names]
sample_embeddings_combined = np.hstack(sample_embeddings).reshape(1, -1)
predictions = loaded_model.predict(sample_embeddings_combined)
print(predictions)
