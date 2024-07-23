import pandas as pd
import numpy as np
import jellyfish
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import joblib
from concurrent.futures import ThreadPoolExecutor

# Step 1: Data Preparation
# Load the sample data
pairs_df = pd.read_csv("pairs_df.csv")

# Function to get Soundex encoding
def get_soundex(name):
    return jellyfish.soundex(name)

# Function to get Metaphone encoding
def get_metaphone(name):
    return jellyfish.metaphone(name)

# Apply Soundex and Metaphone encoding to names
pairs_df['Soundex1'] = pairs_df['Name1'].apply(get_soundex)
pairs_df['Soundex2'] = pairs_df['Name2'].apply(get_soundex)
pairs_df['Metaphone1'] = pairs_df['Name1'].apply(get_metaphone)
pairs_df['Metaphone2'] = pairs_df['Name2'].apply(get_metaphone)

# Encode labels as required by BERT (e.g., 0 and 1)
pairs_df['Label'] = pairs_df['Label'].astype(int)

# Split the data into training and test sets
train_df, test_df = train_test_split(pairs_df, test_size=0.2, random_state=42)

# Step 2: Fine-Tuning BERT
# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the input
train_encodings = tokenizer(list(train_df['Name1'] + " " + train_df['Name2']), truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(list(test_df['Name1'] + " " + test_df['Name2']), truncation=True, padding=True, max_length=64)

# Convert labels to tensors
train_labels = torch.tensor(train_df['Label'].values)
test_labels = torch.tensor(test_df['Label'].values)

# Create a dataset class
class NameMatchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = NameMatchDataset(train_encodings, train_labels)
test_dataset = NameMatchDataset(test_encodings, test_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Fine-tune the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('fine_tuned_bert')
tokenizer.save_pretrained('fine_tuned_bert')

# Step 3: Extract BERT Embeddings
# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('fine_tuned_bert')
tokenizer = BertTokenizer.from_pretrained('fine_tuned_bert')

# Function to get BERT embeddings
def get_bert_embeddings(name):
    inputs = tokenizer(name, return_tensors='pt', padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model.bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

# Parallelize embedding extraction
def parallel_bert_embeddings(names):
    with ThreadPoolExecutor(max_workers=4) as executor:
        embeddings = list(executor.map(get_bert_embeddings, names))
    return np.array(embeddings)

# Get BERT embeddings for both Name1 and Name2
bert_embeddings1 = parallel_bert_embeddings(pairs_df['Name1'].tolist())
bert_embeddings2 = parallel_bert_embeddings(pairs_df['Name2'].tolist())

# Combine Soundex, Metaphone, and BERT embeddings
combined_embeddings = np.hstack([bert_embeddings1, bert_embeddings2])

# Convert combined embeddings to a list of arrays
X = combined_embeddings
y = pairs_df['Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a new model on the combined features
clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, 'name_matching_model_with_bert.pkl')

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
