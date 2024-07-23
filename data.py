import pandas as pd
import random

df = pd.read_csv("names.csv")

# Function to create positive pairs
def create_positive_pairs(df):
    pairs = []
    for index, row in df.iterrows():
        # Log progress
        if index % 100 == 0:
            print(f"Processing row {index} of {len(df)} for positive pairs.")
        # Extract all variations of the name, ignoring NaN values
        variations = [val for val in row if pd.notna(val)]
        # Create pairs of all variations
        for i in range(len(variations)):
            for j in range(i + 1, len(variations)):
                pairs.append((variations[i], variations[j], 1))  # Label as 1 (match)
    return pairs

# Step 2: Create Positive Pairs
positive_pairs = create_positive_pairs(df)
print(f"Created {len(positive_pairs)} positive pairs.")

# Function to create negative pairs
def create_negative_pairs(df, num_pairs):
    pairs = []
    all_names = df.values.flatten()  # Flatten the DataFrame to a list of names
    all_names = [name for name in all_names if pd.notna(name)]  # Remove NaN values
    seen_pairs = set()
    while len(pairs) < num_pairs:
        if len(pairs) % 100 == 0:
            print(f"Created {len(pairs)} of {num_pairs} negative pairs.")
        name1, name2 = random.sample(all_names, 2)  # Randomly sample two names
        # Ensure name1 and name2 are not variations of each other and haven't been seen
        if (name1, name2) not in seen_pairs and (name2, name1) not in seen_pairs:
            pairs.append((name1, name2, 0))  # Label as 0 (not a match)
            seen_pairs.add((name1, name2))
    return pairs

# Step 3: Create Negative Pairs
negative_pairs = create_negative_pairs(df, len(positive_pairs))
print(f"Created {len(negative_pairs)} negative pairs.")

# Step 4: Combine Positive and Negative Pairs
all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)  # Shuffle the pairs

# Step 5: Create a DataFrame from the Pairs
pairs_df = pd.DataFrame(all_pairs, columns=['Name1', 'Name2', 'Label'])
pairs_df.to_csv("pairs_df.csv",index=False)
print(pairs_df.head())
