import pandas as pd
import tarfile
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import nltk
from nltk import word_tokenize, sent_tokenize

import re

# Ensure required nltk data is downloaded
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download('punkt_tab')
# Extracting the IMDB dataset
def extract_imdb_data(tar_path, extract_to="aclImdb"):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_to)

# Load the unredactor dataset
def load_unredactor_data(filepath):
    column_names = ["split", "name", "context"]

    return pd.read_csv(filepath, sep="\t", on_bad_lines='skip', names=column_names)

# Feature extraction
def extract_features(context, redacted_length):
    tokens = word_tokenize(context)
    preceding_word = tokens[tokens.index("█") - 1] if "█" in tokens and tokens.index("█") > 0 else ""
    following_word = tokens[tokens.index("█") + 1] if "█" in tokens and tokens.index("█") < len(tokens) - 1 else ""
    
    # Extract n-grams around the redacted block
    preceding_bigram = " ".join(tokens[max(0, tokens.index("█") - 2):tokens.index("█")] if "█" in tokens else [])
    following_bigram = " ".join(tokens[tokens.index("█") + 1:tokens.index("█") + 3] if "█" in tokens else [])
    
    features = {
        "redacted_length": redacted_length,
        "preceding_word": preceding_word,
        "following_word": following_word,
        "preceding_bigram": preceding_bigram,
        "following_bigram": following_bigram,
    }
    return features

# Train the logistic regression model
def train_model(train_data, vectorizer):
    match = re.search(r"█+", train_data["context"].iloc[0])
    if match:
        X_train = vectorizer.fit_transform(train_data["context"].apply(lambda x: extract_features(x, len(re.search(r"█+", x).group()))))
    else:
        X_train = vectorizer.fit_transform(train_data["context"])
    y_train = train_data["name"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, vectorizer, val_data):
    match = re.search(r"█+", val_data["context"].iloc[0])
    if match:
        X_val = vectorizer.transform(val_data["context"].apply(lambda x: extract_features(x, len(re.search(r"█+", x).group()))))
    else:
        X_val = vectorizer.transform(val_data["context"])
    # X_val = vectorizer.transform(val_data["context"].apply(lambda x: extract_features(x, len(re.search(r"█+", x).group()))))
    y_val = val_data["name"]
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    return report

# Main function to run the pipeline
def main():
    # Paths
    # tar_path = "/blue/cis6930/share/aclImdb_v1.tar.gz"
    unredactor_path = "unredactor.tsv"
    
    # Step 1: Extract IMDB data (if needed)
    # extract_imdb_data(tar_path)
    
    # Step 2: Load unredactor data
    unredactor_df = load_unredactor_data(unredactor_path)
    
    # Step 3: Split data into training and validation sets
    train_data = unredactor_df[unredactor_df["split"] == "training"]
    val_data = unredactor_df[unredactor_df["split"] == "validation"]
    
    # Step 4: Initialize the vectorizer and train the model
    vectorizer = DictVectorizer()
    model = train_model(train_data, vectorizer)
    
    # Step 5: Evaluate the model
    report = evaluate_model(model, vectorizer, val_data)
    
    # Print evaluation metrics
    print("Precision:", report["weighted avg"]["precision"])
    print("Recall:", report["weighted avg"]["recall"])
    print("F1-Score:", report["weighted avg"]["f1-score"])
    
    # Example: Predict a name
    test_context = "This movie █████████ was incredible."
    test_features = vectorizer.transform([extract_features(test_context, len(re.search(r"█+", test_context).group()))])
    prediction = model.predict(test_features)
    print("Predicted Name:", prediction[0])

# Run the pipeline
if __name__ == "__main__":
    main()
