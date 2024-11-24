import pandas as pd
import tarfile
from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import train_test_split
import nltk
# from nltk import word_tokenize, sent_tokenize
from train_and_evaluate import train_model, evaluate_model
from generate import generate_submission


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download('punkt_tab')
def extract_imdb_data(tar_path, extract_to="aclImdb"):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_to)

def load_unredactor_data(filepath):
    column_names = ["split", "name", "context"]

    return pd.read_csv(filepath, sep="\t", on_bad_lines='skip', names=column_names)


def main():
    unredactor_path = "unredactor.tsv"
    test_file = "test.tsv"
    
    # extract_imdb_data(tar_path)
    
    unredactor_df = load_unredactor_data(unredactor_path)
    
    train_data = unredactor_df[unredactor_df["split"] == "training"]
    val_data = unredactor_df[unredactor_df["split"] == "validation"]
    
    vectorizer = DictVectorizer()
    model = train_model(train_data, vectorizer)
    
    report = evaluate_model(model, vectorizer, val_data)
    
    print("Precision:", report["weighted avg"]["precision"])
    print("Recall:", report["weighted avg"]["recall"])
    print("F1-Score:", report["weighted avg"]["f1-score"])
    
    generate_submission(test_file, model, vectorizer)
    # test_context = "This movie █████████ was incredible."
    # test_features = vectorizer.transform([extract_features(test_context, len(re.search(r"█+", test_context).group()))])
    # prediction = model.predict(test_features)
    # print("Predicted Name:", prediction[0])

if __name__ == "__main__":
    main()
