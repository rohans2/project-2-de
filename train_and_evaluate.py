from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from extract import extract_features
import re

def train_model(train_data, vectorizer):
    
    X_train = vectorizer.fit_transform(train_data["context"].apply(lambda x: extract_features(x, len(re.search(r"█+", x).group()) if re.search(r"█+", x) else 0)))
    y_train = train_data["name"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, vectorizer, val_data):
    
    X_val = vectorizer.transform(val_data["context"].apply(lambda x: extract_features(x, len(re.search(r"█+", x).group()) if re.search(r"█+", x) else 0)))
    
    y_val = val_data["name"]
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    return report