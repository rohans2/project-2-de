from extract import extract_features
import pandas as pd
import re
def generate_submission(test_file, model, vectorizer, output_file="submission.tsv"):
    test_df = pd.read_csv(test_file, sep="\t", names=["id", "context"])
    test_df["name"] = test_df["context"].apply(
        lambda x: model.predict(vectorizer.transform([extract_features(x, len(re.search(r"█+", x).group()) if re.search(r"█+", x) else 0)]))[0]
    )
    test_df[["id", "name"]].to_csv(output_file, sep="\t", index=False, header=False)
    print(f"Submission file '{output_file}' generated successfully!")