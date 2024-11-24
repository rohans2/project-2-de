from nltk import word_tokenize, sent_tokenize
def extract_features(context, redacted_length):
    tokens = word_tokenize(context)
    preceding_word = tokens[tokens.index("█") - 1] if "█" in tokens and tokens.index("█") > 0 else ""
    following_word = tokens[tokens.index("█") + 1] if "█" in tokens and tokens.index("█") < len(tokens) - 1 else ""
    
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
