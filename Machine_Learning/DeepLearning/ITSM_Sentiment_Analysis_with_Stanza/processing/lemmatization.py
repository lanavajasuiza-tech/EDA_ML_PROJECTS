from nltk.stem import WordNetLemmatizer

class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_text(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]
