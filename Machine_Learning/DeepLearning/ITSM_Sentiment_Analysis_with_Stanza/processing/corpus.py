from nltk.corpus import CategorizedPlaintextCorpusReader

class CorpusLoader:
    def __init__(self, path):
        self.path = path
        self.reader = None
        self.corpus = None
        self.categories = None

    def load_corpus(self):
        self.reader = CategorizedPlaintextCorpusReader(
            self.path, r'.*\.txt',
            cat_pattern=r'(\w+)/*', encoding='cp1252'
        )
        self.categories = self.reader.categories()
        self.corpus = {cat: self.reader.fileids(categories=cat) for cat in self.categories}
        return self.reader, self.corpus, self.categories
