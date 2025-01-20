from nltk import FreqDist

class FrequencyAnalyzer:
    def analyze_frequencies(self, tokens):
        return FreqDist(tokens).most_common()
