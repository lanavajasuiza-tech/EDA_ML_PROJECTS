import re

class TextCleaner:
    def __init__(self, stopwords=None):
        self.stopwords = stopwords

    def to_lowercase(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def remove_extra_whitespace(self, text):
        return ' '.join(text.split())

    def remove_stopwords(self, text):
        if not self.stopwords:
            return text
        words = text.split()
        cleaned_words = [word for word in words if word not in self.stopwords]
        return ' '.join(cleaned_words)

    def remove_html_tags(self, text):
        return re.sub(r'<.*?>', '', text)

    def clean_text(self, text):
        text = self.to_lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_extra_whitespace(text)
        if self.stopwords:
            text = self.remove_stopwords(text)
        return text
