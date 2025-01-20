from nltk.tokenize import word_tokenize
import nltk
import re

class Tokenizer:
    def __init__(self):
        """
        Inicializa el tokenizador y verifica recursos necesarios.
        """
        print("Tokenizer inicializado.")
        self.ensure_resources()

    def ensure_resources(self):
        """
        Verifica y descarga los recursos necesarios para la tokenización.
        """
        try:
            nltk.data.find('tokenizers/punkt')
            print("✔ Recurso 'punkt' encontrado.")
        except LookupError:
            print("✘ Recurso 'punkt' no encontrado. Intentando descargar...")
            try:
                nltk.download('punkt')
                print("✔ Recurso 'punkt' descargado correctamente.")
            except Exception as e:
                print(f"✘ Error al descargar 'punkt': {e}")

    def tokenize_text(self, text):
        """
        Tokeniza el texto dado en palabras.
        Args:
            text (str): Texto a tokenizar.
        Returns:
            list: Lista de tokens.
        """
        if not isinstance(text, str):
            print("Advertencia: El texto proporcionado no es válido.")
            return []

        try:
            return word_tokenize(text)
        except Exception as e:
            print(f"Error durante la tokenización con NLTK: {e}")
            print("Usando tokenización personalizada.")
            return re.findall(r'\b\w+\b', text.lower())
