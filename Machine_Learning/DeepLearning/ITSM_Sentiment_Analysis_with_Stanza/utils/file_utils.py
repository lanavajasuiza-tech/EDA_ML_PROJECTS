import json

def save_to_file(data, filepath):
    """
    Guarda datos en un archivo JSON.
    """
    with open(filepath, 'w') as file:
        json.dump(data, file)

def read_from_file(filepath):
    """
    Lee datos desde un archivo JSON.
    """
    with open(filepath, 'r') as file:
        return json.load(file)
