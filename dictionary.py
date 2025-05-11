# Dictionary tool logic will go here
import requests

def define_word(word):
    """Use an API to define a word."""
    response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
    if response.status_code == 200:
        data = response.json()
        return data[0]['meanings'][0]['definitions'][0]['definition']
    else:
        return f"Definition not found for {word}."
