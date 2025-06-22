import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

text = "Bhai ab sab thik ho jaaye"
tokens = word_tokenize(text)
print(tokens)
