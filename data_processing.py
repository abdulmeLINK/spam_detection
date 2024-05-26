import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def read_text_files(directory):
    texts = []
    for category in ['ham', 'spam']:
        category_path = os.path.join(directory, category)
        if os.path.exists(category_path):
            for filename in os.listdir(category_path):
                filepath = os.path.join(category_path, filename)
                with open(filepath, "r", encoding="latin-1") as file:
                    text = file.read()
                    texts.append((text, 0 if category == 'ham' else 1))
    return texts

def preprocess_text(texts):
    stop_words = set(stopwords.words('english'))
    preprocessed_texts = []
    
    for text, label in texts:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        tokens = word_tokenize(text)
        preprocessed_texts.append((tokens, label))
    
    return preprocessed_texts

def text_to_sequence(texts):
    vocab = set()
    for text, _ in texts:
        vocab.update(text)
    word_to_index = {word: idx + 1 for idx, word in enumerate(vocab)}
    sequences = [([word_to_index[word] for word in text], label) for text, label in texts]
    return sequences, word_to_index

def load_data(data_dir):
    data = []
    for enron_folder in os.listdir(data_dir):
        enron_path = os.path.join(data_dir, enron_folder)
        if os.path.isdir(enron_path):
            data.extend(read_text_files(enron_path))
    texts, labels = zip(*data)
    return list(texts), list(labels)
