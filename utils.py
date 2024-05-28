import os
import re
import string
import nltk
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

# Utility function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = get_tokenizer("basic_english")(text)  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Load data from the Enron dataset
def load_data(directory):
    texts = []
    labels = []
    for subdir in os.listdir(directory):
        ham_dir = os.path.join(directory, subdir, 'ham')
        spam_dir = os.path.join(directory, subdir, 'spam')
        for filename in os.listdir(ham_dir):
            with open(os.path.join(ham_dir, filename), 'r', encoding='latin-1') as file:
                texts.append(file.read())
                labels.append(0)
        for filename in os.listdir(spam_dir):
            with open(os.path.join(spam_dir, filename), 'r', encoding='latin-1') as file:
                texts.append(file.read())
                labels.append(1)
    return texts, labels

# Preprocess text data
def preprocess_text(texts, labels):
    cleaned_texts = [clean_text(text) for text in texts]
    return cleaned_texts, labels

# Convert text to sequence of integers
def text_to_sequence(texts):
    tokenizer = get_tokenizer("basic_english")
    vocab = set()
    for text in texts:
        vocab.update(tokenizer(text))
    vocab = {word: idx+1 for idx, word in enumerate(vocab)}  # word index starts from 1

    sequences = []
    for text in texts:
        sequence = [vocab[word] for word in tokenizer(text) if word in vocab]
        sequences.append(sequence)

    return sequences, vocab

# Pad sequences to the same length
def pad_sequence(sequence, max_length):
    padded_sequence = np.zeros(max_length, dtype=int)
    padded_sequence[:len(sequence)] = sequence[:max_length]
    return padded_sequence

# Create datasets for training and testing
def create_datasets(sequences, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=test_size, stratify=labels, random_state=random_state)
    max_length = max(len(seq) for seq in X_train) + 10  # adding padding buffer

    train_sequences = [pad_sequence(seq, max_length) for seq in X_train]
    test_sequences = [pad_sequence(seq, max_length) for seq in X_test]

    return train_sequences, test_sequences, y_train, y_test, max_length

# Save model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load model
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

# Plot results
def plot_results(train_loss_list, train_acc_list, test_acc_list):
    epochs = range(1, len(train_loss_list) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, label="Train Accuracy")
    plt.plot(epochs, test_acc_list, label="Test Accuracy", linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Test Accuracy")
    plt.legend()

    plt.show()

# Compute confusion matrix
def compute_confusion_matrix(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            predicted = torch.round(outputs.squeeze())
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    return cm

# Dataset class for handling the text data
class TextDataset(Dataset):
    def __init__(self, texts, labels, max_length):
        self.texts = [pad_sequence(text, max_length) for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

# LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, vocab_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(self.dropout(last_hidden_state))
        return self.sigmoid(output)

from tqdm import tqdm

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    print("Starting training...")
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # Specify the device for computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * texts.size(0)
            predicted = torch.round(outputs.squeeze())
            correct_train += (predicted == labels).sum().item()
            total_train += texts.size(0)

        train_loss_list.append(train_loss / total_train)
        train_acc_list.append(correct_train / total_train)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss / total_train:.4f}, Accuracy: {correct_train / total_train:.4f}")

    print("Training completed.")
    return train_loss_list, train_acc_list

# Evaluate the model
def evaluate_model(model, test_loader):
    print("Starting evaluation...")
    model.eval()
    correct_test = 0
    total_test = 0

    # Specify the device for computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            predicted = torch.round(outputs.squeeze())
            correct_test += (predicted == labels).sum().item()
            total_test += texts.size(0)

    accuracy = correct_test / total_test
    print(f"Evaluation completed. Accuracy: {accuracy:.4f}")
    return accuracy