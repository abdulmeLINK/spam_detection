import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from utils import load_data, preprocess_text, text_to_sequence, create_datasets, TextDataset, LSTMModel, train_model, evaluate_model, save_model, load_model, plot_results, compute_confusion_matrix

def main(train_model_flag):
    # Load environment variables from .env file
    load_dotenv()

    # Read parameters from environment variables
    EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 100))
    HIDDEN_DIM = int(os.getenv('HIDDEN_DIM', 256))
    OUTPUT_DIM = int(os.getenv('OUTPUT_DIM', 1))
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 10))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 16))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    print("Loading and preprocessing data...")
    if not os.path.exists('./processed_data/vocab.txt') or \
    not os.path.exists('./processed_data/ham.txt') or \
    not os.path.exists('./processed_data/spam.txt') or \
    not os.path.exists('./processed_data/train.txt') or \
    not os.path.exists('./processed_data/test.txt'):
        texts, labels = load_data('./enron')
        cleaned_texts, labels = preprocess_text(texts, labels)
        sequences, vocab = text_to_sequence(cleaned_texts)
        train_sequences, test_sequences, y_train, y_test, max_length = create_datasets(sequences, labels)

        # Save processed data
        with open('./processed_data/ham.txt', 'w') as f:
            for text, label in zip(cleaned_texts, labels):
                if label == 0:
                    f.write(text + '\n')

        with open('./processed_data/spam.txt', 'w') as f:
            for text, label in zip(cleaned_texts, labels):
                if label == 1:
                    f.write(text + '\n')

        with open('./processed_data/train.txt', 'w') as f:
            for text in train_sequences:
                f.write(' '.join(map(str, text)) + '\n')

        with open('./processed_data/test.txt', 'w') as f:
            for text in test_sequences:
                f.write(' '.join(map(str, text)) + '\n')

        # Save vocab
        with open('./processed_data/vocab.txt', 'w') as f:
            f.write('\n'.join(vocab))
    else:
        print("Processed data already exists. Loading from files...")
        with open('./processed_data/train.txt', 'r') as f:
            train_sequences = [list(map(int, line.strip().split())) for line in f]
        with open('./processed_data/test.txt', 'r') as f:
            test_sequences = [list(map(int, line.strip().split())) for line in f]
        with open('./processed_data/ham.txt', 'r') as f:
            y_train = [0] * len(f.readlines())
        with open('./processed_data/spam.txt', 'r') as f:
            y_train += [1] * len(f.readlines())
        y_test = y_train[int(len(y_train) * 0.8):]
        y_train = y_train[:int(len(y_train) * 0.8)]
        max_length = max(max(len(seq) for seq in train_sequences), max(len(seq) for seq in test_sequences))

        # Load vocab
        with open('./processed_data/vocab.txt', 'r') as f:
            vocab = f.read().splitlines()

    # Create datasets and data loaders
    train_dataset = TextDataset(train_sequences, y_train, max_length)
    test_dataset = TextDataset(test_sequences, y_test, max_length)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model, criterion, and optimizer
    model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, len(vocab) + 1).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if train_model_flag or not os.path.exists('lstm_model.pth'):
        # Train the model
        train_loss_list, train_acc_list = train_model(model, train_loader, criterion, optimizer, DEVICE, NUM_EPOCHS)
        
        # Evaluate the model on the test set
        test_acc = evaluate_model(model, test_loader, DEVICE)
        test_acc_list = [test_acc] * NUM_EPOCHS
        
        # Save the model
        save_model(model, 'lstm_model.pth')

        # Plot results
        plot_results(train_loss_list, train_acc_list, test_acc_list)
        
        # Compute confusion matrix
        compute_confusion_matrix(model, test_loader, DEVICE)
    else:
        # Load the pre-trained model
        load_model(model, 'lstm_model.pth', DEVICE)

        # Evaluate the model on the test set
        test_acc = evaluate_model(model, test_loader, DEVICE)

        # Compute confusion matrix
        compute_confusion_matrix(model, test_loader, DEVICE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_model', action='store_true', help='Train the model')
    args = parser.parse_args()

    main(args.train_model)