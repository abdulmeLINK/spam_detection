import os
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from data_processing import load_data, preprocess_text, text_to_sequence
from dataset import TextDataset
from model import LSTMModel
from train_eval import train_model, evaluate_model, plot_results, compute_confusion_matrix
import torch.nn as nn
import torch.optim as optim
# Load environment variables from .env file
load_dotenv()

# Read parameters from environment variables
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM'))
HIDDEN_DIM = int(os.getenv('HIDDEN_DIM'))
OUTPUT_DIM = int(os.getenv('OUTPUT_DIM'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

def main():
    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument("--train_model", action="store_true", help="Train the model if this flag is provided.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "enron"
    texts, labels = load_data(data_dir)

    texts_cleaned = preprocess_text(list(zip(texts, labels)))
    sequences, vocab = text_to_sequence(texts_cleaned)

    X_train, X_test, y_train, y_test = train_test_split([seq for seq, label in sequences], [label for seq, label in sequences], test_size=0.2, stratify=labels, random_state=42)

    max_length = max(len(seq) for seq in X_train) + 10
    train_dataset = TextDataset(X_train, y_train, max_length)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TextDataset(X_test, y_test, max_length)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, len(vocab) + 1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if args.train_model:
    train_loss_list, train_acc_list = train_model(model, train_loader, criterion, optimizer, device, len(vocab) + 1, NUM_EPOCHS)
    test_acc_list = [evaluate_model(model, test_loader, device, len(vocab) + 1) for _ in range(NUM_EPOCHS)]
    
    plot_results(train_loss_list, train_acc_list, test_acc_list)
    
    cm = compute_confusion_matrix(model, test_loader, y_test, device)
    print("Confusion Matrix:")
    print(cm)

    # Save the model
    torch.save(model.state_dict(), 'model.pth')
else:
    test_acc = evaluate_model(model, test_loader, device, len(vocab) + 1)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
