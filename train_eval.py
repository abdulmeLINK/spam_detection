import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def train_model(model, train_loader, criterion, optimizer, device, vocab_size, num_epochs):
    print("Starting training...")
    train_loss_list = []
    train_acc_list = []

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (texts, labels) in enumerate(train_loader):
            print(f"Starting batch {i+1}...")
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
        print(f"Epoch {epoch+1} completed. Loss: {train_loss / total_train}, Accuracy: {correct_train / total_train}")

    print("Training completed.")
    return train_loss_list, train_acc_list

def evaluate_model(model, test_loader, device, vocab_size):
    print("Starting evaluation...")
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for i, (texts, labels) in enumerate(test_loader):
            print(f"Evaluating batch {i+1}...")
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            predicted = torch.round(outputs.squeeze())
            correct_test += (predicted == labels).sum().item()
            total_test += texts.size(0)

    print(f"Evaluation completed. Accuracy: {correct_test / total_test}")
    return correct_test / total_test

def plot_results(train_loss_list, train_acc_list, test_acc_list):
    plt.plot(train_loss_list, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(test_acc_list, label="Test Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.show()

def compute_confusion_matrix(model, test_loader, y_test, device):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            predicted = torch.round(outputs.squeeze()).tolist()
            y_pred.extend(predicted)

    cm = confusion_matrix(y_test, y_pred)
    return cm
