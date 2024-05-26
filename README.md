# Spam Detection

This project is a spam detection system that uses machine learning to classify emails as either "spam" or "ham" (not spam).

## Project Structure

- `data_processing.py`: Contains code for processing the email data.
- `dataset.py`: Defines the dataset used for training and testing the model.
- `model.py`: Defines the machine learning model.
- `train_eval.py`: Contains code for training the model and evaluating its performance.
- `main.py`: The main entry point of the application.
- `enron/`: Contains the Enron email dataset, divided into multiple subfolders.

## Setup

1. Clone this repository.
2. Install the required Python packages: `pip install -r requirements.txt`
3. Run the main script: `python main.py`

## Dataset

The dataset used in this project is the Enron email dataset. It is divided into multiple subfolders, each containing "ham" and "spam" emails.

## Model

The machine learning model used in this project is defined in `model.py`. It is trained using the script in `train_eval.py`.

## Evaluation

The performance of the model is evaluated in terms of accuracy and confusion matrix, as implemented in `train_eval.py`.