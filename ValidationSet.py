import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch

pd.options.mode.chained_assignment = None
plt.switch_backend('agg')

if __name__ == '__main__':
    train_df = torch.load('train_tok.pt')  # Load the tokenized training data
    labels = torch.load('train_labels.pt')  # Load the training labels

    print('.......Validation Set Generating........')
    X = range(len(train_df))  # Create a range of indices for the training data

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.1, random_state=42)

    # Create DataFrames to hold the training and validation data
    train_text_df = pd.DataFrame(index=range(len(X_train)), columns=['text', 'text_b'], dtype=object)
    val_text_df = pd.DataFrame(index=range(len(X_val)), columns=['text', 'text_b'], dtype=object)
    train_label = []  # List to store training labels
    val_label = []  # List to store validation labels

    for i in tqdm(range(len(X_train))):
        train_text_df['text'][i] = train_df['text'][X_train[i]]
        train_text_df['text_b'][i] = train_df['text_b'][X_train[i]]
        train_label.append(labels[X_train[i]])

    for i in tqdm(range(len(X_val))):
        val_text_df['text'][i] = train_df['text'][X_val[i]]
        val_text_df['text_b'][i] = train_df['text_b'][X_val[i]]
        val_label.append(labels[X_val[i]])

    print('.......Validation Set Saving........')

    # Save the training and validation data and labels
    torch.save(train_text_df, 'train_tok.pt')
    torch.save(val_text_df, 'val_tok.pt')
    torch.save(train_label, 'train_labels.pt')
    torch.save(val_label, 'val_labels.pt')
    print('=== END ===')
