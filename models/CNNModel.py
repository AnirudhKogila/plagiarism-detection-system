import torch
import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self, emb_size, max_n_sent, sent_length, n_hidden, windows):
        super(CNNModel, self).__init__()
        self.emb_size = emb_size
        self.max_n_sent = max_n_sent
        self.window_sizes = windows
        self.sent_length = sent_length
        self.n_hidden = n_hidden

        # Create a list of convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=self.emb_size, out_channels=self.n_hidden, kernel_size=h, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(in_channels=self.n_hidden, out_channels=self.n_hidden, kernel_size=h, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.max_n_sent - 1))
            for h in self.window_sizes])

    def forward(self, x1, x2):
        # Take mean across the sentence dimension
        mid1 = torch.mean(x1, 2)
        mid2 = torch.mean(x2, 2)

        # Permute the dimensions for convolution operations
        Mid1 = mid1.permute(0, 2, 1)
        Mid2 = mid2.permute(0, 2, 1)

        # Apply convolutions and max pooling on both inputs
        layer1 = [conv(Mid1) for conv in self.convs]
        layer2 = [conv(Mid2) for conv in self.convs]

        # Concatenate the outputs from different convolutions
        Out1 = torch.cat(layer1, dim=1)
        Out2 = torch.cat(layer2, dim=1)

        # Reshape for cosine similarity
        out1 = Out1.view(-1, Out1.size(1))
        out2 = Out2.view(-1, Out2.size(1))

        # Calculate cosine similarity and clamp between 0 and 1
        y_hat = nn.CosineSimilarity(dim=1)(out1, out2)
        y_hat = torch.clamp(y_hat, 0, 1)
        return y_hat
