import math
from transformers import AutoModel
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):
    def __init__(self, emb_size, max_n_sent, n_hidden, HP_SBERT_AHEADS, n_layers, dropout):
        super().__init__()
        self.model_type = 'Transformer'
        self.emb_size = emb_size
        self.pos_encoder = PositionalEncoding(emb_size, max_n_sent, dropout)

        encoder_layers = TransformerEncoderLayer(emb_size, HP_SBERT_AHEADS, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.CosineSimilarity(dim = 1)
        self.pooling = nn.MaxPool1d(kernel_size = max_n_sent)

    def forward(self, x1, x2) -> Tensor:

        mid1 = torch.mean(x1, 2)      
        mid2 = torch.mean(x2, 2)

        Mid1 = mid1.permute(1, 0, 2)    
        Mid2 = mid2.permute(1, 0, 2)

        Mid1 = self.pos_encoder(Mid1)
        Mid2 = self.pos_encoder(Mid2)

        output1 = self.transformer_encoder(Mid1)   
        output2 = self.transformer_encoder(Mid2)

        output1 = output1.permute(1, 2, 0)      
        output2 = output2.permute(1, 2, 0)
        
        Out1 = self.pooling(output1) 
        Out2 = self.pooling(output2)

        out1 = Out1.view(-1, Out1.size(1)) 
        out2 = Out2.view(-1, Out2.size(1))

        f_output = self.decoder(out1, out2)
        f_output = torch.clamp(f_output, 0, 1)
        return f_output

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_n_sent, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_n_sent).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        pe = torch.zeros(max_n_sent, 1, emb_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
