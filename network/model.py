import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import LSTMLayer, CNNLayer, AttnLayer

class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab, config.embed_dim, padding_idx=config.padding_id)
        self.lstm = LSTMLayer(config.embed_dim, config.hidden_dim, config.n_layer, dropout=config.dropout)
        self.linear = nn.Linear(config.hidden_dim, config.tag_dim)

    def forward(self, x):
        """
        x : shape=(batch_size, max_len)
        """
        x = self.embedding(x)
        x =  self.lstm(x)
        x = x[:, -1, :]
        out = self.linear(x)
        return out

class LSTM2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab, config.embed_dim, padding_idx=config.padding_id)
        self.lstm = LSTMLayer(config.embed_dim, config.hidden_dim, config.n_layer, dropout=config.dropout)
        self.linear = nn.Linear(config.hidden_dim, config.tag_dim)
        self.padding_id = config.padding_id

    def forward(self, x):
        """
        x : shape=(batch_size, max_len)
        """
        batch_size, max_len = x.shape
        lens = max_len - (x == self.padding_id).sum(dim=-1)
        x = self.embedding(x)
        x =  self.lstm(x)
        x = x[torch.arange(batch_size), lens - 1, :]
        out = self.linear(x)
        return out


class CNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab, config.embed_dim, padding_idx=config.padding_id)
        self.conv = CNNLayer(config.embed_dim, config.hidden_dim)
        self.linear = nn.Linear(config.hidden_dim, config.tag_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv(x)
        x, _ = torch.max(x, dim=1)
        x = F.relu(x)
        out = self.linear(x)
        return out

class BiLSTMAttnModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab, config.embed_dim, config.padding_id)
        self.bilstm = LSTMLayer(config.embed_dim, config.hidden_dim // 2, 1, bi=True)
        self.attn = AttnLayer(config.hidden_dim, config.attn_dim)
        self.linear = nn.Linear(config.hidden_dim, config.tag_dim)

    def forward(self, x):
        """
        x : shape=(batch_size, max_len)
        """
        x = self.embedding(x)
        x = self.bilstm(x)
        x = self.attn(x)
        out = self.linear(x)
        return out

class CNNAttnModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab, config.embed_dim, config.padding_id)
        self.cnn = CNNLayer(config.embed_dim, config.hidden_dim)
        self.attn = AttnLayer(config.hidden_dim, config.attn_dim)
        self.linear = nn.Linear(config.hidden_dim, config.tag_dim)

    def forward(self, x):
        """
        x : shape=(batch_size, max_len)
        """
        x = self.embedding(x)
        x = self.cnn(x)
        x = F.relu(x)
        x = self.attn(x)
        out = self.linear(x)
        return out