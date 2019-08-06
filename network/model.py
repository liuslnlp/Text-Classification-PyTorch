import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import LSTMLayer, CNNLayer, AttnLayer, CNNBlock

__all__ = ["LSTMModel", "CNNModel", "TextCNNModel",
           "DPCNNModel", "BiLSTMAttnModel", "CNNAttnModel"]


def cal_seq_len(x, max_seq_len, padding_id):
    return max_seq_len - (x == padding_id).sum(dim=-1)


class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab, config.embed_dim, padding_idx=config.padding_id)
        self.lstm = LSTMLayer(
            config.embed_dim, config.hidden_dim, config.n_layer, dropout=config.dropout)
        self.linear = nn.Linear(config.hidden_dim, config.tag_dim)
        self.padding_id = config.padding_id

    def forward(self, x):
        """
        x : shape=(batch_size, max_len)
        """
        batch_size, max_len = x.shape
        lens = cal_seq_len(x, max_len, self.padding_id)

        x = self.embedding(x)
        # x =  self.lstm(x, lens)
        x = self.lstm(x)
        # x = x[:, -1, :]
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
        """
        Args:
            x: shape=(batch_size, max_seq_len).
        """
        # x.shape=(batch_size, max_seq_len, embed_dim)
        x = self.embedding(x)
        # x.shape=(batch_size, max_seq_len, hidden_dim)
        x = self.conv(x)
        # x.shape=(batch_size, hidden_dim)
        x, _ = torch.max(x, dim=1)
        x = F.relu(x)
        # out.shape=(batch_size, tag_dim)
        out = self.linear(x)
        return out


class TextCNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab, config.embed_dim, config.padding_id)
        # self.conv1 = CNNLayer(config.embed_dim, 2, 2, 0)
        # self.conv2 = CNNLayer(config.embed_dim, 2, 3, 0)
        # self.conv3 = CNNLayer(config.embed_dim, 2, 4, 0)
        self.conv1 = CNNLayer(config.embed_dim, config.hidden_dim, 2, 0)
        self.conv2 = CNNLayer(config.embed_dim, config.hidden_dim, 3, 0)
        self.conv3 = CNNLayer(config.embed_dim, config.hidden_dim, 4, 0)
        # self.linear = nn.Linear(6, config.tag_dim)
        self.linear = nn.Linear(config.hidden_dim * 3, config.tag_dim)

    def forward(self, x):
        """
        Args:
            x: shape=(batch_size, max_seq_len).
        """
        x = self.embedding(x)
        # feat1.shape = (batch_size, len-1, hidden_dim)
        feat1 = F.relu(self.conv1(x))
        # feat1.shape = (batch_size, len-2, hidden_dim)
        feat2 = F.relu(self.conv2(x))
        # feat1.shape = (batch_size, len-3, hidden_dim)
        feat3 = F.relu(self.conv3(x))

        # feat1.shape = (batch_size, hidden_dim)
        feat1, _ = torch.max(feat1, dim=1)
        # feat2.shape = (batch_size, hidden_dim)
        feat2, _ = torch.max(feat2, dim=1)
        # feat3.shape = (batch_size, hidden_dim)
        feat3, _ = torch.max(feat3, dim=1)

        # feat1.shape = (batch_size, hidden_dim *　３)
        feat = torch.cat([feat1, feat2, feat3], -1)
        out = self.linear(feat)
        return out


class DPCNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab, config.embed_dim, config.padding_id)
        hidden_dim = 250
        self.region_embedding = CNNLayer(config.embed_dim, hidden_dim, 3, 1)
        self.conv1 = CNNLayer(hidden_dim, hidden_dim, 3, 1)
        self.conv2 = CNNLayer(hidden_dim, hidden_dim, 3, 1)
        self.blocks = nn.Sequential(*[CNNBlock()
                                      for _ in range(config.n_block)])
        self.linear = nn.Linear(hidden_dim, config.tag_dim)

    def forward(self, x):
        """
        Args:
            x: shape=(batch_size, max_seq_len).
        """
        # x．shape=(batch_size, max_len, embed_dim)
        emb = self.embedding(x)
        # x．shape=(batch_size, max_len, 250)
        emb = self.region_embedding(emb)
        # x．shape=(batch_size, max_len, 250)
        x = F.relu(emb)
        x = self.conv1(x)
        # x．shape=(batch_size, max_len, 250)
        x = F.relu(x)
        x = self.conv2(x)
        # x．shape=(batch_size, max_len / 2^n_block, 250)
        x = self.blocks(x + emb)
        # x．shape=(batch_size, 250)
        x, _ = torch.max(x, dim=1)
        # out.shape=(batch_size, tag_dim)
        out = self.linear(x)
        return out


class BiLSTMAttnModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab, config.embed_dim, config.padding_id)
        self.bilstm = LSTMLayer(
            config.embed_dim, config.hidden_dim // 2, 1, bi=True)
        self.attn = AttnLayer(config.hidden_dim, config.attn_dim)
        self.linear = nn.Linear(config.hidden_dim, config.tag_dim)
        self.padding_id = config.padding_id

    def forward(self, x):
        """
        x : shape=(batch_size, max_len)
        """
        x = self.embedding(x)
        x = self.bilstm(x)
        # x = self.bilstm(x, lens)
        x = self.attn(x)
        out = self.linear(x)
        return out


class CNNAttnModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab, config.embed_dim, config.padding_id)
        self.cnn = CNNLayer(config.embed_dim, config.hidden_dim)
        self.attn = AttnLayer(config.hidden_dim, config.attn_dim)
        self.linear = nn.Linear(config.hidden_dim, config.tag_dim)

    def forward(self, x):
        """
        Args:
            x: shape=(batch_size, max_seq_len).
        """
        x = self.embedding(x)
        x = self.cnn(x)
        x = F.relu(x)
        x = self.attn(x)
        out = self.linear(x)
        return out
