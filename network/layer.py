import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLayer(nn.Module):
    """Conv1d layer.
    nn.Conv1d layer require the input shape is (batch_size, in_channels, length),
    however, our input shape is (batch_size, length, in_channels), so we need to
    transpose our input data into (B, C, L_in) and send it to conv layer, and
    then transpose the conv output into (B, L_out, C).
    """
    def __init__(self, in_dim, out_dim, win=3, pad=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_dim,
                              out_channels=out_dim, kernel_size=win, padding=pad)

    def forward(self, x):
        """
        Args:
            x: shape=(batch_size, max_seq_len, input_dim)
        """
        # x.shape=(batch_size, input_dim, max_seq_len)
        x = x.permute(0, 2, 1)
        # self.conv(x).shape=(batch_size, hidden_dim, max_seq_len)
        # out.shape=(batch_size, max_seq_len, hidden_dim)
        out = self.conv(x).permute(0, 2, 1)
        return out


class MaxPool1d(nn.Module):
    def __init__(self, win=2, stride=None, pad=0):
        super().__init__()
        self.pooling = nn.MaxPool1d(kernel_size=win, stride=stride, padding=pad)

    def forward(self, x):
        """
        Args:
            x: shape=(batch_size, max_seq_len, dim)
        """
        # x.shape=(batch_size, dim, max_seq_len)
        x = x.permute(0, 2, 1)
        x = self.pooling(x).permute(0, 2, 1)
        return x


class CNNBlock(nn.Module):
    """Block of DPCNN.
    """

    def __init__(self):
        super().__init__()
        hidden_dim = 250
        self.pooling = MaxPool1d(3, 2, 1)
        self.conv1 = CNNLayer(hidden_dim, hidden_dim, 3, 1)
        self.conv2 = CNNLayer(hidden_dim, hidden_dim, 3, 1)

    def forward(self, x):
        res = self.pooling(x)
        # pre-activation
        x = self.conv1(F.relu(res))
        x = self.conv2(F.relu(x))
        # res sum.
        out = x + res
        return out


class LSTMLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer, dropout=0, bi=False):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, out_dim, batch_first=True, num_layers=n_layer, dropout=dropout, bidirectional=bi)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class AttnLayer(nn.Module):
    """Attention layer.
    w is context vector.
    Formula:
        $$
        v_i=tanh(Wh_i+b)\\
        \alpha_i = v_i^Tw\\
        \alpha_i = softmax(\alpha_i)\\
        Vec = \sum_0^L \alpha_ih_i
        $$
    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.weight = nn.Linear(hidden_dim, attn_dim)
        self.context = nn.Parameter(torch.randn(attn_dim))

    def forward(self, x):
        """
        x: shape=(batch_size, max_len, hidden_dim)
        """
        # query.shape=(batch_size, max_len, attn_dim)
        query = self.weight(x).tanh()
        # scores.shape=(batch_size, max_len)
        scores = torch.einsum('bld,d->bl', query, self.context)
        scores = F.softmax(scores, dim=-1)
        # attn_vec.shape=(batch_size, hidden_dim)
        attn_vec = torch.einsum('bl,blh->bh', scores, x)
        return attn_vec
