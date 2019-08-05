class Config(object):
    """Model config.
    Common to all models.
    Args:
        vocab: int, vocab size.
        embed_dim: int, word embedding dimension.
        padding_id: int, used to padding sequence less than max length.
        hidden_dim: int, hidden dimension of LSTM/CNN.
        tag_dim: int, target size, in IMDB dataset is 2.
        max_seq_len: int, max sequence length.
        dropout: float, dropout probability of multi-layer LSTM and some CNN models.
        n_layer: int, number of LSTM layers.
        attn_dim: int, context vector dimension(for attention model).
        n_block: int, number of DPCNN blocks.
    """
    def __init__(self, vocab, embed_dim, padding_id, hidden_dim, tag_dim=2, dropout=0.2, n_layer=2, attn_dim=128,
                 max_seq_len=512, n_block=5):
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.padding_id = padding_id
        self.hidden_dim = hidden_dim
        self.tag_dim = tag_dim
        self.max_seq_len = max_seq_len
        self.n_layer = n_layer
        self.dropout = dropout
        self.attn_dim = attn_dim
        self.n_block = n_block
