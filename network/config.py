class Config(object):
    def __init__(self, vocab, embed_dim, padding_id, hidden_dim, tag_dim=2, dropout=0.2, n_layer=2, attn_dim=128, max_seq_len=512):
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.padding_id = padding_id
        self.hidden_dim = hidden_dim
        self.tag_dim = tag_dim
        self.max_seq_len = max_seq_len
        self.n_layer = n_layer
        self.dropout = dropout
        self.attn_dim = attn_dim