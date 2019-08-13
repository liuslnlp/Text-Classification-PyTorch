from typing import List, Dict
from collections import defaultdict
from pathlib import Path
from util import save_dataset, save_word_dict, save_embedding
import torch
import argparse
import nltk
import re
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Corpus(object):
    def __init__(self, input_dir):
        train_neg_dir = f'{input_dir}/train/neg'
        train_pos_dir = f'{input_dir}/train/pos'
        test_neg_dir = f'{input_dir}/test/neg'
        test_pos_dir = f'{input_dir}/test/pos'

        self.train_neg_tokens = self.load_data(train_neg_dir)
        self.train_pos_tokens = self.load_data(train_pos_dir)
        self.test_neg_tokens = self.load_data(test_neg_dir)
        self.test_pos_tokens = self.load_data(test_pos_dir)

    @staticmethod
    def load_data(dir):
        total_tokens = []
        filenames = Path(dir).glob('*.txt')
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as f:
                tokens = Corpus.tokenize(f.read())
                total_tokens.append(tokens)
        return total_tokens

    @staticmethod
    def tokenize(sent):
        sent = sent.lower().strip()
        sent = re.sub(r"<br />", r" ", sent)
        tokens = nltk.word_tokenize(sent)
        return tokens


def stat_word_freq(c:Corpus):
    """Count the frequency of every word."""
    freq_dict = defaultdict(int)
    for data in (c.train_neg_tokens, c.train_pos_tokens, c.test_neg_tokens, c.test_pos_tokens):
        for tokens in data:
            for token in tokens:
                freq_dict[token] += 1
    return freq_dict


def add_to_vocab(word, word_dict_ref):
    """Add a word to word dict."""
    if word not in word_dict_ref:
        word_dict_ref[word] = len(word_dict_ref)


def build_vocab(freq_dict:Dict[str, int], max_size:int):
    """Build word dict based on the frequency of every word."""
    word_dict = {'[PAD]': 0, '[UNK]': 1}
    sorted_items = sorted(freq_dict.items(), key=lambda t: t[1], reverse=True)[
                   :max_size]
    for word, _ in sorted_items:
        add_to_vocab(word, word_dict)
    return word_dict


@torch.jit.script
def convert_tokens_to_ids(datas: List[List[str]], word_dict: Dict[str, int], cls: int, max_seq_len: int):
    """Use @torch.jit.script to speed up."""
    total = len(datas)
    token_ids = torch.full((total, max_seq_len),
                           word_dict['[PAD]'], dtype=torch.long)
    labels = torch.full((total,), cls, dtype=torch.long)
    for i in range(total):
        seq_len = len(datas[i])
        for j in range(min(seq_len, max_seq_len)):
            token_ids[i, j] = word_dict.get(datas[i][j], word_dict['[UNK]'])
    return token_ids, labels


def create_dataset(neg, pos, word_dict, max_seq_len):
    neg_tokens, neg_labels = convert_tokens_to_ids(
        neg, word_dict, 0, max_seq_len)
    pos_tokens, pos_labels = convert_tokens_to_ids(
        pos, word_dict, 1, max_seq_len)
    tokens = torch.cat([neg_tokens, pos_tokens], 0)
    labels = torch.cat([neg_labels, pos_labels], 0)
    return tokens, labels

def load_pretrained_glove(path, freq_dict, max_size):
    word_dict = {'[PAD]': 0, '[UNK]': 1}
    embedding = []

    sorted_items = sorted(freq_dict.items(), key=lambda t: t[1], reverse=True)[:max_size]
    freq_word_set = {word for word, _ in sorted_items}

    with open(path, 'r', encoding='utf-8') as f:
        vecs = f.readlines()
        for line in vecs:
            line = line.strip().split()
            word, *vec = line
            if word in freq_word_set:
                add_to_vocab(word, word_dict)
                vec = [float(num) for num in vec]
                embedding.append(vec)

    embedding = torch.tensor(embedding, dtype=torch.float)
    embedding_dim = embedding.size(1)
    pad = torch.randn(1, embedding_dim)
    unk = torch.randn(1, embedding_dim)
    embedding = torch.cat([pad, unk, embedding], 0)
    return word_dict, embedding

if __name__ == "__main__":
    nltk.download('punkt')
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, default='aclImdb', help='Folder of original dataset.')
    parser.add_argument("-o", "--output_dir", type=str, default='data',
                        help='Folder to save the tensor format of dataset.')
    parser.add_argument("--max_seq_len", type=int, default=256, help='Max sequence length.')
    parser.add_argument("--max_vocab_size", type=int, default=30000, help='Max vocab size.')
    parser.add_argument("--glove_path", type=str, default=None, help='Pre-trained word embedding path.')

    args = parser.parse_args()

    logger.info(
        f"[input]: {args.input_dir} [output]: {args.output_dir} [max seq len]: {args.max_seq_len} [max vocab size]: {args.max_vocab_size}")

    logger.info("Loading and tokenizing...")
    c = Corpus(args.input_dir)

    logger.info("Counting word frequency...")
    freq_dict = stat_word_freq(c)
    logger.info(f"Total number of words: {len(freq_dict)}")

    logger.info("Building vocab...")
    if args.glove_path is not None:
        glove_path = Path(args.glove_path)
        word_dict, embedding = load_pretrained_glove(glove_path, freq_dict, args.max_vocab_size)
        logger.info(f"Embedding dim: {embedding.shape[1]}")
    else:
        word_dict = build_vocab(freq_dict, args.max_vocab_size)
    logger.info(f"Vocab size: {len(word_dict)}")

    logger.info("Creating train dataset...")
    train_tokens, train_labels = create_dataset(
        c.train_neg_tokens, c.train_pos_tokens, word_dict, args.max_seq_len)
    logger.info("Creating test dataset...")
    test_tokens, test_labels = create_dataset(
        c.test_neg_tokens, c.test_pos_tokens, word_dict, args.max_seq_len)

    saved_dir = Path(args.output_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving dataset and word dict[and embedding]...")
    save_word_dict(word_dict, saved_dir)
    save_dataset(train_tokens, train_labels, saved_dir, 'train')
    save_dataset(test_tokens, test_labels, saved_dir, 'test')
    if args.glove_path is not None:
        save_embedding(embedding, saved_dir)

    logger.info("All done!")
