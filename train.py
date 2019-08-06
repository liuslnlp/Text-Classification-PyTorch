import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import logging
from pathlib import Path
from network import Config, BiLSTMAttnModel, CNNAttnModel, CNNModel, LSTMModel, TextCNNModel, DPCNNModel, RCNNModel
import shutil
from util import load_word_dict, load_dataset, test_accuracy

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ModelCache(object):
    """
    Cache model for every epoch and save the best model at end.
    """

    def __init__(self, model_ref, model_name, cache_dir):
        self.model = model_ref
        self.name = model_name
        self.cache_dir = cache_dir

    def cache_model(self, epoch):
        """Save current model to cache dir."""
        filename = f"{self.name}_{epoch}.pkl"
        filename = self.cache_dir / filename
        torch.save(self.model, filename)

    def clear_cache(self):
        """Clear cache folder."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)

    def move_best_cache(self, scores, target_dir, verbose=True):
        """
        Find the model with the best score,
        and save it to output folder.
        """
        best_score, epoch = torch.max(scores, 0)
        if verbose:
            print(f"Model name: {self.name}, Best epoch: {epoch.item()}, Best score: {best_score.item()}")
        filename = f"{self.name}_{epoch.item()}.pkl"
        src_filename = self.cache_dir / filename
        dst_filename = target_dir / filename
        shutil.copyfile(src_filename, dst_filename)


class Trainer(object):
    def __init__(self, model, trainloader, testloader, device, args):
        self.model = model
        model.to(device)

        self.args = args

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model.__class__.__name__
        self.mche = ModelCache(model, self.model_name, cache_dir)
        self.logger = logging.getLogger(self.model_name)

        self.optimizer = optim.Adam(model.parameters(), lr=args.lr)
        self.loss_func = nn.CrossEntropyLoss()
        self.device = device
        self.trainloader = trainloader
        self.testloader = testloader
        self.test_acc_lst = torch.zeros(args.epochs)

    def fit(self):
        args = self.args
        model = self.model
        device = self.device
        logger = self.logger
        for epoch in range(args.epochs):
            logger.info(f"***** Epoch {epoch} *****")
            model.train()
            for step, batch in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                input_ids, targets = tuple(t.to(device) for t in batch)
                logits = model(input_ids)
                loss = self.loss_func(logits, targets)
                loss.backward()
                self.optimizer.step()
                if step % args.print_step == 0:
                    batch_acc = (logits.argmax(-1) == targets).float().mean()
                    logger.info(
                        f"[epoch]: {epoch}, [batch]: {step}, [loss]: {loss.item():.6}, [acc]: {batch_acc:.6}")
            acc = test_accuracy(model, self.testloader, device)
            self.mche.cache_model(epoch)
            self.test_acc_lst[epoch] = acc
            logger.info(f"[Test Accuracy]: {acc:.6}")
        self.mche.move_best_cache(self.test_acc_lst, self.output_dir)
        self.mche.clear_cache()


def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    model_map={'1': CNNModel, '2':TextCNNModel, '3':DPCNNModel, '4':CNNAttnModel, '5':LSTMModel, '6':BiLSTMAttnModel, '7':RCNNModel}

    parser.add_argument("-i", "--input_dir", type=str, default='data', help='Dir of input data.')
    parser.add_argument("-o", "--output_dir", type=str, default='output', help='Dir to save trained model.')
    parser.add_argument("--cache_dir", type=str, default='cache', help='Temporary folder to save trained model.')
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--print_step", type=int, default=5)

    parser.add_argument("--max_seq_len", type=int, default=256, help='Max sequence length.')
    parser.add_argument("--embed_dim", type=int, default=256, help='Word embedding dim.')
    parser.add_argument("--hidden_dim", type=int, default=128, help='Hidden dim.')
    parser.add_argument("--attn_dim", type=int, default=128, help='Context vector dim(for attention model).')
    parser.add_argument("--tag_dim", type=int, default=2, help='Target size.')
    parser.add_argument("--n_layer", type=int, default=2, help='Number of layers(for LSTM).')
    parser.add_argument("--n_block", type=int, default=5, help='Number of blocks(for DPCNN).')
    parser.add_argument("--dropout", type=float, default=0.2, help='Dropout probability.')
    parser.add_argument("--no_cuda", action='store_true', help='Whether not use CUDA.')
    parser.add_argument("--alg", type=str, default='123456', help=f'Model to train. {model_map}')



    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                                    and not args.no_cuda else 'cpu')
    input_dir = Path(args.input_dir)
    word_to_ix = load_word_dict(input_dir)
    vocab_size = len(word_to_ix)
    padding_id = word_to_ix['[PAD]']
    config = Config(vocab=vocab_size, embed_dim=args.embed_dim, padding_id=padding_id,
                    hidden_dim=args.hidden_dim, tag_dim=args.tag_dim, n_layer=args.n_layer,
                    attn_dim=args.attn_dim, max_seq_len=args.max_seq_len, n_block=args.n_block, dropout=args.dropout)

    logger.info(f"***** Loading train data. *****")
    train_tokens, train_labels = load_dataset(input_dir, 'train')
    logger.info(f"***** Loading test data. *****")
    test_tokens, test_labels = load_dataset(input_dir, 'test')

    logger.info(f"***** Building data loader. *****")
    trainset = TensorDataset(train_tokens, train_labels)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True)
    testset = TensorDataset(test_tokens, test_labels)
    testloader = DataLoader(testset, batch_size=args.batch_size)

    

    logger.info(f"***** Initializing models. *****")
    models = [v(config) for k, v in model_map.items() if k in args.alg]
    logger.info(f"***** Training. *****")
    for model in models:
        trainer = Trainer(model, trainloader, testloader, device, args)
        logger.info(f"Training {trainer.model_name}...")
        trainer.fit()

    logger.info(f"***** All Done. *****")

if __name__ == "__main__":
    main()
