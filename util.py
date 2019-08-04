import torch
import pickle


def save_word_dict(word_dict, saved_dir):
    with open(saved_dir / 'vocab.dict', 'wb') as f:
        pickle.dump(word_dict, f)


def save_dataset(neg, pos, saved_dir, dsettype):
    torch.save(neg, saved_dir / f'{dsettype}_tokens.pt')
    torch.save(pos, saved_dir / f'{dsettype}_labels.pt')


def load_word_dict(saved_dir):
    with open(saved_dir / 'vocab.dict', 'rb') as f:
        word_to_ix = pickle.load(f)
    return word_to_ix


def load_dataset(saved_dir, dsettype):
    tokens = torch.load(saved_dir / f'{dsettype}_tokens.pt')
    labels = torch.load(saved_dir / f'{dsettype}_labels.pt')
    return tokens, labels


def test_accuracy(model, testloader, device, batch_size=128):
    model.eval()
    data_size = len(testloader.dataset)
    total = 0
    for batch in testloader:
        token_ids, label_ids = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(token_ids)
        pred = torch.argmax(logits, dim=-1)
        total += (pred == label_ids).sum().item()
    return total / data_size
