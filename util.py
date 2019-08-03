import torch
import pickle
from sklearn.metrics import precision_recall_curve

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

def pr_curve_coor(y_pred, y_true):
    """Calculate precision and recall.
    Args:
        y_pred: shape=(total, num_rel).
        y_true: shape=(total, ).
    Returns:
        precision: shape=(num_point, ).
        recall: shape=(num_point, ).
    """

    y_true = F.one_hot(y_true)
    p, r, _ = precision_recall_curve(y_true.flatten(), y_pred.flatten())
    return p, r


def draw_pr_curve(model, test_tokens, test_labels, device, save_path="pr.png", batch_size=128):
    model.eval()
    data_size = test_tokens.size(0)
    y_probs = []
    for i in range(0, data_size, batch_size):
        token_ids = test_tokens[i:i + batch_size].to(device)
        with torch.no_grad():
            logits = model(token_ids)
        logits = F.softmax(logits, dim=-1)
        y_probs.append(logits)

    y_probs = torch.cat(y_probs, 0).cpu()  # .numpy()
    test_labels = test_labels.cpu()  # .numpy()
    p, r = pr_curve_coor(y_probs, test_labels)

    plt.plot(r, p, lw=1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(save_path)
