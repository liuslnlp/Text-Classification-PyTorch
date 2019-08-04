import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import precision_recall_curve, accuracy_score
from pathlib import Path
from util import load_dataset
import pandas as pd


def pr_curve_coor(model, test_tokens, test_labels, device, batch_size=128):
    model.to(device)
    model.eval()
    data_size = test_tokens.size(0)
    y_probs = []
    for i in range(0, data_size, batch_size):
        token_ids = test_tokens[i:i + batch_size].to(device)
        with torch.no_grad():
            logits = model(token_ids)
        logits = F.softmax(logits, dim=-1)
        y_probs.append(logits)
    y_probs = torch.cat(y_probs, 0).cpu()
    test_labels = test_labels.cpu()
    acc = accuracy_score(test_labels, y_probs.argmax(dim=-1))
    y_true = F.one_hot(test_labels)
    p, r, _ = precision_recall_curve(y_true.flatten(), y_probs.flatten())
    return p, r, acc


def evaluate(filenames, test_tokens, test_labels, device, save_path="pr.png"):
    names = []
    accs = []
    for filename in filenames:
        model = torch.load(filename)
        model_name = model.__class__.__name__
        print(f"Evaling {model_name}...")
        model_name = model_name[:-5]
        p, r, acc = pr_curve_coor(model, test_tokens, test_labels, device)
        plt.plot(r, p, lw=1, label=model_name)
        names.append(model_name)
        accs.append(acc)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(save_path)
    df = pd.DataFrame(accs, index=names, columns=['Accuracy'])
    df = df.sort_values('Accuracy')
    print(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--model_dir", type=str, default='output')
    parser.add_argument("--name", type=str, default='prcurve.png',
                        help='Path to save PR-Curve, such as XXX/XXX.png, XXX.svg, XX.jpg')
    parser.add_argument("--no_cuda", action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                                    and not args.no_cuda else 'cpu')
    test_tokens, test_labels = load_dataset(Path(args.data_dir), 'test')

    filenames = Path(args.model_dir).glob('*.pkl')
    evaluate(filenames, test_tokens, test_labels, device, args.name)


if __name__ == "__main__":
    main()
