"""
training/evaluate.py — Đánh giá model sau training

Chạy: python training/evaluate.py
      python training/evaluate.py --model stgcn
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score

from data_collection.dataset import VSLDataset
from preprocessing.graph_builder import build_adjacency
from models.stgcn  import STGCN
from models.ctrgcn import CTRGCN


def evaluate(model, loader, labels, device):
    model.eval().to(device)
    all_preds, all_true, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=-1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_true.extend(y.tolist())

    all_probs = np.array(all_probs)

    top1 = np.mean(np.array(all_preds) == np.array(all_true))
    top5 = top_k_accuracy_score(all_true, all_probs, k=min(5, len(labels)))

    print(f"\nTop-1 Accuracy: {top1:.2%}")
    print(f"Top-5 Accuracy: {top5:.2%}")
    print("\n" + classification_report(all_true, all_preds,
                                       target_names=labels, zero_division=0))

    # Confusion matrix
    os.makedirs("outputs", exist_ok=True)
    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(max(8, len(labels)//2), max(6, len(labels)//2)))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels,
                annot=(len(labels) <= 25), fmt='d',
                cmap='Blues', ax=ax, linewidths=0.3)
    ax.set_title(f"Confusion Matrix — {model.__class__.__name__}\n"
                 f"Top-1: {top1:.2%} | Top-5: {top5:.2%}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fname = f"outputs/confusion_{model.__class__.__name__}.png"
    plt.savefig(fname, dpi=150)
    print(f"\n✓ Đã lưu: {fname}")
    plt.close()

    return top1, top5


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',   default='both', choices=['stgcn','ctrgcn','both'])
    p.add_argument('--data',    default='data/processed')
    p.add_argument('--labels',  default='data/vsl_labels.txt')
    p.add_argument('--weights', default='weights')
    p.add_argument('--batch',   type=int, default=16)
    p.add_argument('--T',       type=int, default=64)
    args = p.parse_args()

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels    = open(args.labels, encoding='utf-8').read().splitlines()
    n_classes = len(labels)
    A         = build_adjacency()

    test_ds = VSLDataset(args.data, args.labels, split='test', T=args.T)
    loader  = DataLoader(test_ds, batch_size=args.batch, num_workers=2)

    def load(ModelClass, fname):
        m = ModelClass(A, n_classes)
        ckpt = torch.load(f"{args.weights}/{fname}", map_location=device)
        m.load_state_dict(ckpt['model_state'])
        print(f"✓ Loaded {fname} (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.2%})")
        return m

    if args.model in ('stgcn', 'both'):
        stgcn = load(STGCN, "STGCN_best.pth")
        evaluate(stgcn, loader, labels, device)

    if args.model in ('ctrgcn', 'both'):
        ctrgcn = load(CTRGCN, "CTRGCN_best.pth")
        evaluate(ctrgcn, loader, labels, device)


if __name__ == '__main__':
    main()
