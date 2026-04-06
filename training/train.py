"""
training/train.py — Script chính để train ST-GCN và CTR-GCN

Chạy: python training/train.py
      python training/train.py --model stgcn --epochs 80 --batch 16
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from data_collection.dataset import VSLDataset
from preprocessing.graph_builder import build_adjacency
from models.stgcn import STGCN
from models.ctrgcn import CTRGCN
from training.trainer import VSLTrainer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',   default='both',   choices=['stgcn','ctrgcn','both'])
    p.add_argument('--data',    default='data/processed')
    p.add_argument('--labels',  default='data/vsl_labels.txt')
    p.add_argument('--weights', default='weights')
    p.add_argument('--epochs',  type=int,   default=80)
    p.add_argument('--batch',   type=int,   default=16)
    p.add_argument('--lr',      type=float, default=1e-3)
    p.add_argument('--T',       type=int,   default=64)
    return p.parse_args()


def main():
    args   = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    labels    = open(args.labels, encoding='utf-8').read().splitlines()
    n_classes = len(labels)
    A         = build_adjacency()

    train_ds = VSLDataset(args.data, args.labels, split='train', T=args.T)
    val_ds   = VSLDataset(args.data, args.labels, split='val',   T=args.T)

    if args.model in ('stgcn', 'both'):
        print("\n" + "="*60)
        print("  ST-GCN")
        stgcn = STGCN(A, n_classes)
        VSLTrainer(
            stgcn, train_ds, val_ds, n_classes,
            lr=args.lr, epochs=args.epochs,
            batch_size=args.batch, device=device,
            save_dir=args.weights
        ).train()

    if args.model in ('ctrgcn', 'both'):
        print("\n" + "="*60)
        print("  CTR-GCN")
        ctrgcn = CTRGCN(A, n_classes)
        VSLTrainer(
            ctrgcn, train_ds, val_ds, n_classes,
            lr=args.lr * 0.5, epochs=args.epochs,
            batch_size=args.batch, device=device,
            save_dir=args.weights
        ).train()

    print("\n✓ Training hoàn tất!")
    print(f"  Weights lưu tại: {args.weights}/")


if __name__ == '__main__':
    main()
