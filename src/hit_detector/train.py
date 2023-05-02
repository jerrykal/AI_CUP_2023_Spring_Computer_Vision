import os
import time
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dataset import HitNetDataset, preprocess_data
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from utils import eval_metrics, fix_seeds

from model import HitNet


def parse_args():
    parser = ArgumentParser()

    # Paths
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str)
    parser.add_argument("--save_path", type=str)

    # Hyperparameters
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--concat_n", type=int, default=14)
    parser.add_argument("--stepsize", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


def main(args):
    # Fix random seeds
    fix_seeds(0)

    # Path to save weights and summaries
    save_path = (
        args.save_path
        if args.save_path is not None
        else os.path.join(
            "saved",
            "HitNet",
            f"train_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
        )
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, "weights"))

    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(save_path, "summary"))

    # Prepare training datas
    train_features = []
    train_labels = []
    for data_id in list(os.listdir(args.train_dataset)):
        features, labels = preprocess_data(
            args.train_dataset,
            data_id,
            args.concat_n,
            args.stepsize,
        )
        train_features.append(features)
        train_labels.append(labels)

    train_features = np.concatenate(train_features)
    train_labels = np.concatenate(train_labels)

    # Create training dataset
    train_dataset = HitNetDataset(train_features, train_labels)

    # Create training dataloader with weighted random sampler
    # See https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
    # for more info
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            weights=train_dataset.get_sample_weights(),
            num_samples=len(train_dataset),
            replacement=True,
        ),
    )

    # Prepare validation datas
    if args.val_dataset is not None:
        val_features = []
        val_labels = []
        for data_id in list(os.listdir(args.val_dataset)):
            features, labels = preprocess_data(
                args.val_dataset,
                data_id,
                args.concat_n,
                args.stepsize,
            )
            val_features.append(features)
            val_labels.append(labels)

        val_features = np.concatenate(val_features)
        val_labels = np.concatenate(val_labels)

        # Create validation dataset
        val_dataset = HitNetDataset(val_features, val_labels)

        # Create validation dataloader
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create a HitNet model
    model = HitNet(train_features.shape[-1])
    model.to(args.device)

    # Create optimizer and loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0
    best_val_loss = None
    best_acc = 0
    best_recall = 0
    best_prec = 0
    best_f1 = 0

    # Start training
    for epoch in range(args.num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # Training
        model.train()
        for feature, label in train_loader:
            feature = feature.to(args.device)
            label = label.to(args.device)

            optimizer.zero_grad()
            pred = model(feature)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            train_acc += torch.sum(torch.argmax(pred, dim=1) == label.detach()).item()
            train_loss += loss.item()

        train_acc /= len(train_dataset)
        train_loss /= len(train_loader)

        writer.add_scalar("Train/Acc", train_acc, epoch)
        writer.add_scalar("Train/Loss", train_loss, epoch)

        # Validation
        if args.val_dataset is not None:
            model.eval()
            with torch.no_grad():
                y_true = []
                y_pred = []

                for feature, label in val_loader:
                    feature = feature.to(args.device)
                    label = label.to(args.device)

                    pred = model(feature)
                    loss = criterion(pred, label)

                    y_true += label.detach().cpu().numpy().tolist()
                    y_pred += torch.argmax(pred, dim=1).detach().cpu().numpy().tolist()

                    val_acc += torch.sum(
                        torch.argmax(pred, dim=1) == label.detach()
                    ).item()
                    val_loss += loss.item()

                # Evaluate model
                val_acc /= len(val_dataset)
                val_loss /= len(val_loader)

                # NOTE: The accuracy here is different from validation accuracy
                acc, recall, prec, f1 = eval_metrics(y_true, y_pred)

                # Logging
                print(
                    f"[{epoch + 1:03d}/{args.num_epoch:03d}] Train Acc: {train_acc:.6f}, Loss: {train_loss:.6f} | "
                    f"Val Acc: {val_acc:.6f}, Loss: {val_loss:.6f} | "
                    f"Eval Acc: {acc:.6f}, Recall: {recall:.6f}, Precision: {prec:.6f}, F1: {f1:.6f}"
                )
                writer.add_scalar("Val/Acc", val_acc, epoch)
                writer.add_scalar("Val/Loss", val_loss, epoch)
                writer.add_scalar("Eval/Acc", acc, epoch)
                writer.add_scalar("Eval/Recall", recall, epoch)
                writer.add_scalar("Eval/Precision", prec, epoch)
                writer.add_scalar("Eval/F1", f1, epoch)

                # Saving checkpoints
                if val_acc > best_val_acc:
                    print(
                        f"Saving model with highest validation accuracy {val_acc:.6f}"
                    )

                    best_val_acc = val_acc
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_path, "weights", "best_val_acc.pt"),
                    )
                if best_val_loss is None or val_loss < best_val_loss:
                    print(f"Saving model with lowest validation loss {val_loss:.6f}")

                    best_val_loss = val_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_path, "weights", "best_val_loss.pt"),
                    )
                if acc > best_acc:
                    print(f"Saving model with highest evaluation accuracy {acc:.6f}")

                    best_acc = acc
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_path, "weights", "best_acc.pt"),
                    )
                if recall > best_recall:
                    print(f"Saving model with highest evaluation recall {recall:.6f}")

                    best_recall = recall
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_path, "weights", "best_recall.pt"),
                    )
                if prec > best_prec:
                    print(f"Saving model with highest evaluation precision {prec:.6f}")

                    best_prec = prec
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_path, "weights", "best_prec.pt"),
                    )
                if f1 > best_f1:
                    print(f"Saving model with highest evaluation F1 score {f1:.6f}")

                    best_f1 = f1
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_path, "weights", "best_f1.pt"),
                    )
        else:
            print(
                f"[{epoch + 1:03d}/{args.num_epoch:03d}] Train Acc: {train_acc:.6f}, Loss: {train_loss:.6f}"
            )

    # Save last model
    print("Saving last model")
    torch.save(
        model.state_dict(),
        os.path.join(save_path, "weights", "last.pt"),
    )

    # Evaluation
    print("Evaluating models...")
    if args.val_dataset is not None:
        df_eval = pd.DataFrame(
            columns=["Name", "Accuracy", "Recall", "Precision", "F1"]
        )

        p = Path(os.path.join(save_path, "weights"))
        for model_path in p.glob("*.pt"):
            model_name = model_path.stem
            model.load_state_dict(torch.load(model_path))
            model.eval()
            with torch.no_grad():
                y_true = []
                y_pred = []

                for feature, label in val_loader:
                    feature = feature.to(args.device)
                    pred = model(feature)

                    y_true += label.detach().cpu().tolist()
                    y_pred += torch.argmax(pred, dim=1).detach().cpu().tolist()

                # Save evaluation metrics
                acc, recall, prec, f1 = eval_metrics(y_true, y_pred)
                df_eval.loc[len(df_eval)] = {
                    "Name": model_name,
                    "Accuracy": acc,
                    "Recall": recall,
                    "Precision": prec,
                    "F1": f1,
                }

                print(f"Saving confusion matrix plot for {model_path}")
                disp = ConfusionMatrixDisplay.from_predictions(
                    y_true,
                    y_pred,
                    display_labels=["no_hit", "near_hit", "far_hit"],
                    normalize="true",
                )
                disp.plot(cmap=plt.cm.Blues)
                plt.savefig(os.path.join(save_path, f"cm_{model_name}.png"))

        print("Saving evaluation metrics")
        df_eval.to_csv(os.path.join(save_path, "eval_metrics.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
