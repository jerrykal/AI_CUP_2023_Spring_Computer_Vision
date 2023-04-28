import os
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import HitNetDataset, balance_dataset, preprocess_data
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from model import HitNet


def argparse():
    parser = ArgumentParser()

    # Paths
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str)
    parser.add_argument("--save_path", type=str)

    # Hyperparameters
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--concat_n", type=int, default=14)
    parser.add_argument("--stepsize", type=int, default=3)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


def fix_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    train_features = torch.cat(train_features)
    train_labels = torch.cat(train_labels)

    # Create and balance training dataset
    train_dataset = HitNetDataset(train_features, train_labels)
    balance_dataset(train_dataset)

    # Create training dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

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
        val_features = torch.cat(val_features)
        val_labels = torch.cat(val_labels)

        # Create and balance validation dataset
        val_dataset = HitNetDataset(val_features, val_labels)
        balance_dataset(val_dataset)

        # Create validation dataloader
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create a HitNet model
    model = HitNet(train_features.shape[-1])
    model.to(args.device)

    # Create optimizer and loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Start training
    best_loss = None
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
                for feature, label in val_loader:
                    feature = feature.to(args.device)
                    label = label.to(args.device)

                    pred = model(feature)
                    loss = criterion(pred, label)

                    val_acc += torch.sum(
                        torch.argmax(pred, dim=1) == label.detach()
                    ).item()
                    val_loss += loss.item()

                val_acc /= len(val_dataset)
                val_loss /= len(val_loader)
                print(
                    f"[{epoch + 1:03d}/{args.num_epoch:03d}] Train Acc: {train_acc:.6f}, Loss: {train_loss:.6f} | Val Acc: {val_acc:.6f}, Loss: {val_loss:.6f}"
                )

                writer.add_scalar("Val/Acc", val_acc, epoch)
                writer.add_scalar("Val/Loss", val_loss, epoch)

                # Save best model
                if best_loss is None or val_loss < best_loss:
                    if best_loss is not None:
                        print(
                            f"val_loss improved from {best_loss:.6f} to {val_loss:.6f}, saving model to {save_path}/weights/best.pt"
                        )

                    best_loss = val_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_path, "weights", "best.pt"),
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

    # Save confusion matrix plot
    if args.val_dataset is not None:
        print("Saving confusion matrix plot")
        model.load_state_dict(torch.load(os.path.join(save_path, "weights", "best.pt")))

        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []

            for feature, label in val_loader:
                feature = feature.to(args.device)
                pred = model(feature)

                y_true += label.detach().cpu().tolist()
                y_pred += torch.argmax(pred, dim=1).detach().cpu().tolist()

            disp = ConfusionMatrixDisplay.from_predictions(
                y_true,
                y_pred,
                display_labels=["no_hit", "near_hit", "far_hit"],
            )
            disp.plot(cmap=plt.cm.Blues)
            plt.savefig(os.path.join(save_path, "confusion_matrix.png"))


if __name__ == "__main__":
    args = argparse()
    main(args)
