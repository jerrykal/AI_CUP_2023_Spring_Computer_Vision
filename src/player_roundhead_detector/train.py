import os
import time
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from dataset import RoundHeadDataset, load_features, load_labels
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
# from utils import fix_seeds

from model import RoundHeadClassifier


def parse_args():
    parser = ArgumentParser()

    # Paths
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True)
    parser.add_argument("--save_path", type=str)

    # Hyperparameters
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


def main(args):
    # Fix random seeds
    # fix_seeds(0)

    # Path to save weights and summaries
    save_path = (
        args.save_path
        if args.save_path is not None
        else os.path.join(
            "saved",
            "BackhandClassifier",
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
        features = load_features(
            os.path.join(args.train_dataset, data_id, f"{data_id}_S2.csv"),
            os.path.join(args.train_dataset, data_id, f"{data_id}_trajectory.csv"),
            os.path.join(args.train_dataset, data_id, f"{data_id}_player_poses.csv"),
            os.path.join(args.train_dataset, data_id, f"{data_id}_court.csv"),
        )
        labels = load_labels(
            os.path.join(args.train_dataset, data_id, f"{data_id}_S2.csv")
        )

        train_features += features
        train_labels += labels

    # Create training dataset
    train_dataset = RoundHeadDataset(train_features, train_labels)

    # Create training data loader
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            weights=train_dataset.get_sample_weights(),
            num_samples=len(train_dataset),
            replacement=True,
        ),
    )

    # Prepare validation datas
    val_features = []
    val_labels = []
    for data_id in list(os.listdir(args.val_dataset)):
        features = load_features(
            os.path.join(args.val_dataset, data_id, f"{data_id}_S2.csv"),
            os.path.join(args.val_dataset, data_id, f"{data_id}_trajectory.csv"),
            os.path.join(args.val_dataset, data_id, f"{data_id}_player_poses.csv"),
            os.path.join(args.val_dataset, data_id, f"{data_id}_court.csv"),
        )
        labels = load_labels(
            os.path.join(args.val_dataset, data_id, f"{data_id}_S2.csv")
        )

        val_features += features
        val_labels += labels

    # Create validation dataset
    val_dataset = RoundHeadDataset(val_features, val_labels)

    # Create validation dataloader
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = RoundHeadClassifier(
        input_size=train_dataset.features.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=train_dataset.num_classes,
        dropout_prob=args.dropout_prob,
    ).to(args.device)

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0.0
    best_loss = 1e9
    for epoch in range(args.num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # Training
        model.train()
        for features, labels in train_data_loader:
            features = features.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()

            pred = model(features)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            train_acc += (pred.argmax(dim=1) == labels).sum().item()
            train_loss += loss.item()

        train_acc /= len(train_dataset)
        train_loss /= len(train_data_loader)

        writer.add_scalar("Train/Acc", train_acc, epoch)
        writer.add_scalar("Train/Loss", train_loss, epoch)

        # Validation
        model.eval()
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(args.device)
                labels = labels.to(args.device)

                pred = model(features)
                loss = criterion(pred, labels)

                val_acc += (pred.argmax(dim=1) == labels).sum().item()
                val_loss += loss.item()

        val_acc /= len(val_dataset)
        val_loss /= len(val_loader)

        writer.add_scalar("Val/Acc", val_acc, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)

        print(
            f"[{epoch + 1:03d}/{args.num_epoch:03d}] Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Saving checkpoint
        if val_acc > best_acc:
            print(f"Saving best model with val acc {val_acc:.4f}")
            best_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(save_path, "weights", "best_acc.pt")
            )

        if val_loss < best_loss:
            print(f"Saving best model with val loss {val_loss:.4f}")
            best_loss = val_loss
            torch.save(
                model.state_dict(), os.path.join(save_path, "weights", "best_loss.pt")
            )

    print("Saving last model")
    torch.save(model.state_dict(), os.path.join(save_path, "weights", "last.pt"))

    # Evaluation
    p = Path(os.path.join(save_path, "weights"))

    for model_path in p.glob("*.pt"):
        model_name = model_path.stem
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []

            for features, labels in val_loader:
                features = features.to(args.device)
                labels = labels.to(args.device)

                pred = model(features)

                y_true += labels.cpu().tolist()
                y_pred += pred.argmax(dim=1).cpu().tolist()

            print("Saving confusion matrix plot")
            disp = ConfusionMatrixDisplay.from_predictions(
                y_true,
                y_pred,
                display_labels=[
                    "is Round Head",
                    "not Round Head",
                ],
                normalize="true",
            )
            disp.plot(cmap=plt.cm.Blues)
            plt.savefig(os.path.join(save_path, f"cm_{model_name}.png"))

    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
