import os
from argparse import ArgumentParser

import pandas as pd
import torch
from dataset import RoundHeadDataset, load_features

from model import RoundHeadClassifier


def main(args):
    model = RoundHeadClassifier(
        input_size=67,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    for data_id in sorted(os.listdir(args.dataset_path)):
        print(f"Processing {data_id}... ", end="", flush=True)

        data_path = os.path.join(args.dataset_path, data_id)

        # Prepare dataset
        test_features = load_features(
            os.path.join(data_path, f"{data_id}_hit.csv"),
            os.path.join(data_path, f"{data_id}_trajectory.csv"),
            os.path.join(data_path, f"{data_id}_player_poses.csv"),
            os.path.join(data_path, f"{data_id}_court.csv"),
        )
        dataset = RoundHeadDataset(test_features)

        # Inference
        df_round_head = pd.DataFrame(columns=["ShotSeq", "RoundHead"])
        model.eval()
        with torch.no_grad():
            for i, features in enumerate(dataset):
                shot_type = model(features.unsqueeze(0).to(args.device)).argmax().item()

                df_round_head.loc[len(df_round_head)] = [i + 1, shot_type + 1]

        # Save the result
        csv_path = os.path.join(data_path, f"{data_id}_round_head.csv")
        df_round_head.to_csv(csv_path, index=False)

        print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    args = parser.parse_args()

    main(args)
