import os
from argparse import ArgumentParser

import pandas as pd
import torch
from dataset import ShotTypeDataset, load_features

from model import ShotTypeClassifier


def main(args):
    model = ShotTypeClassifier(
        input_size=11,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    for data_id in sorted(os.listdir(args.dataset_path)):
        if int(data_id) > 2:
            break
        print(f"Processing {data_id}... ", end="", flush=True)

        data_path = os.path.join(args.dataset_path, data_id)

        # Prepare dataset
        test_features = load_features(
            os.path.join(data_path, f"{data_id}_S2.csv"),
            os.path.join(data_path, f"{data_id}_trajectory.csv"),
            os.path.join(data_path, f"{data_id}_court.csv"),
        )
        dataset = ShotTypeDataset(test_features)

        # Inference
        df_shot_type = pd.DataFrame(columns=["ShotSeq", "BallType"])
        model.eval()
        with torch.no_grad():
            for i, features in enumerate(dataset):
                shot_type = model(features.unsqueeze(0).to(args.device)).argmax().item()

                df_shot_type.loc[len(df_shot_type)] = [i + 1, shot_type + 1]

        # Save the result
        csv_path = os.path.join(data_path, f"{data_id}_shot_type.csv")
        df_shot_type.to_csv(csv_path, index=False)

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
