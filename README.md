# AI CUP 2023 Spring Computer Vision

Source code for AICUP 2023 Spring Competition: [Teaching Computer to Watch Badminton Matches - Taiwan's first competition combining AI and sports](https://aidea-web.tw/topic/cbea66cc-a993-4be8-933d-1aa9779001f8)

## Environment Setup

The code has been run and tested on **Ubuntu 22.04 LTS** with **CUDA 1.18** and **CudNN 8.6.0.163**.

Environment are created using **Conda** with **Python 3.9**.

```shell
conda create -n aicup python=3.9
conda activate aicup
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tensorflow opencv-python scikit-learn parse piexif pandas matplotlib tqdm json-tricks openmim mmdet
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.0.0"
```

## Data

Download the dataset from the competition official and put the data under the `data` directory. Rename the `data/val` directory to `data/test`. Then manually/programmatically split the data in `data/train` into training and validation set and store them in `data/train` and `data/val` respectively.

The resulting dataset should have the following structure:

```
data
├── train
│  └── <data_id>
│ 	 ├── <data_id>.mp4
│ 	 └── <data_id>_S2.csv
├── val
│  └── <data_id>
│ 	 ├── <data_id>.mp4
│ 	 └── <data_id>_S2.csv
└── test
   └── <data_id>
     	 └── <data_id>.mp4
```

Where `<data_id>.mp4` are video clips of a single rally, and `<data_id>_S2.csv` is the ground truth labels provided by the official. The data under `data/test` are used for evaluating our models' performance by the official, thus it does not contain any ground truth files. Our final goal is to train our models to be able to predict these labels ourself.

## Data Preprocessing

Before starting to train our models, we need to first preprocess the dataset to extract useful informations from the videos. Namely, the coordinate of shuttlecock, court corners and key points of both player's poses per-frame.

First, download the pre-trained model for [`TrackNetV2`](https://drive.google.com/file/d/1_mrzOAAGsn2DAI7T1igJ9pYKabV278lb/view) and place it wherever you want, then run the following commands in the shell line-by-line to preprocess our data:

```shell
$ python src/shuttlecock_tracker/process_dataset.py --dataset_path data/train --load_weights <path_to_pre_trained_tracknet_model>
$ python src/shuttlecock_tracker/process_dataset.py --dataset_path data/val --load_weights <path_to_pre_trained_tracknet_model>
$ python src/shuttlecock_tracker/process_dataset.py --dataset_path data/test --load_weights <path_to_pre_trained_tracknet_model>
$ python src/court_detector/process_dataset.py --dataset_path data/train
$ python src/court_detector/process_dataset.py --dataset_path data/val
$ python src/court_detector/process_dataset.py --dataset_path data/test
$ python src/player_pose_estimator/process_dataset.py --dataset_path data/train --det_config configs/mmdet/faster_rcnn_r50_fpn_coco.py --det_checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --pose_config configs/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-384x288.py --pose_checkpoint https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-384x288-c161b7de_20220915.pth
$ python src/player_pose_estimator/process_dataset.py --dataset_path data/val --det_config configs/mmdet/faster_rcnn_r50_fpn_coco.py --det_checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --pose_config configs/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-384x288.py --pose_checkpoint https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-384x288-c161b7de_20220915.pth
$ python src/player_pose_estimator/process_dataset.py --dataset_path data/test --det_config configs/mmdet/faster_rcnn_r50_fpn_coco.py --det_checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --pose_config configs/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-384x288.py --pose_checkpoint https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-384x288-c161b7de_20220915.pth
$ python src/hit_detector/organize_xy.py --data_dir data/train
$ python src/hit_detector/organize_xy.py --data_dir data/val
$ python src/hit_detector/organize_xy.py --data_dir data/test
```

The dataset structure should looks like the following after this step:

```
data
├── test
│  └── <data_id>
│ 	 ├── <data_id>.mp4
│ 	 ├── <data_id>_court.csv
│ 	 ├── <data_id>_hitnet_x.csv
│ 	 ├── <data_id>_player_poses.csv
│ 	 └── <data_id>_trajectory.csv
├── train
│  └── <data_id>
│ 	 ├── <data_id>.mp4
│ 	 ├── <data_id>_court.csv
│ 	 ├── <data_id>_hitnet_x.csv
│ 	 ├── <data_id>_hitnet_y.csv
│ 	 ├── <data_id>_player_poses.csv
│ 	 ├── <data_id>_S2.csv
│ 	 └── <data_id>_trajectory.csv
└── val
   └── <data_id>
      ├── <data_id>.mp4
      ├── <data_id>_court.csv
      ├── <data_id>_hitnet_x.csv
      ├── <data_id>_hitnet_y.csv
      ├── <data_id>_player_poses.csv
      ├── <data_id>_S2.csv
      └── <data_id>_trajectory.csv
```

Each of the newly generated csv files are used in the next step to train our models.

## Training

```shell
$ python src/hit_detector/train.py --train_dataset data/train --val_dataset data/val
$ python src/shot_type_classifier/train.py --train_dataset data/train --val_dataset data/val
$ python src/player_backhand_detector/train.py --train_dataset data/train --val_dataset data/val
$ python src/player_roundhead_detector/train.py --train_dataset data/train --val_dataset data/val
```

All the model's check points and logs should be saved under the `saved` directory.

The hyperparameters can be passed as command-line arguments. Feel free to modify and experiment with them. Refer to the `train.py` scripts of each model for the list of adjustable hyparameters.

## Inference

Pre-trained models:

* [Hit Detector](https://github.com/jerrykal/AI_CUP_2023_Spring_Computer_Vision/releases/download/0.0.0/hit_detector_weights.pt)
* [Shot Type Classifier](https://github.com/jerrykal/AI_CUP_2023_Spring_Computer_Vision/releases/download/0.0.0/shot_type_classifier.pt)
* [Back Hand Classifier](https://github.com/jerrykal/AI_CUP_2023_Spring_Computer_Vision/releases/download/0.0.0/back_hand_classifier_weights.pt)
* [Round Head Classifier](https://github.com/jerrykal/AI_CUP_2023_Spring_Computer_Vision/releases/download/0.0.0/round_head_classifier_weights.pt)

Use the models trained from the last step which are saved under `saved/<ModelName>/weights/` directory or the pre-trained model we provided to perform inferences on the `data/test` dataset:

```shell
$ python src/hit_detector/process_dataset.py --dataset_path data/test --model_path <path_to_pre_trained_hit_detector_model> --concat_n 14 --conf_threshold 0.8
$ python src/shot_type_classifier/process_dataset.py --dataset_path data/test --model_path <path_to_pre_trained_shot_type_classifier_model>
$ python src/player_backhand_detector/process_dataset.py --dataset_path data/test --model_path <path_to_pre_trained_back_hand_detector_model>
$ python src/player_roundhead_detector/process_dataset.py --dataset_path data/test --model_path <path_to_pre_trained_round_head_detector_model>
```

Some new csv files should be generated under the `data/test/<data_id>` directory, each containing inference results from our models.

## Organize Answer

Now we just need to run the following command to generate a `answer.csv` file, ready to be evaluated:

```shell
$ python organize_answer.py --dataset_path data/test --det_config configs/mmdet/faster_rcnn_r50_fpn_coco.py --det_checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --pose_config configs/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-wholebody-384x288.py --pose_checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth
```
