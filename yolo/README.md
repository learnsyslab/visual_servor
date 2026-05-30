# Visual Servor YOLO Model

The goal of the model is to identify people raising their hand to signal that
they want to be served. One approach is to use the off-the-shelf YOLO
segmentation and keypoint pose models to segment people, detect their pose
keypoints, and then use some logic to check if those keypoints correspond to a
raised hand (e.g., wrist keypoints above face keypoints).

Instead, we train a combined model that is faster than running the segmentation
model and keypoint pose models separately, to facilitate better closed-loop
control performance when deployed on a robot. In particular, this code trains a
YOLO model to produce segmentation masks of people with two classes:
1. people with their hand up;
2. everyone else.
using the [COCO dataset](https://cocodataset.org). This is done by first using
the COCO pose keypoint data to label all instances of humans in images as
`person_hand_up` or `person_hand_down`, and then training a segmentation model
on those labels.

## Usage

You need to have the COCO dataset. Update the `path` variable in the
`visual_servor_yolo.yaml` config file to point to its location on disk. Then:
```
# setup virtual environment
uv sync
source .venv/bin/activate

# convert COCO dataset annotations to YOLO labels
python annotations_to_labels.py <path/to/annotations/JSON/file>

# train the model:
python train.py --checkpoint <path/to/checkpoint>
# or if using slurm (you'll probably need to change some parameters in the slrm
# file):
sbatch launch_job.slrm
```
