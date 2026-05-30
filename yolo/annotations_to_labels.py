"""Convert COCO annotations to YOLO labels."""

import argparse
import os
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from ultralytics.data.converter import convert_coco, merge_multi_segment

HAND_UP_CLASS = 0
HAND_DOWN_CLASS = 1


def parse_segmentation(seg, img):
    h, w = img["height"], img["width"]
    if len(seg) > 1:
        s = merge_multi_segment(seg)
        s = np.concatenate(s, axis=0)
    else:
        s = [j for i in seg for j in i]  # all segments concatenated
        s = np.array(s).reshape(-1, 2)
    s = (s / np.array([w, h])).reshape(-1)

    # fix any out-of-bounds segments
    s[s < 0] = 0
    s[s > 1] = 1
    return s.astype(np.float32).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to annotations JSON file.")
    parser.add_argument("-o", "--output", help="Path to output label directory.")
    args = parser.parse_args()

    coco = COCO(args.path)

    seg_labels = {}
    num_hand_up = 0
    num_hand_down = 0

    for ann in coco.anns.values():
        # compute head location from first 5 keypoints
        # v=0: not labeled, v=1: labeled but not visible, and v=2: labeled and visible
        # see https://cocodataset.org/#keypoints-eval
        kpts = np.array(ann["keypoints"]).reshape((17, 3))
        head_kpts = kpts[:5, :]
        head_mask = head_kpts[:, 2] > 0

        # cannot see head, do not include
        if not np.any(head_mask):
            continue
        head_pos = np.mean(head_kpts[head_mask, :2], axis=0)
        head_height = head_pos[1]

        left_wrist = kpts[9, :]
        right_wrist = kpts[10, :]

        if left_wrist[2] > 0 and left_wrist[1] < head_height:
            class_id = HAND_UP_CLASS
            num_hand_up += 1
        elif right_wrist[2] > 0 and right_wrist[1] < head_height:
            class_id = HAND_UP_CLASS
            num_hand_up += 1
        else:
            class_id = HAND_DOWN_CLASS
            num_hand_down += 1

        # no segmentation
        seg = ann["segmentation"]
        if len(seg) == 0:
            continue

        image_id = ann["image_id"]
        img = coco.imgs[image_id]
        s = parse_segmentation(seg, img)

        # prepend class ID and fix to 6 decimal points
        row = str(class_id) + " " + " ".join(f"{x:.6f}" for x in s)

        if image_id in seg_labels:
            seg_labels[image_id].append(row)
        else:
            seg_labels[image_id] = [row]

    print(f"num_hand_up   = {num_hand_up}")
    print(f"num_hand_down = {num_hand_down}")

    if args.output is not None:
        dirname = Path(args.output)
        os.mkdir(dirname)
        for image_id, rows in seg_labels.items():
            image_name = coco.imgs[image_id]["file_name"]
            label_path = (dirname / image_name).with_suffix(".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(rows))


if __name__ == "__main__":
    main()
