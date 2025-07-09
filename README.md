# Robot Waiter Serving Demo

## Setup and install

This project was built and run on a laptop running Ubuntu 20.04 with the Orbbec
Femto Bolt camera.

### YOLO

YOLOv11 from Ultralytics is used for vision. Install with:
```
pip install ultralytics
```

The code for training the custom segmentation network for detecting people with
their hand up can be found [here](https://github.com/adamheins/yolo_seg_hand_up).

### Resources
* <https://robodev.blog/hand-gesture-recognition-in-ros>
