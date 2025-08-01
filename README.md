# Visual Servor: A Robot Waiter Serving Demo

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

## Usage

### Collision Teleop

Collision avoidance can be tested independently. In one terminal, run
`rosrun visual_servor collision_teleop.py`. In another terminal, run `roslaunch
visual_servor keyboard_teleop_base.launch` to control the robot's motion with the
keyboard.

### Serving Demo

In three terminals, respectively run:

* `roslaunch visual_servor visual_servor.launch`
* `rosrun visual_servor vision_node.py --display`
* `rosrun visual_servor control_node.py`

