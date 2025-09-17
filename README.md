# Visual Servor: A Robot Waiter with a Hanging Tray

The code for training the custom segmentation network for detecting people with
their hand up can be found [here](https://github.com/adamheins/yolo_seg_hand_up).

## Setup and install

This project was built and run on a laptop running Ubuntu 20.04 with ROS1
Noetic. Clone this repo into your catkin workspace. Then install dependencies:
```
# apt packages
sudo apt install python3-opencv ros-noetic-cv-bridge

# pip packages
cd visual_servor
pip install -r requirements.txt
```

Follow the instructions to setup
[mobile_manipulation_central](https://github.com/utiasDSL/mobile_manipulation_central)
and the [ROS1 SDK](https://github.com/orbbec/OrbbecSDK_ROS1) for the Orbbec
Femto Bolt camera.

Finally, build your catkin workspace.

## Usage

### Simulation

Simulation experiments are run using `scripts/simulation.py`.

### Hardware Experiments

Trials of transporting objects with different motion profiles and trays are
done using the script `scripts/trial.py`. Record the data to a rosbag by
running `scripts/record.py` at the same time in a separate terminal.

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

