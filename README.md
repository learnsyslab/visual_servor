# Visual Servor

This repository is contains the code for the experiments shown in the paper
[Robotic Nonprehensile Object Transportation with a Hanging Tray](https://arxiv.org/abs/2606.10039),
to appear in the Proceedings of the IEEE/ASME International Conference on
Advanced Intelligent Mechatronics, 2026. A video of the experiments can be
found [here](https://tiny.cc/visual-servor).

## Model training

The YOLO model used to segment people with a raised hand can be trained using
the self-contained code in the [yolo](./yolo) directory.

## Setup and install

This project was built and run on a laptop running Ubuntu 20.04 with ROS1
Noetic. It controls a Ridgeback mobile base using a Femto Bolt camera and a
Vicon motion capture system.

Follow the instructions to setup
[mobile_manipulation_central](https://github.com/utiasDSL/mobile_manipulation_central)
and the [ROS1 SDK](https://github.com/orbbec/OrbbecSDK_ROS1) for the Orbbec
Femto Bolt camera.

Clone the repository into your catkin workspace:
```
cd catkin_ws/src
git clone https://github.com/learnsyslab/visual_servor
```

Install additional dependencies:
```
# apt packages
sudo apt install python3-opencv ros-noetic-cv-bridge ros-noetic-ros-numpy

# pip packages
cd visual_servor
pip install -r requirements.txt
```

Finally, build your catkin workspace.

## Usage

### Simulation Experiments

Simulation experiments are run using `scripts/run_simulation_experiment.py`.

### Hardware Experiments

You first need to measure the rest position of the hanging tray relative to the
robot's end effector. Use the script `scripts/calibrate_hanging_tray.py` and
save the result in the `config` directory.

Trials of transporting objects with different motion profiles and trays are
done using the script `scripts/run_hardware_experiment.py`. Record the data to
a rosbag by running `scripts/record_bag.py` at the same time in a separate
terminal.

To calculate the friction coefficient for different tray and object
combinations, record a bag during which you slowly manually tilt the tray with
the object on it until the object starts to slide. Then use
`scripts/compute_mu_from_vicon_data.py` to calculate the friction coefficient
corresponding to the tilt angle. Note that friction coefficients are not
required for control.

### Interactive Robot Waiter Demo

In three terminals, respectively run:

* `roslaunch visual_servor visual_servor.launch`
* `rosrun visual_servor waiter_vision_node.py --display`
* `rosrun visual_servor waiter_control_node.py`

## License

MIT. See the LICENSE file.
