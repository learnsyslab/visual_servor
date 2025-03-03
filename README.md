# Robot Waiter Serving Demo

## Setup and install

This project was built and run on a laptop running Ubuntu 20.04 with the
PREEMPT_RT realtime patch and a L515 RealSense camera.

### Camera

I found it necessary to build
[librealsense](https://github.com/IntelRealSense/librealsense) and its [ROS
wrapper](https://github.com/IntelRealSense/realsense-ros) from source for the
camera to work properly. The camera model is L515.

* v2.50.0 of librealsense was used: `git checkout v2.50.0`; build using cmake
* ROS1 version of realsense-ros was used: `git checkout ros1-legacy`; build
  using catkin

### YOLO

YOLOv11 from Ultralytics is used for human pose keypoint detection. Install
with:
```
pip install ultralytics
```
