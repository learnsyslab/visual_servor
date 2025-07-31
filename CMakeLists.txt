cmake_minimum_required(VERSION 3.10.2)
project(serving_demo)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  Target.msg
  SystemState.msg
)
generate_messages(DEPENDENCIES std_msgs)
