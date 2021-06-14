# DUI_VIO
## A RGB-D camera based Visual-Inertial State Estimator 

**Videos:** run with some data sequences in [VCU-RVI](https://github.com/rising-turtle/VCU_RVI_Benchmark)

<a href="https://www.youtube.com/watch?v=nhIlObPyj9I" target="_blank"><img src="http://i.ytimg.com/vi/nhIlObPyj9I/maxresdefault.jpg" 
alt="cla" width="240" height="180" border="10" /></a>
  <a href="https://www.youtube.com/watch?v=IbUUxuumMM0" target="_blank"><img src="http://i.ytimg.com/vi/IbUUxuumMM0/maxresdefault.jpg" 
alt="cla" width="240" height="180" border="10" /></a>
  <a href="https://www.youtube.com/watch?v=Ul80tpgYLRk" target="_blank"><img src="http://i.ytimg.com/vi/Ul80tpgYLRk/maxresdefault.jpg" 
alt="cla" width="240" height="180" border="10" /></a>

**Related Paper**

He Zhang and Cang Ye, **"DUI-VIO: Depth uncertainty incorporated visual inertial odometry based on an RGB-D camera"**, *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2020. [pdf](https://ieeexplore.ieee.org/document/9341592)

## 1. Prerequisites
1.1 **Ubuntu** and **ROS**
Ubuntu  16.04.
ROS Kinetic. [ROS Installation](http://wiki.ros.org/ROS/Installation)
additional ROS pacakge
```
    sudo apt-get install ros-YOUR_DISTRO-cv-bridge ros-YOUR_DISTRO-tf ros-YOUR_DISTRO-message-filters ros-YOUR_DISTRO-image-transport
```

1.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html), remember to **make install**.
(Our testing environment: Ubuntu 16.04, ROS Kinetic, OpenCV 3.3.1) 

## 2. Build DUI_VIO on ROS
Clone the repository and catkin_make:
```
    cd ~/catkin_ws/src
    git clone https://github.com/rising-turtle/DUI_VIO.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```
## 3. Build with [OpenGV](https://github.com/laurentkneip/opengv) (Optional)
OpenGV is not required in default compilation. It is used to improve the accuracy of 2D-2D feature matches for rotation estimation in the HybridPnP algorithm. To compile with OpenGV, set WITH_OPENGV as ON in CMakeLists.txt and set up "opengv_DIR" accordingly 

```
option(WITH_OPENGV "use opengv" ON) #use opengv or not in hybridPnP 
set(opengv_DIR "/PATH_TO_OPENGV/opengv/build")
```

## 4. Demos
Download the [bag_files](https://vcu-rvi-dataset.github.io/2020/08/14/Dataset-Download/) which were collected by a structure-core camera handheld or robot carried. Open a terminal, navigate to the 'DUI_VIO/launch' folder, and launch the dui_vio_sc_run. Open another terminal and play the bag file. 

```
    roslaunch dui_vio_sc_run.launch
    rosbag play YOUR_PATH_TO_DATASET/lab_*.bag 
```

## 5. Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.
