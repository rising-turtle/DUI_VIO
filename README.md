# DUI_VIO
## A RGB-D camera based Visual-Inertial State Estimator 

**Videos:** run with some data sequences in [VCU-RVI](https://github.com/rising-turtle/VCU_RVI_Benchmark)

**Related Paper**


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

## 4. Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.
