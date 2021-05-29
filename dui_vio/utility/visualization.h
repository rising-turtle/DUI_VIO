/*
	Oct. 22 2019, He Zhang, hzhang8@vcu.edu 

	functions to display results

*/

#pragma once

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
// #include "CameraPoseVisualization.h"
#include <eigen3/Eigen/Dense>
#include "../state_estimation/dui_vio.h"
#include "../state_estimation/parameters.h"
#include "CameraPoseVisualization.h"
#include <fstream>

extern ros::Publisher pub_odometry;
extern ros::Publisher pub_path, pub_pose;
extern ros::Publisher pub_cloud, pub_map;
extern ros::Publisher pub_key_poses;
extern ros::Publisher pub_ref_pose, pub_cur_pose;
extern ros::Publisher pub_key;
extern nav_msgs::Path path;
extern ros::Publisher pub_pose_graph;
extern int IMAGE_ROW, IMAGE_COL;

void registerPub(ros::NodeHandle &n);

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t);

void pubTrackImage(const cv::Mat &imgTrack, const double t);

// void printStatistics(const DUI_VIO &dvio, double t);

void pubOdometry(const DUI_VIO &dvio, const std_msgs::Header &header);

void pubKeyPoses(const DUI_VIO &dvio, const std_msgs::Header &header);

void pubCameraPose(const DUI_VIO &dvio, const std_msgs::Header &header);

void pubPointCloud(const DUI_VIO &dvio, const std_msgs::Header &header);

void pubTF(const DUI_VIO &dvio, const std_msgs::Header &header);

void pubKeyframe(const DUI_VIO &dvio, const std_msgs::Header &header);