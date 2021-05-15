/*
    Aug. 12 2018, He Zhang, hzhang8@vcu.edu 

    A depth handler node
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>
#include "../vo/depth_handler.h"
#include "../utility/tic_toc.h"
using namespace std; 

ros::Publisher *depthCloudPubPointer = NULL;
ros::Publisher *currDepthCloudPubPointer = NULL;

DepthHandler* pDptHandler = new DepthHandler(); 

void voDataHandler(const nav_msgs::Odometry::ConstPtr& voData); 
void syncCloudHandler(const sensor_msgs::Image::ConstPtr& syncCloud2); 

int main(int argc, char** argv)
{
  ros::init(argc, argv, "depth_handler");
  ros::NodeHandle nh;
  ros::NodeHandle np("~");
  // ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  
  ros::Subscriber voDataSub = nh.subscribe<nav_msgs::Odometry> ("/cam_to_init", 5, voDataHandler);
  ros::Subscriber syncCloudSub = nh.subscribe<sensor_msgs::Image>
	("/cam0/depth", 5, syncCloudHandler);

  ros::Publisher depthCloudPub = nh.advertise<sensor_msgs::PointCloud2> ("/depth_cloud", 5);
  depthCloudPubPointer = &depthCloudPub;

  ros::Publisher cur_depthCloudPub = nh.advertise<sensor_msgs::PointCloud2> ("/current_depth_cloud", 5);
  currDepthCloudPubPointer = &cur_depthCloudPub;

    string param_file(""); 
    np.param("config_file", param_file, param_file); 
    if(param_file != "")
    { 
	pDptHandler->readParam(param_file);
    }

  ros::spin();

  return 0;
}


void syncCloudHandler(const sensor_msgs::Image::ConstPtr& syncCloud2)
{
    pDptHandler->cloudHandler2(syncCloud2); 
    sensor_msgs::PointCloud2 depthCloud2; 
    pcl::toROSMsg(*(pDptHandler->mCloudArray[pDptHandler->mSyncCloudId]), depthCloud2);
    depthCloud2.header.frame_id = "camera"; 
    depthCloud2.header.stamp = syncCloud2->header.stamp; 
    currDepthCloudPubPointer->publish(depthCloud2);

    return; 
}

void voDataHandler(const nav_msgs::Odometry::ConstPtr& voData)
{
    static double sum_pt_t = 0; 
    static int sum_pt_cnt = 0;
    // TicToc t_pt; 
    pDptHandler->voDataHandler(voData);
    // double whole_t = t_pt.toc();
    // ROS_WARN("depth_handler_node.cpp: depth cloud cost %f ms", whole_t); 
    // sum_pt_t += whole_t; 
    // ROS_WARN("depth_handler_node.cpp: average depth cloud cost %f ms", sum_pt_t/(++sum_pt_cnt));
		
    // publish the result 
    sensor_msgs::PointCloud2 depthCloud2;
    pcl::toROSMsg(*(pDptHandler->mCloudPub), depthCloud2);
    depthCloud2.header.frame_id = "camera";
    depthCloud2.header.stamp = voData->header.stamp;
    depthCloudPubPointer->publish(depthCloud2);
    // cout<<"depth_handler.cpp: publish depth cloud "<<depthCloud2.height * depthCloud2.width<<" points"<<endl;
    ROS_DEBUG("depth_handler: publish depth cloud at %f ",  depthCloud2.header.stamp.toSec()); 
    return ; 
}



