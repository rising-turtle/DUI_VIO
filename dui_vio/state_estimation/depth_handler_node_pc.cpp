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
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "../vo/depth_handler.h"
using namespace std; 

ros::Publisher *depthCloudPubPointer = NULL;
ros::Publisher *currDepthCloudPubPointer = NULL;

DepthHandler* pDptHandler = new DepthHandler(); 

queue<sensor_msgs::PointCloud2ConstPtr> pc_buf; 
std::mutex m_buf; 
std::condition_variable con; 

void voDataHandler(const nav_msgs::Odometry::ConstPtr& voData); 
void syncCloudHandler(const sensor_msgs::PointCloud2ConstPtr& syncCloud2); 

void processPC();

int main(int argc, char** argv)
{
  ros::init(argc, argv, "depth_handler");
  ros::NodeHandle nh;
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
  
  ros::NodeHandle np("~"); 
  string param_file(""); 
  np.param("config_file", param_file, param_file); 
  if(param_file != "")
  { 
      pDptHandler->readParam(param_file);
  }

  ros::Subscriber voDataSub = nh.subscribe<nav_msgs::Odometry> ("/cam_to_init", 5, voDataHandler);
  ros::Subscriber syncCloudSub = nh.subscribe<sensor_msgs::PointCloud2>
	("/pointcloud", 5, syncCloudHandler);

  ros::Publisher depthCloudPub = nh.advertise<sensor_msgs::PointCloud2> ("/depth_cloud", 5);
  depthCloudPubPointer = &depthCloudPub;

  ros::Publisher cur_depthCloudPub = nh.advertise<sensor_msgs::PointCloud2> ("/current_depth_cloud", 5);
  currDepthCloudPubPointer = &cur_depthCloudPub;


  std::thread pc_process{processPC}; 

  ros::spin();

  return 0;
}

void syncCloudHandler(const sensor_msgs::PointCloud2ConstPtr&  syncCloud2)
{
    ROS_INFO("receive syncCloud2 at time = %f point cloud width %d height %d", syncCloud2->header.stamp.toSec(), syncCloud2->width, syncCloud2->height); 
    m_buf.lock();
    pc_buf.push(syncCloud2); 
    m_buf.unlock(); 
    con.notify_one();

    return; 
}

void processPC()
{
    queue<sensor_msgs::PointCloud2ConstPtr> tmp; 
    while(ros::ok())
    {
      std::unique_lock<std::mutex> lk(m_buf); 
      con.wait(lk, [&]
        {
          tmp.swap(pc_buf); 
          return (!tmp.empty()); 
        });
      lk.unlock(); 

      while(!tmp.empty())
      {
        sensor_msgs::PointCloud2ConstPtr syncCloud2 = tmp.front();
        tmp.pop();
        ROS_DEBUG("handle pc at time = %f", syncCloud2->header.stamp.toSec());
        if(syncCloud2->width * syncCloud2->height < 100)
        {
          ROS_DEBUG("syncCloud2 width %d height %d skip this pc", syncCloud2->width, syncCloud2->height); 
          continue; 
        }
        pDptHandler->cloudHandler3(syncCloud2); 
        sensor_msgs::PointCloud2 depthCloud2; 
        pcl::toROSMsg(*(pDptHandler->mCloudArray[pDptHandler->mSyncCloudId]), depthCloud2);
        depthCloud2.header.frame_id = "camera"; 
        depthCloud2.header.stamp = syncCloud2->header.stamp; 
        currDepthCloudPubPointer->publish(depthCloud2);
        ros::spinOnce(); 
      }


    }   

}

/*
void syncCloudHandler(const sensor_msgs::PointCloud2ConstPtr&  syncCloud2)
{
    ROS_INFO("receive syncCloud2 at time = %f", syncCloud2->header.stamp.toSec()); 
    pDptHandler->cloudHandler3(syncCloud2); 
    sensor_msgs::PointCloud2 depthCloud2; 
    pcl::toROSMsg(*(pDptHandler->mCloudArray[pDptHandler->mSyncCloudId]), depthCloud2);
    depthCloud2.header.frame_id = "camera"; 
    depthCloud2.header.stamp = syncCloud2->header.stamp; 
    currDepthCloudPubPointer->publish(depthCloud2);

    return; 
}*/

void voDataHandler(const nav_msgs::Odometry::ConstPtr& voData)
{
    pDptHandler->voDataHandler(voData);
    
    // publish the result 
    sensor_msgs::PointCloud2 depthCloud2;
    pcl::toROSMsg(*(pDptHandler->mCloudPub), depthCloud2);
    depthCloud2.header.frame_id = "camera";
    depthCloud2.header.stamp = voData->header.stamp;
    depthCloudPubPointer->publish(depthCloud2);
    // cout<<"depth_handler.cpp: publish depth cloud "<<depthCloud2.height * depthCloud2.width<<" points"<<endl;
    return ; 
}

