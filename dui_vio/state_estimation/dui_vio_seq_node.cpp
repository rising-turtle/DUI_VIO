/*
    Oct. 9 2019, He Zhang, hzhang8@vcu.edu 
    
    dvio node, handle data in sequential

*/
#include <ros/ros.h>
#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include "dvio.h"
#include "parameters.h"
#include "../utility/tic_toc.h"
#include "visualization.h"

#define R2D(r) ((r)*180./M_PI)

DVIO dvio; 
std::mutex m_rgbd_buf;
std::condition_variable con;

ros::Publisher *dvioDataPubPointer = NULL;
ros::Publisher *depthPointsPubPointer = NULL;
ros::Publisher *imagePointsProjPubPointer = NULL;
ros::Publisher *imageShowPubPointer = NULL;
ros::Publisher *groundtruthPubPointer = NULL; 
ros::Publisher *estimatePathPubPointer = NULL; 

tf::TransformBroadcaster * tfBroadcasterPointer = NULL; // camera_init to camera
tf::TransformBroadcaster * tfBroadcastTWI; // world to imu
tf::TransformBroadcaster * tfBroadcastTWC_init; // world to camera_init

double sum_vo_t = 0; 
int sum_vo_cnt = 0; 

queue<sensor_msgs::Image::ConstPtr> rgb_img_buf;
queue<sensor_msgs::Image::ConstPtr> dpt_img_buf; 

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv::Mat ret_img; 
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
        ret_img = ptr->image.clone(); 
    }
    else if(img_msg->encoding == "8UC3"){
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "bgr8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
        ret_img = ptr->image.clone(); 
        cv::cvtColor(ret_img, ret_img, cv::COLOR_BGR2GRAY);
    }else if(img_msg->encoding == "16UC1"){
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono16";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO16);
        ret_img = ptr->image.clone(); 
    }
    else{
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        ret_img = ptr->image.clone(); 
    }

    // cv::Mat img = ptr->image.clone();
    return ret_img;
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // m_buf.lock();
    // imu_buf.push(imu_msg);
    // m_buf.unlock();
    // con.notify_one();
    double t = imu_msg->header.stamp.toSec();

    double dx = imu_msg->linear_acceleration.x *(-9.8) ;
    double dy = imu_msg->linear_acceleration.y *(-9.8);
    double dz = imu_msg->linear_acceleration.z *(-9.8);

    double rx = imu_msg->angular_velocity.x ;
    double ry = imu_msg->angular_velocity.y ;
    double rz = imu_msg->angular_velocity.z ;
    Vector3d acc(dx, dy, dz); 
    Vector3d gyr(rx, ry, rz);
    dvio.inputIMU(t, acc, gyr);
}

void colorImageHandler(const sensor_msgs::Image::ConstPtr& rgb_img)
{   
    m_rgbd_buf.lock();
    rgb_img_buf.push(rgb_img);
    m_rgbd_buf.unlock();
    con.notify_one();
}

void depthImageHandler(const sensor_msgs::Image::ConstPtr& dpt_img)
{
    m_rgbd_buf.lock(); 
    dpt_img_buf.push(dpt_img); 
    m_rgbd_buf.unlock(); 
    con.notify_one();
}

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloud2ConstPtr>>
getMeasurements();

std::vector<std::pair<sensor_msgs::Image::ConstPtr, sensor_msgs::Image::ConstPtr>>
getRGBD(); 

// void process(); 
// void process_depthcloud();
// void process_depthimage();
// void imageDataHandler(const sensor_msgs::Image::ConstPtr& imageData);
void processRGBD();
void depthDataHandler(const sensor_msgs::Image::ConstPtr& depthData); 
void publishMsg(sensor_msgs::PointCloud2ConstPtr& img_msg);
// void groundtruthHandler(const geometry_msgs::PointStampedConstPtr& gt_point); 

int main(int argc, char **argv)
{
    ros::init(argc, argv, "dvio_seq_node");
    ros::NodeHandle nh("~");
    // ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
    readParameters(nh);
    dvio.setParameter();
    nh.param("stereo_input", dvio.mbStereo, dvio.mbStereo);  
    // if(dvio.mbStereo)
    {
        // ROS_DEBUG("dvio_node: stereo input is set!");
    }
    std::cout<<std::fixed<< std::setprecision(6); 
    ROS_WARN("waiting for image and imu...");

    registerPub(nh); // register publishers 
    
    tf::TransformBroadcaster tfBroadcaster;
    tfBroadcasterPointer = &tfBroadcaster;
    
    
    tf::TransformBroadcaster tfBroadcaster1;
    tfBroadcastTWI = &tfBroadcaster1;
    
    
    tf::TransformBroadcaster tfBroadcaster2;
    tfBroadcastTWC_init = &tfBroadcaster2;
    

    ros::Subscriber sub_imu = nh.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    // ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_rgb = nh.subscribe(IMAGE_TOPIC, 2000, colorImageHandler);
    ros::Subscriber sub_dpt = nh.subscribe(DPT_IMG_TOPIC, 2000, depthImageHandler); 
    
    // ros::Subscriber imagePointsSub = nh.subscribe<sensor_msgs::PointCloud2>
    //                                ("/image_points_last", 5, imagePointsHandler);

    // ros::Subscriber depthCloudSub = nh.subscribe<sensor_msgs::PointCloud2> 
    //                              ("/depth_cloud", 5, depthCloudHandler); // transformed depth cloud for the last stamp 

    // ros::Subscriber currDepthCloudSub = nh.subscribe<sensor_msgs::PointCloud2>
    //                               ("/current_depth_cloud", 5, currDepthCloudHandler); // observed depth cloud for the curr stamp

    // ros::Subscriber depthImageSub = nh.subscribe<sensor_msgs::Image>
    //                              ("/cam0/depth", 5, depthImageHandler); // depth image handler 

    // ros::Publisher imagePointsProjPub = nh.advertise<sensor_msgs::PointCloud2> ("/image_points_proj", 5);
    // imagePointsProjPubPointer = &imagePointsProjPub;

    // ros::Publisher voDataPub = nh.advertise<nav_msgs::Odometry> ("/cam_to_init", 5);
    // voDataPubPointer = &voDataPub;
    
    ros::Publisher dvioDataPub = nh.advertise<nav_msgs::Odometry>("/world_to_imu", 5); 
    dvioDataPubPointer = &dvioDataPub; 

    // ros::Subscriber imageDataSub = nh.subscribe<sensor_msgs::Image>("/image/show", 1, imageDataHandler);

    ros::Publisher imageShowPub = nh.advertise<sensor_msgs::Image>("/image/show_2", 1);
    imageShowPubPointer = &imageShowPub;
    
    // ros::Publisher imuEulerPub = nh.advertise<std_msgs::Float32MultiArray>("/euler_msg", 10); 
    // imuEulerPubPointer = &imuEulerPub; 

    // ros::Subscriber gtSub = nh.subscribe<geometry_msgs::PointStamped>("/leica/position", 7, groundtruthHandler);
    ros::Publisher gtPub = nh.advertise<nav_msgs::Path>("/ground_truth_path", 7); 
    ros::Publisher estimatePub = nh.advertise<nav_msgs::Path>("/estimate_path", 7); 
    groundtruthPubPointer = &gtPub;
    estimatePathPubPointer = &estimatePub; 

    // std::thread measurement_process{process};
    // std::thread depthcloud_process{process_depthcloud}; 
    std::thread depth_img_process{processRGBD}; 
    ros::spin();

    return 0;
}

void processRGBD()
{
    while (ros::ok())
    {
        std::vector<std::pair<sensor_msgs::Image::ConstPtr, sensor_msgs::Image::ConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_rgbd_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getRGBD()).size() != 0;
                 });
        lk.unlock();

        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.first;
            // ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            // estimator.processImage(image, img_msg->header);
            // dvio.processImage(img_msg); 

            cv::Mat rgb = getImageFromMsg(measurement.first); 
            cv::Mat dpt = getImageFromMsg(measurement.second); 

            dvio.inputRGBD(img_msg->header.stamp.toSec(), rgb, dpt); 
            double whole_t = t_s.toc();
            // ROS_WARN("dvio_node_seq.cpp: ddvio cost %f ms", whole_t); 
            sum_vo_t += whole_t; 
            // ROS_WARN("dvio_node_seq.cpp: average ddvio cost %f ms", sum_vo_t/(++sum_vo_cnt));

            // static ofstream ouf("demo_rgbd_time.log"); 
            // ouf<<whole_t<<endl; 

            // publishMsg(img_msg); 
            ros::spinOnce();
            // static ofstream process_time_log("process_time.log"); 
            // process_time_log << whole_t<<endl; 
        }
    }

}

void publishMsg(sensor_msgs::Image::ConstPtr& img_msg)
{
      // // publish msg voData 
      //   nav_msgs::Odometry voData; 
      //   voData.header.frame_id = "/camera_init"; 
      //   voData.header.stamp = img_msg->header.stamp;
      //   voData.child_frame_id = "/camera";

      //   tf::Transform vo_to_init = dvio.mInitCamPose.inverse() * dvio.mCurrPose;  
      //   tf::Quaternion q = vo_to_init.getRotation(); 
      //   tf::Vector3 t = vo_to_init.getOrigin(); 
      //   voData.pose.pose.orientation.x = q.getX(); 
      //   voData.pose.pose.orientation.y = q.getY(); 
      //   voData.pose.pose.orientation.z = q.getZ(); 
      //   voData.pose.pose.orientation.w = q.getW(); 
      //   voData.pose.pose.position.x = t.getX(); 
      //   voData.pose.pose.position.y = t.getY();
      //   voData.pose.pose.position.z = t.getZ(); 
      //   voDataPubPointer->publish(voData);

      //   // cout<<"vo node at "<<std::fixed<<voData.header.stamp.toSec()<<" vo result: "<<t.getX()<<" "<<t.getY()<<" "<<t.getZ()<<endl;

      //   {
      //   // broadcast voTrans camera_init -> camera
      //   tf::StampedTransform voTrans;
      //   voTrans.frame_id_ = "/camera_init";
      //   voTrans.child_frame_id_ = "/camera";
      //   voTrans.stamp_ = img_msg->header.stamp;
      //   voTrans.setRotation(q); 
      //   voTrans.setOrigin(t); 
      //   tfBroadcasterPointer->sendTransform(voTrans); 
      //   }

      //   // publish dvio result 
      //   q = dvio.mCurrIMUPose.getRotation(); 
      //   t = dvio.mCurrIMUPose.getOrigin(); 
      //   nav_msgs::Odometry dvioData; 
      //   dvioData.header.frame_id = "/world"; 
      //   dvioData.header.stamp = img_msg->header.stamp;
      //   dvioData.child_frame_id = "/imu";

      //   dvioData.pose.pose.orientation.x = q.getX(); 
      //   dvioData.pose.pose.orientation.y = q.getY(); 
      //   dvioData.pose.pose.orientation.z = q.getZ(); 
      //   dvioData.pose.pose.orientation.w = q.getW();
      //   dvioData.pose.pose.position.x = t.getX(); 
      //   dvioData.pose.pose.position.y = t.getY();
      //   dvioData.pose.pose.position.z = t.getZ(); 
      //   dvioDataPubPointer->publish(dvioData);
      //   cout <<"dvio publish: dvio t "<<t.getX()<<" "<<t.getY() <<" "<<t.getZ()<<endl;
      //   cout <<"dvio publish: dvio q "<< q.getX()<<" "<< q.getY()<<" "<<q.getZ()<<" "<<q.getW()<<endl;
      //   ros::spinOnce();
    
      //   // write result to file
      //   ofstream foutC(VINS_RESULT_PATH, ios::app);
      //   foutC.setf(ios::fixed, ios::floatfield);
      //   foutC.precision(0);
      //   foutC << dvioData.header.stamp.toSec() * 1e9 << ",";
      //   foutC.precision(5);
      //   foutC << t.getX() << ","
      //         << t.getY() << ","
      //         << t.getZ() << ","
      //         << q.getW() << ","
      //         << q.getX() << ","
      //         << q.getY() << ","
      //         << q.getZ() << "," << endl;
      //   foutC.close();

      //   {
      //   // broadcast voTrans imu -> camera 
      //   tf::StampedTransform voTrans;
      //   voTrans.frame_id_ = "/world";
      //   voTrans.child_frame_id_ = "/camera_init";
      //   voTrans.stamp_ = img_msg->header.stamp;
      //   voTrans.setData(dvio.mInitCamPose);
      //   tfBroadcastTWC_init->sendTransform(voTrans); 
      //   }


      //   {
      //   // broadcast voTrans imu -> camera 
      //   tf::StampedTransform voTrans;
      //   voTrans.frame_id_ = "/world";
      //   voTrans.child_frame_id_ = "/imu";
      //   voTrans.stamp_ = img_msg->header.stamp;
      //   voTrans.setData(dvio.mCurrIMUPose);
      //   tfBroadcastTWI->sendTransform(voTrans); 
      //   }

      //   // deal with image
      //   {  
      //       // publish points with depth 
      //       sensor_msgs::PointCloud2 imagePointsProj2;
      //       pcl::toROSMsg(*(dvio.mImagePointsProj), imagePointsProj2);
      //       imagePointsProj2.header.frame_id = "camera";
      //       imagePointsProj2.header.stamp = ros::Time().fromSec(dvio.mTimeLast);
      //       imagePointsProjPubPointer->publish(imagePointsProj2);    
      //   }

      //   {
      //       // publish ground truth path 
      //       dvio.m_gt_path.header.frame_id = dvioData.header.frame_id;
      //       groundtruthPubPointer->publish(dvio.m_gt_path);

      //   }
      //   {
      //       // publish estiamted path 
      //       geometry_msgs::PoseStamped ps;   
      //       ps.header = dvioData.header; 
      //       ps.pose.position = dvioData.pose.pose.position;
      //       ps.pose.orientation = dvioData.pose.pose.orientation; 
      //       dvio.m_est_path.poses.push_back(ps); 
      //       dvio.m_est_path.header = ps.header;  
      //       estimatePathPubPointer->publish(dvio.m_est_path);
      //   }
      //   // cout <<"publish imagePointsProj2 with "<<imagePointsProj2.height * imagePointsProj2.width<<" points!"<<" at time "<<std::fixed<<dvio.mTimeLast<<endl;

}

std::vector<std::pair<sensor_msgs::Image::ConstPtr, sensor_msgs::Image::ConstPtr>>
getRGBD()
{
    std::vector<std::pair<sensor_msgs::Image::ConstPtr, sensor_msgs::Image::ConstPtr>> measurements;

    while (true)
    {
    if (rgb_img_buf.empty() || dpt_img_buf.empty())
        return measurements;

    if ((dpt_img_buf.front()->header.stamp > rgb_img_buf.front()->header.stamp))
    {
        ROS_WARN("throw color img, only should happen at the beginning");
        rgb_img_buf.pop();
        continue;
    }

    if((rgb_img_buf.front()->header.stamp > dpt_img_buf.front()->header.stamp))
    {
        ROS_WARN("throw depth img, only should happen at the beginning");
        dpt_img_buf.pop();
        continue;
    }
    measurements.emplace_back(rgb_img_buf.front(), dpt_img_buf.front());
    rgb_img_buf.pop(); 
    dpt_img_buf.pop(); 
    }
    return measurements;
}

// void imageDataHandler(const sensor_msgs::Image::ConstPtr& imageData) 
// {
//     cv_bridge::CvImagePtr ptr = cv_bridge::toCvCopy(imageData, "bgr8");

//     cv::Mat show_img = ptr->image; 

//     // double kImage[9] = {525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0};
//     // double kImage[9] = {617.306, 0.0, 326.245, 0.0, 617.714, 239.974, 0.0, 0.0, 1.0};
//     double kImage[9] = {FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0};
//     double showDSRate = 2.;
//     vector<ip_M> ipRelations = dvio.mPtRelations; 
//     int ipRelationsNum = ipRelations.size();
//     // cout<<"vo_node display image at "<<std::fixed<<imageData->header.stamp.toSec()<<endl;
//     for (int i = 0; i < ipRelationsNum; i++) 
//     {
//     ip_M pt = ipRelations[i];
//     if ( pt.v == ip_M::NO_DEPTH) 
//     {   
//         // cout<<"No depth: pt.uj = "<<(kImage[2] - pt.uj * kImage[0])<<" pt.vj: "<<(kImage[5] - pt.vj * kImage[4]) <<" pt.s = "<<pt.s<<endl;
//         cv::circle(show_img, cv::Point((kImage[2] + pt.uj * kImage[0]) / showDSRate, (kImage[5] + pt.vj * kImage[4]) / showDSRate), 1, CV_RGB(255, 0, 0), 2);
//     } else if (pt.v == ip_M::DEPTH_MES) {
//         // cout<<"Depth MES: pt.uj = "<<(kImage[2] - pt.uj * kImage[0])<<" pt.vj: "<<(kImage[5] - pt.vj * kImage[4]) <<" pt.s = "<<pt.s<<endl;
//         cv::circle(show_img, cv::Point((kImage[2] + pt.uj * kImage[0]) / showDSRate,(kImage[5] + pt.vj * kImage[4]) / showDSRate), 1, CV_RGB(0, 255, 0), 2);
//     } else if (pt.v == ip_M::DEPTH_TRI) {
//         // cout<<"Depth TRI: pt.uj = "<<(kImage[2] - pt.uj * kImage[0])<<" pt.vj: "<<(kImage[5] - pt.vj * kImage[4]) <<" pt.s = "<<pt.s<<endl;
//         cv::circle(show_img, cv::Point((kImage[2] + pt.uj * kImage[0]) / showDSRate,(kImage[5] + pt.vj * kImage[4]) / showDSRate), 1, CV_RGB(0, 0, 255), 2);
//     } /*else {
//         cv::circle(bridge->image, cv::Point((kImage[2] - ipRelations->points[i].z * kImage[0]) / showDSRate,
//         (kImage[5] - ipRelations->points[i].h * kImage[4]) / showDSRate), 1, CV_RGB(0, 0, 0), 2);
//         }*/
//     }
//     ptr->image = show_img; 
//     sensor_msgs::Image::Ptr imagePointer = ptr->toImageMsg();
//     imageShowPubPointer->publish(imagePointer);
// }

