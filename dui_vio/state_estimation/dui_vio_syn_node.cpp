/*
    Dec. 11, 2019, He Zhang, hzhang8@vcu.edu 
    
    synchronize msg similar to VINS-Mono

*/

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "dui_vio.h"
#include "parameters.h"
#include "../utility/visualization.h"


#define R2D(r) ((r)*180./M_PI)

DUI_VIO dui_vio;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

std::mutex m_dpt_buf; // depth 
queue<sensor_msgs::Image::ConstPtr> dpt_img_buf;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

sensor_msgs::Image::ConstPtr getDptImage(double timestamp)
{
    ROS_WARN("dui_vio_syn_node.cpp: try to find out dpt img at %lf", timestamp); 
    while(true){
        if(dpt_img_buf.empty())
            return NULL; 
        sensor_msgs::Image::ConstPtr dpt_img = dpt_img_buf.front();
        double current_time = dpt_img->header.stamp.toSec(); 

        if(fabs(current_time - timestamp) < 1e-3){
            ROS_DEBUG("dui_vio_syn_node.cpp: found syn depth image at timestamp: %lf", current_time); 
            return dpt_img; 
        }else if(current_time < timestamp){
            dpt_img_buf.pop(); 
            ROS_INFO("dui_vio_syn_node.cpp: remove older depth at timestamp: %lf", current_time);
            continue; 
        }else {
            ROS_ERROR("dui_vio_syn_node.cpp: what? cannot find syn depth img"); 
            return NULL; 
        }
    }
    return NULL; 
}

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() ))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec()))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() )
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void dpt_callback(const sensor_msgs::Image::ConstPtr& dpt_img)
{
    m_dpt_buf.lock(); 
        dpt_img_buf.push(dpt_img);
    m_dpt_buf.unlock();
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

// thread: visual-inertial odometry
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec();
                if (t <= img_t)
                { 
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x*nG;
                    dy = imu_msg->linear_acceleration.y*nG;
                    dz = imu_msg->linear_acceleration.z*nG;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    Vector3d acc(dx, dy, dz);
                    Vector3d gyo(rx, ry, rz); 
                    dui_vio.processIMU(dt, acc, gyo);
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x*nG;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y*nG;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z*nG;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    Vector3d acc(dx, dy, dz);
                    Vector3d gyo(rx, ry, rz); 
                    dui_vio.processIMU(dt_1, acc, gyo);
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            // map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            map<int, vector<pair<int, Eigen::Matrix<double, 10, 1>>>> image;


            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                // double z = img_msg->points[i].z;
                double z = 0.; // img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                // Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                Eigen::Matrix<double, 10, 1> xyz_uv_velocity;
                double lambda = 0; 
                double sig_d = 0; 
                double sig_lambda = 0; 
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y, lambda, sig_d, sig_lambda;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }

            // retrieve depth data 
            m_dpt_buf.lock(); 
            sensor_msgs::Image::ConstPtr dpt_ptr = getDptImage(img_msg->header.stamp.toSec());
            m_dpt_buf.unlock(); 

            if(dpt_ptr != NULL) {
                if(dpt_ptr->encoding == "16UC1"){
                    sensor_msgs::Image img;
                    img.header = dpt_ptr->header;
                    img.height = dpt_ptr->height;
                    img.width = dpt_ptr->width;
                    img.is_bigendian = dpt_ptr->is_bigendian;
                    img.step = dpt_ptr->step;
                    img.data = dpt_ptr->data;
                    img.encoding = "mono16";
                    // ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO16);
                    cv_bridge::CvImageConstPtr ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO16);
                    if(!USE_GMM){
                        dui_vio.associateDepthSimple(image, ptr->image); 
                    }else{
                        dui_vio.associateDepthGMM(image, ptr->image, USE_GMM_EXT); 
                    }
                }else{
                    cv_bridge::CvImageConstPtr ptr = cv_bridge::toCvCopy(dpt_ptr, sensor_msgs::image_encodings::MONO16);
                    if(!USE_GMM){
                        dui_vio.associateDepthSimple(image, ptr->image); 
                    }else{
                        dui_vio.associateDepthGMM(image, ptr->image, USE_GMM_EXT); 
                    }
                }
            }

            dui_vio.processImage_Init(image, img_msg->header.stamp.toSec());

            double whole_t = t_s.toc();
            // printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(dui_vio, header);
            // pubKeyPoses(estimator, header);
            // pubCameraPose(estimator, header);
            // pubPointCloud(estimator, header);
            // pubTF(estimator, header);
            // pubKeyframe(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "dui_vio_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug); // Info
    readParameters(n);
    dui_vio.setParameter();
    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_dpt = n.subscribe("/cam0/depth", 2000, dpt_callback); // /cam0/depth
    // ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    // ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
