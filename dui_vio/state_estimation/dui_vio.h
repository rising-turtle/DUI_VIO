/*
    Aug. 21 2018, He Zhang, hzhang8@vcu.edu 
    
    vio: tightly couple features [no depth, triangulated, measured] 
    with IMU integration 

*/

#pragma once

#include <vector>
#include <queue>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/core/eigen.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
// #include "../utility/pointDefinition.h"
#include "imu_factor.h"
#include "parameters.h"
#include "tf/tf.h"
#include "feature_manager.h"
#include "marginalization_factor.h"
#include "../feature_tracking/feature_tracker.h"

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "../initialization/solve_5pts.h"
#include "../initialization/initial_sfm.h"
#include "../initialization/initial_alignment.h"
#include "../initialization/initial_ex_rotation.h"

using namespace std; 

class GMM_Model;
const static int NOT_INITIED = -100000;

enum SolverFlag
{
    INITIAL,
    NON_LINEAR
};

enum MarginalizationFlag
{
    MARGIN_OLD = 0,
    MARGIN_SECOND_NEW = 1
};

class DUI_VIO
{
public:
    DUI_VIO(); 
    virtual ~DUI_VIO(); 
    void clearState();

    void processIMU(double dt, Eigen::Vector3d & linear_acceleration, Eigen::Vector3d& angular_velocity); 
    virtual void solveOdometry(); 
    virtual void slideWindow(); 
    virtual void slideWindowNew();
    virtual void slideWindowOld();

    virtual void priorOptimize(); 
    virtual void afterOptimize(); 

    void associateDepthSimple(map<int, vector<pair<int, Eigen::Matrix<double, 10, 1>>>>& featureFrame, const cv::Mat& dpt);
    void associateDepthGMM(map<int, vector<pair<int, Eigen::Matrix<double, 10, 1>>>>& featureFrame, const cv::Mat& dpt, bool use_sim=true);

    void setParameter();
    void showStatus();

    // initialization functions 
    virtual void processImage_Init(const map<int, vector<pair<int, Eigen::Matrix<double, 10, 1>>>> &image, const double header);

    bool initialStructure();
    bool visualInitialAlign();
    bool visualInitialAlignWithDepth();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);

    // use hybrid pnp method 
    bool relativePoseHybrid(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    bool relativePoseHybridCov(Matrix3d &relative_R, Vector3d &relative_T, int &l);

    // initialization variables 
    double initial_timestamp;
    MotionEstimator motion_estimator; 
    Vector3d g;

    GMM_Model* mp_gmm; 
 
    tf::Transform mInitCamPose; // Initial campose 
    tf::Transform mLastPose; // camera pose for the mImgPTLast 
    tf::Transform mCurrPose; // camera pose for the mImgPTCurr 
    tf::Transform mCurrIMUPose; // current IMU pose  
    tf::Transform mTIC;  // transform from imu to camera  
 
    Eigen::Vector3d mg;
    Eigen::Matrix3d ric[NUM_OF_CAM]; 
    Eigen::Vector3d tic[NUM_OF_CAM]; 
    
    double Headers[WINDOW_SIZE+1];
    Eigen::Vector3d Ps[WINDOW_SIZE+1]; 
    Eigen::Vector3d Vs[WINDOW_SIZE+1]; 
    Eigen::Matrix3d Rs[WINDOW_SIZE+1]; 
    Eigen::Vector3d Bas[WINDOW_SIZE+1]; 
    Eigen::Vector3d Bgs[WINDOW_SIZE+1]; 
    Eigen::Matrix3d R_imu; 

    vector<Vector3d> key_poses;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;

    FeatureManager f_manager; 

    // for marginalization 
    SolverFlag solver_flag;
    MarginalizationFlag marginalization_flag;
    MarginalizationInfo * last_marginalization_info; 
    vector<double*> last_marginalization_parameter_blocks;

    // for initialization 
    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    bool mbFirstIMU; 
    Vector3d acc_0; 
    Vector3d gyr_0; 
 
    std::mutex m_process;
    std::mutex m_buf;
    queue<pair<double, Eigen::Vector3d>> accBuf;
    queue<pair<double, Eigen::Vector3d>> gyrBuf;
    queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;
    double prevTime, currTime;
 
    IntegrationBase * pre_integrations[WINDOW_SIZE+1]; 
    vector<double> dt_buf[WINDOW_SIZE+1]; 
    vector<Eigen::Vector3d> linear_acceleration_buf[WINDOW_SIZE+1]; 
    vector<Eigen::Vector3d> angular_velocity_buf[WINDOW_SIZE+1]; 
    
    int frame_count; 
 
    double para_Pose[WINDOW_SIZE+1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE+1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_FEAT][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Td[1][1]; 
 

    nav_msgs::Path m_gt_path;  // groundtruth path 
    nav_msgs::Path m_est_path; // estimated path 
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
