#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

#define SQ(x) ((x)*(x))

const double FOCAL_LENGTH = 460.0; // 460.0; // when use downsampled input change it to 230
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1; //2; // test stereo type representation 1;
const int NUM_OF_FEAT = 1000;
const double LOOP_INFO_VALUE = 50.0;
const double RIG_LEN = 0.1; // stereo rig len , about 0.1 meters 
//#define DEPTH_PRIOR
//#define GT
// #define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern int ESTIMATE_EXTRINSIC;
extern int MIN_USED_NUM; // features used times
extern double MIN_PARALLAX; // minimal parallax 

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string VINS_FOLDER_PATH;

extern int LOOP_CLOSURE;
extern int MIN_LOOP_NUM;
extern int IMAGE_ROW;
extern int IMAGE_COL;
extern double CX, CY, FX, FY;
extern std::string PATTERN_FILE;
extern std::string VOC_FILE;
extern std::string CAM_NAMES;
extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string DPT_IMG_TOPIC; 
extern double PIX_SIGMA; 

extern int MAX_CNT;
extern int MIN_DIST;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int FLOW_BACK;

// for feature tracker 
// extern int EQUALIZE;
// extern int FISHEYE;
extern bool PUB_THIS_FRAME;
extern double FREQ;
extern double nG; // normalize value 

extern bool USE_GMM; // whether use gmm to compute covariance 
extern bool USE_GMM_EXT; // whether extend gmm by introducing similarity 

void readParameters(ros::NodeHandle &n);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1,
    SIZE_PLANE =4
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
