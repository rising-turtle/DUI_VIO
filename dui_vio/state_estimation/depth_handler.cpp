/*
    Aug. 10 2018, He Zhang, hzhang8@vcu.edu 

    A handler to process depth data 
*/

#include "depth_handler.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include "opencv/cv.h"

using namespace std;

namespace{

    void fromTF2Eigen(tf::Transform & T, Eigen::Matrix<double, 3, 3>& R, Eigen::Matrix<double, 3, 1>& t)
    {
	tf::Quaternion q = T.getRotation(); 
	tf::Vector3 tt = T.getOrigin(); 
	tf::Matrix3x3 tR(q); 
	for(int i=0; i<3; i++)
	for(int j=0; j<3; j++)
	{
	    R(i,j) = tR[i][j];
	}
	t << tt.getX(), tt.getY(), tt.getZ(); 
	return ;
    }
    void fromEigen2TF(tf::Transform & T, Eigen::Matrix<double, 3, 3>& R, Eigen::Matrix<double, 3, 1>& t)
    {
	Eigen::Quaterniond q(R); 
	tf::Quaternion tq(q.x(), q.y(), q.z(), q.w()); 
	tf::Vector3 tt(t(0), t(1), t(2));
	T = tf::Transform(tq, tt); 
	return ; 
    }
}

// template<int CLOUD_NUM>
DepthHandler::DepthHandler(): 
mCloudCnt(-1),
mCloudSkip(2), 
mInitTime(-1),
mCloudDSRate(5),
mMaxDepth(2.2), // 3
mSyncCloudId(-1),
mRegCloudId(0),
mTimeRec(0),
mZoomDis(10.),
mCloudRec(new pcl::PointCloud<pcl::PointXYZI>()),
mCloudPub(new pcl::PointCloud<pcl::PointXYZI>())
{
    for(int i=0; i<CLOUD_NUM; i++)
    {
	mCloudStamp[i] = 0; 
	mCloudArray[i].reset(new pcl::PointCloud<pcl::PointXYZI>()); 
    }
    mk[0] = 617.31; // 525.; // 617.31; // fx
    mk[1] = 617.71; // 525.; // 617.71; // fy
    mk[2] = 326.24; // 319.5; // 326.24; // cx
    mk[3] = 239.97; // 239.5; // 239.97; // cy
}

// template<int CLOUD_NUM>
DepthHandler::~DepthHandler(){}

void DepthHandler::cloudHandler3(const sensor_msgs::PointCloud2ConstPtr& depthCloud2)
{
	mCloudCnt = (mCloudCnt+1)%(mCloudSkip+1); 
    if(mCloudCnt != 0) 
		return ; 
    if(mInitTime == -1)
		mInitTime = depthCloud2->header.stamp.toSec(); 
    double time = depthCloud2->header.stamp.toSec(); 
    double timeElapsed = time - mInitTime; 

    mSyncCloudId = (mSyncCloudId+1)%(CLOUD_NUM); 
    mCloudStamp[mSyncCloudId] = time; 

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpPC(new pcl::PointCloud<pcl::PointXYZRGB>); 
	// ROS_INFO("before convert pc at time = %f", time); 
    pcl::fromROSMsg(*depthCloud2, *tmpPC); 

	/*{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpPC_filtered(new pcl::PointCloud<pcl::PointXYZRGB>); 
    pcl::VoxelGrid<pcl::PointXYZRGB> downSizeFilter;
    downSizeFilter.setInputCloud(tmpPC);
    downSizeFilter.setLeafSize(0.1, 0.1, 0.1);
    downSizeFilter.filter(*tmpPC_filtered);
    tmpPC.swap(tmpPC_filtered);
	}*/

    // ROS_INFO("after convert pc at time = %f points: %d", time, tmpPC->points.size()); 
    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpPC2(new pcl::PointCloud<pcl::PointXYZI>); 
    // tmpPC2->points.resize(tmpPC->points.size()); 
    tmpPC2->points.reserve(tmpPC->points.size());
    for(int i=0; i<tmpPC->points.size(); i++)
    {
    	pcl::PointXYZRGB& pt1 = tmpPC->points[i]; 
    	pcl::PointXYZI& pt2 = tmpPC2->points[i];
    	if(pt1.x != pt1.x || pt1.y != pt1.y || pt1.z != pt1.z)
    	{
    		ROS_ERROR("depth_handler: invalid point exist!");
    		continue; 
    	}
    	if(pt1.z <= 0.3 || pt1.z >= mMaxDepth)
        {
    		// ROS_DEBUG("i = %d pt.z = %f", i, pt1.z); 
            continue; 
        }
    	pt2.x = pt1.x; pt2.y = pt1.y; pt2.z = pt1.z; 
    	pt2.intensity = timeElapsed; 
    	tmpPC2->points.push_back(pt2); 
    	// ROS_INFO("i = %d pt assign pt = %f %f %f", i, pt1.x, pt1.y, pt1.z);
    }
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPointer = mCloudArray[mSyncCloudId]; 
    // ROS_INFO("mSyncCloudId = %d cloudPointer = %f", mSyncCloudId, cloudPointer);

    cloudPointer->points.clear();
    
    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
    downSizeFilter.setInputCloud(tmpPC2);
    downSizeFilter.setLeafSize(0.1, 0.1, 0.1);
    downSizeFilter.filter(*cloudPointer);
    cout <<std::fixed<<"depth_handler.cpp: finish pc at "<<time<<" with input "<<cloudPointer->points.size()<<" points!"<<endl; 
    // ROS_INFO("finish cloudHandler3");
    return ; 
}
bool DepthHandler::readParam(string config_file)
{
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "depth_handler: ERROR: Wrong path to settings" << std::endl;
	 return false; 
    }
   cv::FileNode n = fsSettings["projection_parameters"]; 
   mk[0] = static_cast<double>(n["fx"]); 
   mk[1] = static_cast<double>(n["fy"]); 
   mk[2] = static_cast<double>(n["cx"]);
   mk[3] = static_cast<double>(n["cy"]);

   mCloudDSRate = fsSettings["cloud_ds_rate"]; 
   mCloudSkip = fsSettings["cloud_skip_rate"]; 
   mMaxDepth = fsSettings["cloud_max_depth"]; 
   ROS_DEBUG("depth_handler: read param mCloudDSRate = %d mCloudSkip = %d mMaxDepth = %f", mCloudDSRate, mCloudSkip, mMaxDepth);

   fsSettings.release(); 
   return true; 
}


void DepthHandler::cloudHandler2(const sensor_msgs::Image::ConstPtr& dpt_img_msg)
{
    mCloudCnt = (mCloudCnt+1)%(mCloudSkip+1); 
    if(mCloudCnt != 0) 
	return ; 
    if(mInitTime == -1)
	mInitTime = dpt_img_msg->header.stamp.toSec(); 
    double time = dpt_img_msg->header.stamp.toSec(); 
    double timeElapsed = time - mInitTime; 
    mSyncCloudId = (mSyncCloudId+1)%(CLOUD_NUM); 
    mCloudStamp[mSyncCloudId] = time; 

    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpPC(new pcl::PointCloud<pcl::PointXYZI>());
    double halfDS = mCloudDSRate/2. - 0.5; 
    
    // median filter to get rid some noise 
    cv::Mat dpt_img = cv_bridge::toCvCopy(dpt_img_msg)->image;
    cv::Mat dst; 
    cv::medianBlur(dpt_img, dst, 5);  
    dpt_img = dst; 

    // 
   // const float* syncCloud2Pointer = reinterpret_cast<const float*>(&dpt_img_msg->data[0]);
    float scale = 0.001; 
    float min_dis = 0.3; 
    float max_dis = mMaxDepth;  // keep depth range 
    for(double i = halfDS; i < dpt_img.rows; i += mCloudDSRate)
	for(double j = halfDS; j < dpt_img.cols; j += mCloudDSRate)
	{
	    int pixelCnt = 0; 
	    float vd, vd_sum = 0; 
	    int is = (int)(i - halfDS); int ie = (int)(i + halfDS); 
	    int js = (int)(j - halfDS); int je = (int)(j + halfDS);
	    for(int ii = is; ii<= ie; ii++)
		for(int jj = js; jj<= je; jj++)
		{
		    unsigned short _dpt = dpt_img.at<unsigned short>(ii, jj); 
		    vd = _dpt * scale; 
		    // vd = syncCloud2Pointer[ii * dpt_img.cols + jj]; 
		    if(vd > min_dis && vd < max_dis)
		    {
			pixelCnt++; 
			vd_sum += vd; 
		    }
		}
	    if(pixelCnt > 0)
	    {
		double u = (j - mk[2])/mk[0];
		double v = (i - mk[3])/mk[1]; 
		double mean_vd = vd_sum / pixelCnt; 
		pcl::PointXYZI pt;
		pt.x = u * mean_vd; 
		pt.y = v * mean_vd;
		pt.z = mean_vd; 
		pt.intensity = timeElapsed;
		tmpPC->points.push_back(pt); 
	    }
	}
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPointer = mCloudArray[mSyncCloudId]; 
    cloudPointer->clear();
    
    // cout <<std::fixed<<"depth_handler.cpp: receive "<<time<<" input "<<tmpPC->points.size()<<" points!"<<endl; 

    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
    downSizeFilter.setInputCloud(tmpPC);
    downSizeFilter.setLeafSize(0.1, 0.1, 0.1);
    downSizeFilter.filter(*cloudPointer);
    
    // cout <<"depth_handler.cpp: after downsampling cloudPointer has "<<cloudPointer->points.size()<<" points!"<<endl;

    return ; 

}

// template<int CLOUD_NUM>
void DepthHandler::cloudHandler(const sensor_msgs::Image::ConstPtr& dpt_img_msg)
{
    mCloudCnt = (mCloudCnt+1)%(mCloudSkip+1); 
    if(mCloudCnt != 0) 
	return ; 
    if(mInitTime == -1)
	mInitTime = dpt_img_msg->header.stamp.toSec(); 
    double time = dpt_img_msg->header.stamp.toSec(); 
    double timeElapsed = time - mInitTime; 
    mSyncCloudId = (mSyncCloudId+1)%(CLOUD_NUM); 
    mCloudStamp[mSyncCloudId] = time; 

    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpPC(new pcl::PointCloud<pcl::PointXYZI>());
    double halfDS = mCloudDSRate/2. - 0.5; 
    
    cv::Mat dpt_img = cv_bridge::toCvCopy(dpt_img_msg)->image;
    
    // 
    const float* syncCloud2Pointer = reinterpret_cast<const float*>(&dpt_img_msg->data[0]);
    float scale = 0.001; 
    float min_dis = 0.3; 
    float max_dis = 7;  // keep depth range 
    for(double i = halfDS; i < dpt_img.rows; i += mCloudDSRate)
	for(double j = halfDS; j < dpt_img.cols; j += mCloudDSRate)
	{
	    int pixelCnt = 0; 
	    float vd, vd_sum = 0; 
	    int is = (int)(i - halfDS); int ie = (int)(i + halfDS); 
	    int js = (int)(j - halfDS); int je = (int)(j + halfDS);
	    for(int ii = is; ii<= ie; ii++)
		for(int jj = js; jj<= je; jj++)
		{
		    // unsigned short _dpt = dpt_img.at<unsigned short>(ii, jj); 
		    // vd = _dpt * scale; 
		    vd = syncCloud2Pointer[ii * dpt_img.cols + jj]; 
		    if(vd > min_dis && vd < max_dis)
		    {
			pixelCnt++; 
			vd_sum += vd; 
		    }
		}
	    if(pixelCnt > 0)
	    {
		double u = (j - mk[2])/mk[0];
		double v = (i - mk[3])/mk[1]; 
		double mean_vd = vd_sum / pixelCnt; 
		pcl::PointXYZI pt;
		pt.x = u * mean_vd; 
		pt.y = v * mean_vd;
		pt.z = mean_vd; 
		pt.intensity = timeElapsed;
		tmpPC->points.push_back(pt); 
	    }
	}
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPointer = mCloudArray[mSyncCloudId]; 
    cloudPointer->clear();
    
    // cout <<std::fixed<<"depth_handler.cpp: receive "<<time<<" input "<<tmpPC->points.size()<<" points!"<<endl; 

    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
    downSizeFilter.setInputCloud(tmpPC);
    downSizeFilter.setLeafSize(0.1, 0.1, 0.1);
    downSizeFilter.filter(*cloudPointer);
    
    // cout <<"depth_handler.cpp: after downsampling cloudPointer has "<<cloudPointer->points.size()<<" points!"<<endl;

    return ; 
}

// template<int CLOUD_NUM>
void DepthHandler::voDataHandler(const nav_msgs::Odometry::ConstPtr& vo_trans)
{
    double time = vo_trans->header.stamp.toSec();
    geometry_msgs::Quaternion vo_q = vo_trans->pose.pose.orientation; 
    geometry_msgs::Point vo_t = vo_trans->pose.pose.position; 
    tf::Vector3 tj(vo_t.x, vo_t.y, vo_t.z); 
    tf::Quaternion qj(vo_q.x, vo_q.y, vo_q.z, vo_q.w); 
    tf::Transform currPose(qj, tj); 
    // cout <<"DP: receive vo at "<<std::fixed<<time<<" location: "<<vo_t.x<<" "<< vo_t.y<<" "<<vo_t.z<<endl;
    
    tf::Quaternion qi = mLastPose.getRotation(); 
    tf::Vector3 ti = mLastPose.getOrigin(); 
    tf::Transform Tj = currPose; 
    tf::Transform Tij = mLastPose.inverse() * currPose; 
    tf::Transform Tji = Tij.inverse(); 
    pcl::PointCloud<pcl::PointXYZI>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZI>); 

    // process last recorded point cloud 
    if(time - mTimeRec < 0.5 && mSyncCloudId >= 0)
    {
	pcl::PointXYZI pj; 
	int recPtNum = mCloudRec->points.size(); 
        // cout <<std::fixed<<"DP: handle vo at "<<time<<" lastPC has "<<recPtNum<<" points!"<<endl;
	for(int i=0; i<recPtNum; i++)
	{
	    pcl::PointXYZI pi = mCloudRec->points[i]; 
	    pj = pi; 
	    tf::Vector3 tf_pi(pi.x, pi.y, pi.z);
	    tf::Vector3 tf_pj = Tji * tf_pi; 
	    pj.x = tf_pj.x(); pj.y = tf_pj.y(); pj.z = tf_pj.z(); 
	    double elasp_time = time - mInitTime - pj.intensity;
	    if(fabs(pj.z) >0.3 && fabs(pj.x/pj.z) < 2 && fabs(pj.y/pj.z) < 1 && tf_pj.length() < 15 && elasp_time <= 5.)
		tmp->points.push_back(pj); 
	}
    }
    // cout <<"DP: after transform lastPC now tempCloud has "<<tmp->points.size()<<" points!"<<endl;

    while(mCloudStamp[mRegCloudId] <= time && mRegCloudId != (mSyncCloudId + 1)%CLOUD_NUM)
    {
	double ratio = (time - mCloudStamp[mRegCloudId])/(time - mTimeRec); 
	ratio = ratio > 1 ? 1:ratio; 
	ratio = ratio < 0 ? 0:ratio; 
	tf::Quaternion qt = (qi.slerp(qj, 1 - ratio)).normalized(); 
	tf::Vector3 tt = ti.lerp(tj, 1 - ratio); 
	tf::Transform Tt(qt, tt); 
	tf::Transform Tjt = Tj.inverse() * Tt; 
	pcl::PointCloud<pcl::PointXYZI>::Ptr ptrCP = mCloudArray[mRegCloudId]; 
	int numCP = ptrCP->points.size(); 
	for(int i=0; i<numCP; i++)
	{
	    pcl::PointXYZI ptt = ptrCP->points[i]; 
	    pcl::PointXYZI pj = ptt; 
	    tf::Vector3 tf_ptt(ptt.x, ptt.y, ptt.z); 
	    tf::Vector3 tf_pj = Tjt * tf_ptt; 
	    pj.x = tf_pj.x(); pj.y = tf_pj.y(); pj.z = tf_pj.z(); 
	    if(fabs(pj.z) >0.3 && fabs(pj.x/pj.z) < 2 && fabs(pj.y/pj.z) < 1.5 && tf_pj.length() < 15)
		tmp->points.push_back(pj); 
	}

	mRegCloudId = (mRegCloudId + 1) % CLOUD_NUM;    
    }

    // cout<<"DP: after register tempCloud has "<<tmp->points.size() <<" points"<<endl;

    mCloudRec->clear(); 
    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
    downSizeFilter.setInputCloud(tmp);
    downSizeFilter.setLeafSize(0.05, 0.05, 0.05);
    downSizeFilter.filter(*mCloudRec);
    int depthCloudNum = mCloudRec->points.size();

    tmp->clear();
    for(int i=0; i<depthCloudNum; i++)
    {
	pcl::PointXYZI pt = mCloudRec->points[i]; 
	if(fabs(pt.x/pt.z) < 1 && fabs(pt.y/pt.z) < 0.6)
	{
	    pcl::PointXYZI pp = pt; 
	    pp.intensity = pt.z;
	    pp.x = pp.x * (mZoomDis / pt.z); 
	    pp.y = pp.y * (mZoomDis / pt.z);
	    pp.z = mZoomDis; 
	    tmp->points.push_back(pp); 
	}
    }
    // cout <<"DP: after downsample depth cloud has "<<depthCloudNum<<" points"<<endl;

    // pcl::PointCloud<pcl::PointXYZI>::Ptr tmp2(new pcl::PointCloud<pcl::PointXYZI>); 
    mCloudPub->clear();
    downSizeFilter.setInputCloud(tmp);
    downSizeFilter.setLeafSize(0.1, 0.1, 0.1);
    downSizeFilter.filter(*mCloudPub);
    int tempCloud2Num = mCloudPub->points.size();
/*
    for(int i=0; i<tempCloud2Num; i++)
    {
	mCloudPub->points[i].z = mCloudPub->points[i].intensity;
	mCloudPub->points[i].x = mCloudPub->points[i].x * mCloudPub->points[i].z / mZoomDis; 
	mCloudPub->points[i].y = mCloudPub->points[i].y * mCloudPub->points[i].z / mZoomDis; 
	mCloudPub->points[i].intensity = mZoomDis;
    }*/
    mTimeRec = time; 
    mLastPose = currPose; 
    // cout <<"DP: after downsample again depth cloud has "<<tempCloud2Num<<" points to publish "<<endl;
}


namespace
{
    void decleration()
    {
	// DepthHandler<5>* dp = new DepthHandler<5>();
    }

}
