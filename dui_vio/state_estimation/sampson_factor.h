/*
	Aug. 21, 2019, He Zhang, hzhang8@vcu.edu 

	sampson approximation to the geometric distance

*/


#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"

/*
struct SampsonCostFunctor
{
public:
	SampsonCostFunctor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j, 
		const Eigen::Matrix4d& _sqrt_info):
	pts_i_(_pts_i), pts_j_(_pts_j), sqrt_info(_sqrt_info){}

	template<typename T>
	bool operator()( T const *const *parameters, T *residuals) const{

		typename Eigen::Matrix<T, 3, 1> Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    	typename Eigen::Quaternion<T> Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    	typename Eigen::Matrix<T, 3, 1> Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    	typename Eigen::Quaternion<T> Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    	typename Eigen::Matrix<T, 3, 1> tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    	typename Eigen::Quaternion<T> qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    	T inv_dep_i = parameters[3][0];
    	typename Eigen::Matrix<T, 3, 1> pts_i = pts_i_.cast<T>(); 
    	typename Eigen::Matrix<T, 3, 1> pts_j = pts_j_.cast<T>(); 

    	typename Eigen::Matrix<T, 3, 1> pts_camera_i = pts_i / inv_dep_i;
    	typename Eigen::Matrix<T, 3, 1> pts_imu_i = qic * pts_camera_i + tic;
    	typename Eigen::Matrix<T, 3, 1> pts_w = Qi * pts_imu_i + Pi;
    	typename Eigen::Matrix<T, 3, 1> pts_imu_j = Qj.inverse() * (pts_w - Pj);
    	typename Eigen::Matrix<T, 3, 1> pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    
    	T dep_j = pts_camera_j.z();
    	typename Eigen::Matrix<T, 2, 1> epsilon;
    	epsilon(0) = (pts_camera_j(0) / dep_j) - pts_j(0);
    	epsilon(1) = (pts_camera_j(1) / dep_j) - pts_j(1);

	    // compute J = d_epsilon_d_X, [2x4], where X = (pi_x, pi_y, pj_x, pj_y) 
	    typename Eigen::Matrix<T, 2, 3> d_epsilon_d_pts_cam_j;
	    d_epsilon_d_pts_cam_j << 1./dep_j, 0, -pts_camera_j.x()/(dep_j*dep_j), 
	    						0, 1./dep_j, -pts_camera_j.y()/(dep_j*dep_j); 
	    typename Eigen::Matrix<T, 3, 3> d_pts_cam_j_d_pts_cam_i; 
	    d_pts_cam_j_d_pts_cam_i = qic.inverse()*Qj.inverse()*Qi*qic; 
	    typename Eigen::Matrix<T, 3, 2> d_pts_cam_i_d_pts_i; 
	    d_pts_cam_i_d_pts_i << 1./inv_dep_i, 0, 
	    					   0, 1./inv_dep_i,
	    					   0, 0;

	    typename Eigen::Matrix<T, 2, 2> d_epsilon_d_pts_i = d_epsilon_d_pts_cam_j * d_pts_cam_j_d_pts_cam_i * d_pts_cam_i_d_pts_i; 
	   	typename Eigen::Matrix<T, 2, 2> d_epsilon_d_pts_j = Eigen::Matrix<T,2,2>::Identity*(T(-1)); 
	    typename Eigen::Matrix<T, 2, 4> d_epsilon_d_X; 
	    d_epsilon_d_X.block<2,2>(0, 0) = d_epsilon_d_pts_i; 
	    d_epsilon_d_X.block<2,2>(0, 2) = d_epsilon_d_pts_j;

	    typename Eigen::Matrix<T, 2, 4> J = d_epsilon_d_X; 
	    typename Eigen::Matrix<T, 2, 2> JtJ = J*J.transpose(); 

	    typename Eigen::Map<Eigen::Matrix<T, 4,1>> residual(residuals);
	    residual = -J.transpose()*JtJ.inverse()*epsilon;
	    // residual = sqrt_info*residual;

	    return true; 
	}

private:
	Eigen::Vector3d pts_i_, pts_j_;
	Eigen::Matrix4d sqrt_info; 
	
};
*/
class SampsonFactor : public ceres::SizedCostFunction<4, 7, 7, 7, 1>
{
public:
	SampsonFactor(const Eigen::Vector3d& _pts_i, const Eigen::Vector3d& _pts_j);
		//const SampsonCostFunctor* functor); 

	virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

	Eigen::Vector3d pts_i, pts_j;
	static Eigen::Matrix<double, 4, 4> sqrt_info;  

	void check(double **parameters); 

// private:
	// std::unique_ptr<const SampsonCostFunctor> functor_;

};

class SampsonFactorCross : public ceres::SizedCostFunction<4, 7, 7, 7, 1>
{
public:
	SampsonFactorCross(const Eigen::Vector3d& _pts_i, const Eigen::Vector3d& _pts_j);
		//const SampsonCostFunctor* functor); 

	virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

	Eigen::Vector3d pts_i, pts_j;
	static Eigen::Matrix<double, 4, 4> sqrt_info;  

	void check(double **parameters); 

	void compute_Jacobian_pose(double const *const *parameters, double** jacobians) const; 
	// void compute_Jacobian_pose_j(double const *const *parameters, double** jacobians) const; 


// private:
	// std::unique_ptr<const SampsonCostFunctor> functor_;

};

class SampsonFactorEssential : public ceres::SizedCostFunction<4, 7, 7, 7>
{
public:
	SampsonFactorEssential(const Eigen::Vector3d& _pts_i, const Eigen::Vector3d& _pts_j);
		//const SampsonCostFunctor* functor); 

	virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

	Eigen::Vector3d pts_i, pts_j;
	static Eigen::Matrix<double, 4, 4> sqrt_info;  

	void compute_Jacobian_pose(double const *const *parameters, double** jacobians) const; 
	
	void check(double **parameters); 

};

class SampsonFactorWithLambda : public ceres::SizedCostFunction<5, 7, 7, 7, 1>
{
public:
	SampsonFactorWithLambda(const Eigen::Vector3d& _pts_i, const Eigen::Vector3d& _pts_j);

	virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

	Eigen::Vector3d pts_i, pts_j;
	static Eigen::Matrix<double, 5, 5> sqrt_info;  

	void check(double **parameters); 

};


class SampsonFactorCrossWithLambda : public ceres::SizedCostFunction<5, 7, 7, 7, 1>
{
public:
	SampsonFactorCrossWithLambda(const Eigen::Vector3d& _pts_i, const Eigen::Vector3d& _pts_j);

	virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

	Eigen::Vector3d pts_i, pts_j;
	static Eigen::Matrix<double, 5, 5> sqrt_info;  

	void check(double **parameters); 

};