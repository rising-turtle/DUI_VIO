/*
    Aug. 20 2018, He Zhang, hzhang8@vcu.edu 
    
    A projection factor using quaternion in ceres
*/

#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"

namespace Eigen
{
typedef Eigen::Matrix<double, 3, 2> Matrix32; 
typedef Eigen::Matrix<double, 6, 2> Matrix62; 
}

namespace QUATERNION_VIO 
{

class Unit3
{   
  public:
    Unit3();
    Unit3(Eigen::Vector3d& n); 
    ~Unit3();
    Eigen::Matrix32 getBasis(Eigen::Matrix62* H = NULL) ;
    Eigen::Vector3d p_; 
    Eigen::Matrix32* B_; 
    Eigen::Matrix62* H_B_; 
};

class PlaneFactor_P1 : public ceres::SizedCostFunction<3, 7>
{
  public:
    PlaneFactor_P1(const Eigen::Matrix<double,4,1>& plane_g, const Eigen::Matrix<double, 4, 1>& plane_l); 
    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const;
    void check(double ** parameters);
    // Eigen::Vector3d nv_g, nv_l;  // normal vector
    Unit3 nv_g; 
    Unit3 nv_l;
    double d_g, d_l;  // distance 
    Eigen::Matrix3d sqrt_info; 
};

class ProjectionFactor_Y2 : public ceres::SizedCostFunction<1, 7, 7, 7>
{
  public:
    ProjectionFactor_Y2(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j;
    // Eigen::Matrix<double, 2, 3> tangent_base;
    // Eigen::Matrix2d sqrt_info;
    double sqrt_info; 
    double scale; 
};

/*
class ProjectionFactor_Y34 : public ceres::SizedCostFunction<2, 7, 7, 7, 1>
{
  public:
    ProjectionFactor_Y3(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    // void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j;
    // Eigen::Matrix<double, 2, 3> tangent_base;
    // Eigen::Matrix2d sqrt_info;
    double sqrt_info; 
};

class ProjectionFactor_Y4 : public ceres::SizedCostFunction<1, 7>
{
  public:
    ProjectionFactor_Y4(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    // void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j;
    // Eigen::Matrix<double, 2, 3> tangent_base;
    // Eigen::Matrix2d sqrt_info;
    double sqrt_info; 
};*/

class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1>
{
  public:
    ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix2d sqrt_info;
};

class PoseLocalPrameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};

}
