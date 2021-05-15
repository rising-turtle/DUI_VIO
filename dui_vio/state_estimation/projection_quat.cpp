/*
    Aug. 20 2018, He Zhang, hzhang8@vcu.edu 
    
    A projection factor using quaternion in ceres 
*/

#include "projection_quat.h"
#include "parameters.h"
#include <fstream>
#include <iostream>

using namespace std; 
using namespace Eigen;

namespace QUATERNION_VIO{


Unit3::Unit3():
p_(0, 0, 1.),
B_(NULL),
H_B_(NULL){}

Unit3::Unit3(Eigen::Vector3d& n): p_(n.normalized()),
B_(NULL),
H_B_(NULL)
{}

Unit3::~Unit3()
{
    if(B_) 
    {
        delete B_; 
        B_ = NULL; 
    }
    if(H_B_)
    {
        delete H_B_;
        H_B_ = NULL;
    }
}
Eigen::Matrix<double, 3, 2> Unit3::getBasis(Eigen::Matrix62* H) 
{
    if(B_ && !H)
        return *B_;
    if(B_ && H && H_B_)
    {
        *H = *H_B_;
        return *B_;
    }
    Eigen::Vector3d n = p_; 
    Eigen::Vector3d axis(0, 0, 1); 
    double mx = fabs(n(0)); double my = fabs(n(1)); double mz = fabs(n(2)); 
    if((mx <= my) && (mx <= mz))
    {
        axis = Eigen::Vector3d(1.0, 0., 0.);
    }else if((my <= mx) && (my <= mz))
    {
        axis = Eigen::Vector3d(0., 1.0, 0.);
    }
  
    Eigen::Vector3d B1 = n.cross(axis); 
    Eigen::Vector3d b1 = B1/B1.norm();
    Eigen::Vector3d b2 = n.cross(b1); 
    if(B_ == NULL)
        B_ = new Eigen::Matrix32; 
    *(B_) << b1.x(), b2.x(), b1.y(), b2.y(), b1.z(), b2.z(); 

    if(H)
    {
        Eigen::Matrix<double, 3 ,3> d_B1_d_n  = Utility::skewSymmetric(-axis); 
        double bx = B1.x(); double by = B1.y(); double bz = B1.z(); 
        double bx2 = bx*bx; double by2 = by*by; double bz2 = bz*bz; 
        Eigen::Matrix<double, 3, 3> d_b1_d_B1; 
        d_b1_d_B1 << by2+bz2, -bx*by, -bx*bz, -bx*by, bx2+bz2, -by*bz, -bx*bz, -by*bz, bx2+by2;
        d_b1_d_B1 /= std::pow(bx2 + by2 + bz2, 1.5);
        Eigen::Matrix<double, 3 ,3> d_b2_d_n, d_b2_d_b1; 
        d_b2_d_n = Utility::skewSymmetric(-b1); 
        d_b2_d_b1 = Utility::skewSymmetric(n); 

        Matrix32& d_n_d_p = *B_; 
        Matrix32 d_b1_d_p = d_b1_d_B1 * d_B1_d_n * d_n_d_p; 
        Matrix32 d_b2_d_p = d_b2_d_b1 * d_b1_d_p + d_b2_d_n * d_n_d_p; 
        if(H_B_ == NULL)
            H_B_ = new Eigen::Matrix62; 
        (*H_B_) << d_b1_d_p, d_b2_d_p; 
        *H = *H_B_;
    }

    return (*B_); 
}

PlaneFactor_P1::PlaneFactor_P1(const Eigen::Matrix<double,4,1>& plane_g, const Eigen::Matrix<double, 4, 1>& plane_l)
{
    Eigen::Vector3d nv_g_e = plane_g.block<3,1>(0,0); 
    nv_g = Unit3(nv_g_e); 
    Eigen::Vector3d nv_l_e = plane_l.block<3,1>(0,0); 
    nv_l = Unit3(nv_l_e); 

    d_g = plane_g(3); 
    d_l = plane_l(3); 
    sqrt_info = Eigen::Matrix3d::Identity() * 700.; 
}

bool PlaneFactor_P1::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]); 

    // plane_l = plane_g.transform(Tgl) 
    Eigen::Quaterniond Qi_inv = Qi.inverse(); 
    Eigen::Vector3d nl_e = Qi_inv*nv_g.p_; 
    nl_e /= nl_e.norm();
    Unit3 nl(nl_e);
    double dl = nv_g.p_.dot(Pi) + d_g;
   
    Eigen::Map<Eigen::Vector3d> residual(residuals);
    Eigen::Matrix62 d_B_d_p; 
    Eigen::Matrix<double, 3, 2> B = nv_l.getBasis(&d_B_d_p); 
    Eigen::Matrix<double, 2, 3> Bt = B.transpose();
    residual.block<2,1>(0,0) = Bt * nl.p_; 
    residual(2) = d_l - dl; 
    residual = sqrt_info * residual; 

    if(jacobians)
    {
        Eigen::Matrix<double, 3, 6> d_nl_dTgl = Eigen::Matrix<double, 3, 6>::Zero(); 
        d_nl_dTgl.block<2,3>(0,0) = nl.getBasis().transpose() * Utility::skewSymmetric(nl.p_);
        d_nl_dTgl.block<1,3>(2,3) = nl.p_.transpose();   
        // Eigen::Matrix<double, 1, 3> d_r1_d_b1 = nl.p_.transpose(); 
        // Eigen::Matrix<double, 1, 3> d_r2_d_b2 = nl.p_.transpose();
        // Eigen::Matrix<double, 3, 2> d_b1_d_p = d_B_d_p.block<3,2>(0,0); 
        // Eigen::Matrix<double, 3, 2> d_b2_d_p = d_B_d_p.block<3,2>(3,0); 
        // Eigen::Matrix<double, 1, 2> d_r1_d_p = d_r1_d_b1 * d_b1_d_p; 
        // Eigen::Matrix<double, 1, 2> d_r2_d_p = d_r2_d_b2 * d_b2_d_p; 
        // Eigen::Matrix<double, 2, 2> d_r_d_p; 
        // d_r_d_p << d_r1_d_p, d_r2_d_p;
        Eigen::Matrix<double, 2,2> d_r_d_nl; 
        Matrix32 d_nlp_d_nl = nl.getBasis(); 
        Matrix<double,2,3> d_r_d_nlp = Bt; 
        d_r_d_nl = d_r_d_nlp * d_nlp_d_nl; 
	    Eigen::Matrix<double, 2, 6> d_nl_dTgl_2 = d_nl_dTgl.block<2,6>(0,0);
        Eigen::Matrix<double, 2, 6> d_r_d_Tgl = d_r_d_nl * d_nl_dTgl_2; 
        Eigen::Matrix<double, 1, 6> d_r3_d_Tgl; 
        d_r3_d_Tgl << -nv_g.p_(0), -nv_g.p_(1), -nv_g.p_(2), 0, 0, 0; 

        if(jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 3,7, Eigen::RowMajor> > jacobian_pose_i(jacobians[0]); 
            // dnv_dt = 0
            jacobian_pose_i.block<2,3>(0,0) = d_r_d_Tgl.block<2,3>(0, 3);

            // dnv_dqi 
            jacobian_pose_i.block<2,3>(0,3) = d_r_d_Tgl.block<2,3>(0,0);
            
            // dd_dq = 0, 
            jacobian_pose_i.block<1,6>(2,0) = d_r3_d_Tgl; 
            
	        jacobian_pose_i.block<3,6>(0,0) = sqrt_info * jacobian_pose_i.block<3,6>(0,0); 

            jacobian_pose_i.rightCols<1>().setZero(); 
        }
    }

    return true; 
}

void PlaneFactor_P1::check(double ** parameters)
{
    double *res = new double[1]; 
    double **jaco = new double *[1]; 
    jaco[0] = new double[3*7]; 
    Evaluate(parameters, res, jaco); 
    puts("PlaneFactor_P1 check begins"); 
    puts("my"); 
    
    Eigen::Map<Eigen::Vector3d > res_v(res); 

    cout <<"res: "<<res_v.transpose()<<endl; 
    cout <<Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> >(jaco[0]) <<endl; 
  
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]); 

    Eigen::Quaterniond Qi_inv = Qi.inverse(); 
    Eigen::Vector3d nl_e = Qi_inv*nv_g.p_; 
    nl_e /= nl_e.norm();
    Unit3 nl(nl_e);
    double dl = nv_g.p_.dot(Pi) + d_g;

    Eigen::Matrix<double, 3, 3> sqrt_info = Eigen::Matrix<double, 3, 3>::Identity()*700.;   
    Eigen::Vector3d y2;
    Eigen::Matrix62 d_B_d_p; 
    Eigen::Matrix<double, 3, 2> B = nv_l.getBasis(&d_B_d_p); 
    Eigen::Matrix<double, 2, 3> Bt = B.transpose();
    y2.block<2,1>(0,0) = Bt * nl.p_; 
    y2(2) = d_l - dl; 
    y2 = sqrt_info * y2; 

    puts("num"); 
    cout << "res: "<<y2.transpose()<<endl; 
    
    const double eps = 1e-6;
    Eigen::Matrix<double, 3, 6> num_jacobians;
    for(int k=0; k<6;k++)
    {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]); 

        int a = k / 3 ; int b = k % 3; 
        Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b==2)*eps; 
        if(a == 0)
            Pi += delta; 
        else if(a == 1)
            Qi = Qi * Utility::deltaQ(delta); 
    
        Eigen::Vector3d tmp_y2;

        Eigen::Quaterniond Qi_inv = Qi.inverse(); 
        Eigen::Vector3d nl_e = Qi_inv*nv_g.p_; 
        nl_e /= nl_e.norm();
        Unit3 nl(nl_e);
        double dl = nv_g.p_.dot(Pi) + d_g;
       
        Eigen::Matrix62 d_B_d_p; 
        Eigen::Matrix<double, 3, 2> B = nv_l.getBasis(&d_B_d_p); 
        Eigen::Matrix<double, 2, 3> Bt = B.transpose();
        tmp_y2.block<2,1>(0,0) = Bt * nl.p_; 
        tmp_y2(2) = d_l - dl; 
        tmp_y2 = sqrt_info * tmp_y2;
	// cout<<"tmp_y2: "<<tmp_y2.transpose()<<" y2: "<<y2.transpose()<<endl;
        num_jacobians.col(k) = (tmp_y2 - y2)/eps; 
    }
    cout<<num_jacobians<<endl; 

    return ;   
}

ProjectionFactor_Y2::ProjectionFactor_Y2(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j): 
pts_i(_pts_i), pts_j(_pts_j)
{
    sqrt_info = 1.; 
    scale = 24; // 24; //24; // 10 
}

bool ProjectionFactor_Y2::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const 
{
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]); 
    
    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]); 

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]); 

    // double inv_dep_i = parameters[3][0]; 
    
    Eigen::Quaterniond Qi_c = (Qi * qic).normalized(); 
    Eigen::Vector3d Pi_c = Qi * tic + Pi; 

    Eigen::Quaterniond Qj_c = (Qj * qic).normalized(); 
    Eigen::Vector3d Pj_c = Qj * tic + Pj; 

    Eigen::Quaterniond Qij_c = (Qi_c.inverse()*Qj_c).normalized(); 
    Eigen::Vector3d Pij_c = Qi_c.inverse()*(Pj_c - Pi_c); 
    // Eigen::Quaterniond Qji_c = (Qj_c.inverse()*Qi_c).normalized(); 
    // Eigen::Vector3d Pji_c = Qj_c.inverse()*(Pi_c - Pj_c); 

    double tx = Pij_c(0);   // x
    double ty = Pij_c(1);   // y
    double tz = Pij_c(2);   // z
    double qx = Qij_c.x();  // qx
    double qy = Qij_c.y();  // qy
    double qz = Qij_c.z();  // qz
    double qw = Qij_c.w();  // qw
    
    Eigen::Quaterniond q(qw, qx, qy, qz); 
    Eigen::Matrix<double, 3, 3> R = q.toRotationMatrix(); 

    // double u0 = pts_i(0); double v0 = pts_i(1); 
    // double u1 = pts_j(0); double v1 = pts_j(1); 
    double u0 = pts_j(0); double v0 = pts_j(1); 
    double u1 = pts_i(0); double v1 = pts_i(1);

    double tmp1 = -tz * v1 + ty; 
    double tmp2 =  u1 * tz - tx;
    double tmp3 = -u1 * ty + v1 * tx;
    
    Eigen::Vector3d X0(u0, v0, 1.); 
    Eigen::Vector3d tmp0 = R * X0; 
    double y2 = tmp1* tmp0(0) + tmp2*tmp0(1) + tmp3*tmp0(2); 
  
    residuals[0] = sqrt_info * scale * y2; 
    if(jacobians )
    {
	// dy2_dPij_c 
	double dy2_dx = -tmp0(1) + v1*tmp0(2); 
	double dy2_dy = tmp0(0) - u1*tmp0(2);
	double dy2_dz = -v1*tmp0(0) + u1*tmp0(1); 
	Eigen::Vector3d dy2_dpij_c(dy2_dx, dy2_dy, dy2_dz); 

	// dy2_dQij_c
	Eigen::Vector3d dy2_dqij_c; 
	Eigen::Vector3d tp1(tmp1, tmp2, tmp3); 
	Eigen::Matrix3d dtmp0_dq = R * -Utility::skewSymmetric(X0); 
	dy2_dqij_c = tp1.transpose() * dtmp0_dq;

	// dy2_dTij_c
	Eigen::Matrix<double, 1, 6> dy2_dpij;
	dy2_dpij<< dy2_dx, dy2_dy, dy2_dz, dy2_dqij_c(0), dy2_dqij_c(1), dy2_dqij_c(2); 

	Eigen::Matrix<double, 3, 3> Rj = Qj.toRotationMatrix(); 
	Eigen::Matrix<double, 3, 3> Ri = Qi.toRotationMatrix();
	Eigen::Matrix<double, 3, 3> Ri_c = Qi_c.toRotationMatrix(); 
	Eigen::Matrix<double, 3, 3> Rij_c = Qij_c.toRotationMatrix();
	Eigen::Matrix<double, 3, 3> I3 = Eigen::Matrix<double, 3 ,3>::Identity(); 
	Eigen::Matrix<double, 3, 3> Z3 = Eigen::Matrix<double, 3, 3>::Zero(); 

	if(jacobians[0]) // dy2/dTi
	{
	    Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > jacobian_pose_i(jacobians[0]); 

	    Eigen::Matrix<double, 6, 6> jacobian_pij_pi; 
	    // dpij_c/dpi, Pij_c = Qi_c.inverse()*(Pj_c - Pi_c); Qi_c = Qi*qic;  Pi_c = Qi * tic + Pi; 
	    jacobian_pij_pi.block<3,3>(0, 0) = -Ri_c.inverse(); 
	    
	    // dpij_c/dqi, dpij_c/dqi_c = skew[Ri_c.inverse()*(Pj_c-Pi_c)]; dqi_c/dqi = qic.inverse(); 
	    // dpij_c/dpi_c = -Qi_c.inverse(); dpi_c/dqi = Qi*skew[-tic]
	    jacobian_pij_pi.block<3,3>(0, 3) = Utility::skewSymmetric(Ri_c.inverse()*(Pj_c - Pi_c)) * qic.toRotationMatrix().inverse() 
				 - Ri_c.inverse() * Ri * Utility::skewSymmetric(-tic); 
	    
	    // dqij_c/dpi, Qij_c = (Qi_c.inverse()*Qj_c).normalized(); Qi_c = (Qi * qic).normalized(); 
	    jacobian_pij_pi.block<3,3>(3, 0) = Z3;
	    // dqij_c/dqi, dqij_c/dqi_c = -Rij_c.transpose(); dqi_c/dqi = Ric.transpose(); 
	    jacobian_pij_pi.block<3,3>(3, 3) = -Rij_c.transpose() * qic.toRotationMatrix().transpose(); 
	    
	    jacobian_pose_i.leftCols<6>() = dy2_dpij * jacobian_pij_pi; 
	    jacobian_pose_i.rightCols<1>().setZero(); 

	    for(int j=0; j<6; j++)
		jacobian_pose_i[j] *= scale; 
	}
	if(jacobians[1]) // dy2/dTj 
	{
	    Eigen::Map<Eigen::Matrix<double, 1,7, Eigen::RowMajor> > jacobian_pose_j(jacobians[1]); 
	    Eigen::Matrix<double, 6, 6> jacobian_pij_pj; 
	    
	    // dpij_c/dpj, Pij_c = Qi_c.inverse()*(Pj_c - Pi_c);  Pj_c = Qj * tic + Pj; 
	    jacobian_pij_pj.block<3,3>(0, 0) = Ri_c.inverse(); 
	    // dpij_c/dqj, 
	    jacobian_pij_pj.block<3,3>(0, 3) = - Ri_c.inverse() * Rj * Utility::skewSymmetric(tic); 
	    
	    // dqij_c/dpj, Qij_c = (Qi_c.inverse()*Qj_c).normalized(); Qj_c = (Qj * qic).normalized(); 
	    jacobian_pij_pj.block<3,3>(3, 0) = Z3; 
	    // dqij_c/dqj, dqij_c/dqj_c = I3; dqj_c/dqj = qic'; 
	    jacobian_pij_pj.block<3,3>(3, 3) = qic.toRotationMatrix().transpose(); 
	    
	    jacobian_pose_j.leftCols<6>() = dy2_dpij * jacobian_pij_pj; 
	    jacobian_pose_j.rightCols<1>().setZero(); 
	    for(int j=0; j<6; j++)
		jacobian_pose_j[j] *= scale; 

	}
	if(jacobians[2]) // dy2/dTic
	{
	    Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > jacobian_ex_pose(jacobians[2]); 
	    Eigen::Matrix<double, 6,6> jacobian_pij_pic; 
	    
	    // dpij_c/dtic, Pij_c = Qi_c.inverse()*(Pj_c - Pi_c); Pj_c = Qj * tic + Pj; Pi_c = Qi * tic + Pi; 
	    jacobian_pij_pic.block<3,3>(0, 0) = Ri_c.inverse()*Rj - Ri_c.inverse()*Ri; 
	    // dpij_c/dqic, Qi_c = Qi*qic, dpij_c/dqi_c = skew[Qi_c.inverse()*(Pj_c-Pi_c)], dqi_c/dqic = I3 
	    jacobian_pij_pic.block<3,3>(0, 3) = Utility::skewSymmetric(Ri_c.inverse()*(Pj_c-Pi_c)); 

	    // dqij_c/dtic, Qij_c = (Qi_c.inverse()*Qj_c).normalized(); Qi_c = (Qi * qic).normalized; Qj_c = (Qj * qic).normalized(); 
	    jacobian_pij_pic.block<3,3>(3, 0) = Z3; 

	    // dqij_c/dqic, dqij_c/dqi_c = -Rij_c.transpose(); dqi_c/dqic = I3;  dqij_c/dqj_c = I3; dqj_c/dqic = I3
	    jacobian_pij_pic.block<3,3>(3, 3) = -Rij_c.transpose() + I3; 
	    
	    jacobian_ex_pose.leftCols<6>() = dy2_dpij * jacobian_pij_pic; 
	    jacobian_ex_pose.rightCols<1>().setZero(); 
	    for(int j=0; j<6; j++)
		jacobian_ex_pose[j] *= scale; 
	}
	// if(jacobians[3])
	// {
	    //jacobians[3][0] = 0; // 
	// }
    }
    return true; 
}

void ProjectionFactor_Y2::check(double ** parameters)
{
    double *res = new double[1]; 
    double **jaco = new double *[3]; //*[4]; 
    jaco[0] = new double[1*7]; 
    jaco[1] = new double[1*7];
    jaco[2] = new double[1*7]; 
    // jaco[3] = new double[1*1]; 
    Evaluate(parameters, res, jaco); 
    puts("ProjectionFactor_Y2 check begins"); 
    puts("my"); 

    cout <<"res: "<<res[0]<<endl; 
    cout <<Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> >(jaco[0]) <<endl; 
    cout <<Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> >(jaco[1]) <<endl; 
    cout <<Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> >(jaco[2]) <<endl; 

    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]); 
    
    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]); 

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]); 

    // double inv_dep_i = parameters[3][0]; 
    
    Eigen::Quaterniond Qi_c = (Qi * qic).normalized(); 
    Eigen::Vector3d Pi_c = Qi * tic + Pi; 

    Eigen::Quaterniond Qj_c = (Qj * qic).normalized(); 
    Eigen::Vector3d Pj_c = Qj * tic + Pj; 

    Eigen::Quaterniond Qij_c = (Qi_c.inverse()*Qj_c).normalized(); 
    Eigen::Vector3d Pij_c = Qi_c.inverse()*(Pj_c - Pi_c); 

    double tx = Pij_c(0);   // x
    double ty = Pij_c(1);   // y
    double tz = Pij_c(2);   // z
    double qx = Qij_c.x();  // qx
    double qy = Qij_c.y();  // qy
    double qz = Qij_c.z();  // qz
    double qw = Qij_c.w();  // qw
    
    Eigen::Quaterniond q(qw, qx, qy, qz); 
    Eigen::Matrix<double, 3, 3> R = q.toRotationMatrix(); 

    double u0 = pts_j(0); double v0 = pts_j(1); 
    double u1 = pts_i(0); double v1 = pts_i(1); 
    
    double tmp1 = -tz * v1 + ty; 
    double tmp2 =  u1 * tz - tx;
    double tmp3 = -u1 * ty + v1 * tx;
    
    Eigen::Vector3d X0(u0, v0, 1.); 
    Eigen::Vector3d tmp0 = R * X0; 
    double y2 = tmp1* tmp0(0) + tmp2*tmp0(1) + tmp3*tmp0(2); 
  
    y2 = sqrt_info * scale * y2; 

    puts("num"); 
    cout << "res: "<<y2<<endl; 
    
    const double eps = 1e-6;
    Eigen::Matrix<double, 1, 18> num_jacobians;
    for(int k=0; k<18;k++)
    {
	Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
	Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]); 

	Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
	Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]); 

	Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
	Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]); 
	
	int a = k / 3 ; int b = k % 3; 
	Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b==2)*eps; 
	if(a == 0)
	    Pi += delta; 
	else if(a == 1)
	    Qi = Qi * Utility::deltaQ(delta); 
	else if(a == 2)
	    Pj += delta; 
	else if(a == 3)
	    Qj = Qj * Utility::deltaQ(delta); 
	else if(a == 4)
	    tic += delta;
	else if(a == 5)
	    qic = qic * Utility::deltaQ(delta); 
	
	Eigen::Quaterniond Qi_c = (Qi * qic).normalized(); 
	Eigen::Vector3d Pi_c = Qi * tic + Pi; 

	Eigen::Quaterniond Qj_c = (Qj * qic).normalized(); 
	Eigen::Vector3d Pj_c = Qj * tic + Pj; 

	Eigen::Quaterniond Qij_c = (Qi_c.inverse()*Qj_c).normalized(); 
	Eigen::Vector3d Pij_c = Qi_c.inverse()*(Pj_c - Pi_c); 

	double tx = Pij_c(0);   // x
	double ty = Pij_c(1);   // y
	double tz = Pij_c(2);   // z
	double qx = Qij_c.x();  // qx
	double qy = Qij_c.y();  // qy
	double qz = Qij_c.z();  // qz
	double qw = Qij_c.w();  // qw

	Eigen::Quaterniond q(qw, qx, qy, qz); 
	Eigen::Matrix<double, 3, 3> R = q.toRotationMatrix(); 

	double u0 = pts_j(0); double v0 = pts_j(1); 
	double u1 = pts_i(0); double v1 = pts_i(1); 

	double tmp1 = -tz * v1 + ty; 
	double tmp2 =  u1 * tz - tx;
	double tmp3 = -u1 * ty + v1 * tx;

	Eigen::Vector3d X0(u0, v0, 1.); 
	Eigen::Vector3d tmp0 = R * X0; 
	double tmp_y2 = tmp1* tmp0(0) + tmp2*tmp0(1) + tmp3*tmp0(2); 

	tmp_y2 = sqrt_info * scale * tmp_y2; 
	num_jacobians(k) = (tmp_y2 - y2)/eps; 
    }
    cout<<num_jacobians<<endl; 
}

/*
ProjectionFactor_Y3::ProjectionFactor_Y3(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j):
pts_i(_pts_i), pts_j(_pts_j)
{
    sqrt_info = 1.; 
}

bool ProjectionFactor_Y3::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    double tx = parameters[0][0];  // x
    double ty = parameters[0][1];  // y
    double tz = parameters[0][2];  // z
    double qx = parameters[0][3];  // qx
    double qy = parameters[0][4];  // qy
    double qz = parameters[0][5];  // qz
    double qw = parameters[0][6];  // qw
    
    Eigen::Quaterniond q(qw, qx, qy, qz); 
    Eigen::Matrix<double, 3, 3> R = q.toRotationMatrix(); 

    double u0 = pts_i(0); double v0 = pts_i(1); double d0 = pts_i(2);
    double u1 = pts_j(0); double v1 = pts_j(1); 
 
    Eigen::Vector3d X0(u0*d0, v0*d0, d0); 
    Eigen::Matrix<double, 1, 3> tmp = R.row(0) - u1*R.row(2); 
    
    double y3 = tmp * X0 + tx - u1*tz; 
  
    *residuals = sqrt_info * y3; 
    
    if(jacobians && jacobians[0])
    {
	// dy3_dx
	jacobians[0][0] = 1; 
	// dy3_dy
	jacobians[0][1] = 0; 
	// dy3_dz
	jacobians[0][2] = -u1; 

	// dy3_dq
	Eigen::Matrix<double, 1, 3> dy_dq = -R.row(0)*Utility::skewSymmetric(X0) + u1*R.row(2)*Utility::skewSymmetric(X0);

	jacobians[0][3] = dy_dq(0); 
	jacobians[0][4] = dy_dq(1);
	jacobians[0][5] = dy_dq(2); 
	jacobians[0][6] = 0; 
    }
    return true; 
}

ProjectionFactor_Y4::ProjectionFactor_Y4(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j):
pts_i(_pts_i), pts_j(_pts_j)
{
    sqrt_info = 1.; 
}

bool ProjectionFactor_Y4::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    double tx = parameters[0][0];  // x
    double ty = parameters[0][1];  // y
    double tz = parameters[0][2];  // z
    double qx = parameters[0][3];  // qx
    double qy = parameters[0][4];  // qy
    double qz = parameters[0][5];  // qz
    double qw = parameters[0][6];  // qw
    
    Eigen::Quaterniond q(qw, qx, qy, qz); 
    Eigen::Matrix<double, 3, 3> R = q.toRotationMatrix(); 

    double u0 = pts_i(0); double v0 = pts_i(1); double d0 = pts_i(2);
    double u1 = pts_j(0); double v1 = pts_j(1); 
 
    Eigen::Vector3d X0(u0*d0, v0*d0, d0); 
    Eigen::Matrix<double, 1, 3> tmp = R.row(1) - v1*R.row(2); 
    double y4 = tmp * X0 + ty - v1*tz; 
    *residuals = sqrt_info * y4;

    if(jacobians && jacobians[0])
    {
	// dy4_dx/dy/dz
	jacobians[0][0] = 0;	
	jacobians[0][1] = 1;	
	jacobians[0][2] = -v1; 	
	// dy4_dq
	Eigen::Matrix<double, 1, 3> dy_dq = -R.row(1)*Utility::skewSymmetric(X0) + v1 * R.row(2) * Utility::skewSymmetric(X0); 
	
	jacobians[0][3] = dy_dq(0); 
	jacobians[0][4] = dy_dq(1);
	jacobians[0][5] = dy_dq(2); 
	jacobians[0][6] = 0; 

    }
    return true; 
}*/


bool PoseLocalPrameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();
    // for(int i=0; i<6; i++)
    //	*(x_plus_delta+i) = *(x+i) + *(delta+i);
    return true; 
}

bool PoseLocalPrameterization::ComputeJacobian(const double* x, double *jacobians) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > j(jacobians); 
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero(); 
    return true; 
}


Eigen::Matrix2d ProjectionFactor::sqrt_info;

ProjectionFactor::ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j) : pts_i(_pts_i), pts_j(_pts_j)
{
#ifdef UNIT_SPHERE_ERROR
    Eigen::Vector3d b1, b2;
    Eigen::Vector3d a = pts_j.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    b2 = a.cross(b1);
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
    // sqrt_info = Eigen::Matrix2d::Identity()*(FOCAL_LENGTH*1./1.5); 
};

bool ProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep_i = parameters[3][0];

    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Eigen::Map<Eigen::Vector2d> residual(residuals);

#ifdef UNIT_SPHERE_ERROR 
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j.normalized());
#else
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
#endif

    residual = sqrt_info * residual;

    if (jacobians)
    {
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
        double norm = pts_camera_j.norm();
        Eigen::Matrix3d norm_jaco;
        double x1, x2, x3;
        x1 = pts_camera_j(0);
        x2 = pts_camera_j(1);
        x3 = pts_camera_j(2);
        norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), - x1 * x2 / pow(norm, 3),            - x1 * x3 / pow(norm, 3),
                     - x1 * x2 / pow(norm, 3),            1.0 / norm - x2 * x2 / pow(norm, 3), - x2 * x3 / pow(norm, 3),
                     - x1 * x3 / pow(norm, 3),            - x2 * x3 / pow(norm, 3),            1.0 / norm - x3 * x3 / pow(norm, 3);
        reduce = tangent_base * norm_jaco;
#else
        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
            0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
#endif
        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
            Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
            jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                                     Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i * -1.0 / (inv_dep_i * inv_dep_i);
        }
    }
    // sum_t += tic_toc.toc();

    return true;
}

}
