#ifndef OCSORT
#define OCSORT

#include <iostream>
#include <map>
#include <functional>
#include <vector>
#include <typeinfo>
#include <Eigen/Dense>

using namespace std;


class KalmanBoxTracker {
public:
    void init(int dim_x, int dim_z, int dim_u=0) {
        if (dim_x < 1) {
            throw std::invalid_argument("dim_x must be 1 or greater");
        }
        if (dim_z < 1) {
            throw std::invalid_argument("dim_z must be 1 or greater");
        }
        if (dim_z < 0) {
            throw std::invalid_argument("dim_u must be 0 or greater");
        }

        

        this->dim_x = dim_x;
        this->dim_z = dim_z;
        this->dim_u = dim_u;

        // VectorXd x(dim_x);
        MatrixXd x(dim_x, 1);
        x.setZero(); // initialize state vector to zero

        MatrixXd P(dim_x, dim_x); // uncertainty covariance
        P.setIdentity(); // initialize uncertainty covariance to identity matrix

        MatrixXd Q(dim_x, dim_x); // process uncertainty
        MatrixXd B; // control transition matrix
        MatrixXd F(dim_x, dim_x); // state transition matrix
        MatrixXd H(dim_z, dim_x); // measurement function
        MatrixXd R(dim_z, dim_z); // measurement uncertainty
        MatrixXd M(dim_x, dim_z); // process-measurement cross correlation
        MatrixXd z(dim_z, 1); // measurement vector
        double _alpha_sq = 1.; // fading memory control

        Q.setIdentity(); // initialize process uncertainty to identity matrix
        F.setIdentity(); // initialize state transition matrix to identity matrix
        H.setZero(); // initialize measurement function to zero matrix
        R.setIdentity(); // initialize measurement uncertainty to identity matrix
        M.setZero(); // initialize cross correlation matrix to zero matrix

        MatrixXd K(dim_x, dim_z); K.setZero();
        MatrixXd y(dim_z, 1); y.setZero();
        MatrixXd S(dim_z, dim_z); S.setZero();
        MatrixXd SI(dim_z, dim_z); SI.setZero();
        MatrixXd I(dim_x, dim_x); SI.setIdentity();

    };
    
    
    void predict();
    void update(MatrixXd z);
};


void OCSort::predict() {
    this->x = this->F * this->x;
    this->P = this->_alpha_sq * this->F * this->P * this->F.transition() + this->Q;

    this->x_prior = this->x;
    this->P_prior = this->P;
}

void OCSort::update(MatrixXd z) {
    this->y = z - this->H * this->x;
    MatrixXd PHT = this->P * this->H.transition();

    this->S = this->H * PHT + this->R;
    this-SI = this->S.inverse();

    this->K = PHT * this->SI;

    this->x = this->x + this->K * this->y;

    I_KH = this->_I - this->K * this->H;
    this->P = I_KH * this->P * I_KH.transition() + this->K * this-> R * this-> K.transition();

    this->x_post = this->x;
    this->P_post = this->P;
}



#endif