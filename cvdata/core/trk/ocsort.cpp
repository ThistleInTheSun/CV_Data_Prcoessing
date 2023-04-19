#ifndef OCSORT
#define OCSORT

#include <iostream>
#include <map>
#include <functional>
#include <vector>
// #include "Eigen/Dense"
using namespace std;


// // Computes IOU between two bboxes in the form [x1,y1,x2,y2] 
// double iou_batch(double** bboxes1, double** bboxes2) { 
//     // bboxes2 is a 1 x n2 array 
//     // bboxes1 is a n1 x 1 array 
//     // , int n1, int n2
//     double xx1 = std::max(bboxes1[0][0], bboxes2[0][0]); 
//     double yy1 = std::max(bboxes1[0][1], bboxes2[0][1]); 
//     double xx2 = std::min(bboxes1[0][2], bboxes2[0][2]); 
//     double yy2 = std::min(bboxes1[0][3], bboxes2[0][3]); 
//     double w = std::max(0.0, xx2 - xx1); 
//     double h = std::max(0.0, yy2 - yy1); 
//     double wh = w * h; 
//     double o = wh / ((bboxes1[0][2] - bboxes1[0][0]) * (bboxes1[0][3] - bboxes1[0][1]) + (bboxes2[0][2] - bboxes2[0][0]) * (bboxes2[0][3] - bboxes2[0][1]) - wh); 
//     return o; 
// }


// std::map<std::string, std::function<double(double**, double**)>> ASSO_FUNCS;
// ASSO_FUNCS["iou"]= iou_batch;


class OCSort {
private:
    double det_thresh = 0.5;
    int max_age = 30;
    int min_hits = 3;
    double iou_threshold = 0.3;
    // std::vector<KalmanBoxTracker> trackers;
    int frame_count = 0;
    int delta_t = 3;
    std::function<double(std::vector<double>, std::vector<double>)> asso_func;
    double inertia = 0.2;
    bool use_byte = true;

public:
    void init(int det_thresh) {
        this->det_thresh = det_thresh;
    };
    void update(int det_thresh) {
        cout << this->det_thresh << endl;
    };
};

extern "C" {
    OCSort obj;

    void init(int output_results) {
        obj.init(output_results);
    }
    void update(int det_thresh) {
        obj.update(det_thresh);
    }
}



#endif