#ifndef OCSORT
#define OCSORT

#include <iostream>
#include <map>
#include <functional>
#include <vector>
#include <typeinfo>
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

    double low_thresh = 0.1;
    double high_thresh = 0.5;

public:
    void init(int det_thresh) {
        this->det_thresh = det_thresh;
    };
    void update(vector<vector<double>>& dets, vector<vector<double>>& res);
    // void test(double** dets, int m, int n);
};


void OCSort::update(vector<vector<double>>& dets, vector<vector<double>>& res) {
    this->frame_count += 1; 

    vector<vector<double>> dets_first; 
    vector<vector<double>> dets_second; 
    cout << dets.size() << endl;
    for (auto& row : dets) {
        cout << "in for" << endl;
        cout << typeid(row).name() << endl;
        for (auto& i : row) {
            cout << i << endl;
        }
        // cout << row << endl;
        cout << row.at(4) << endl;
        if (row[4] >= this->high_thresh) {
            cout << "in if 1" << endl;
            dets_first.push_back(row);
            cout << "in if 2" << endl;
        } else if (row[4] >= this->low_thresh) {
            dets_second.push_back(row);
        }
    }
    cout << "bb3" << endl;
}

// void OCSort::test(double** dets, int* m, int* n) {
//     cout << "in test" << endl;
//     cout << m << "--" << n << endl;
//     cout << dets[0][0] << m << n << endl;
//     // for (auto& row : dets) {
//     //     cout << "test" << endl;
//     //     cout << "in for" << endl;
//     //     cout << typeid(row).name() << endl;
//     //     cout << "in for 1" << endl;
//     //     cout << row << endl;
//     // }
//     cout << "bb3" << endl;
// }

extern "C" {
    OCSort obj;

    void init(int dets) {
        obj.init(dets);
    }
    void update(vector<vector<double>>& dets, vector<vector<double>>& res) {
        cout << "bb" << endl;
        obj.update(dets, res);
        cout << "cc" << endl;
    }
    void test(double** dets, int m, int n) {
        cout << "in extern" << endl;
        cout << m << ", " << n << endl;
        cout << dets << endl;
        cout << dets[0][0] << endl;

        for (int i = 0; i < m; i++) { 
            for (int j = 0; j < n; j++) { 
                cout << dets[i][j] << endl;
            }
        }
        // obj.test(dets, m, n);
        cout << "end extern" << endl;
    }
}



#endif