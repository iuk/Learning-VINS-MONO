#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <cstdlib>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;

struct SFMFeature {
  bool state;
  int id;
  vector<pair<int, Vector2d>> observation;   // frame_id:归一化平面 xy
  double position[3];
  double depth;
};

// 重投影误差的 残差函数定义
struct ReprojectionError3D {
  // 传入参数，特征点在某帧上的 归一化平面 uv
  ReprojectionError3D(double observed_u, double observed_v)
      : observed_u(observed_u), observed_v(observed_v) {}

  // 残差块 计算
  template <typename T>
  bool operator()(const T *const camera_R,  // 输入 相机姿态 4x
                  const T *const camera_T,  // 输入 相机位置 3x
                  const T *point,           // 输入 3D 空间点 3x
                  T *residuals)             // 输出 残差向量 2x
  const {
    T p[3];
    // 3D 空间点应用旋转和平移
    ceres::QuaternionRotatePoint(camera_R, point, p);
    p[0] += camera_T[0];
    p[1] += camera_T[1];
    p[2] += camera_T[2];
    // 到达归一化平面
    T xp         = p[0] / p[2];
    T yp         = p[1] / p[2];
    // 计算距离
    residuals[0] = xp - T(observed_u);
    residuals[1] = yp - T(observed_v);
    return true;
  }

  static ceres::CostFunction *Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<
            ReprojectionError3D, 2, 4, 3, 3>(
        new ReprojectionError3D(observed_x, observed_y)));
  }

  double observed_u;
  double observed_v;
};

class GlobalSFM {
 public:
  GlobalSFM();
  bool construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
                 const Matrix3d relative_R, const Vector3d relative_T,
                 vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

 private:
  bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

  void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
  void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
                            int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                            vector<SFMFeature> &sfm_f);

  int feature_num;
};