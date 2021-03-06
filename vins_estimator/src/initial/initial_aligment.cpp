#include "initial_alignment.h"

//see V-B-1 in Paper
//根据视觉SFM的结果来校正陀螺仪的Bias，注意得到了新的Bias后对应的预积分需要repropagate
/**
 * @brief   陀螺仪偏置校正
 * @optional    根据视觉SFM的结果来校正陀螺仪Bias -> Paper V-B-1
 *              主要是将相邻帧之间SFM求解出来的旋转矩阵与IMU预积分的旋转量对齐
 *              注意得到了新的Bias后对应的预积分需要repropagate
 * @param[in]   all_image_frame所有图像帧构成的map,图像帧保存了位姿、预积分量和关于角点的信息
 * @param[out]  Bgs 陀螺仪偏置
 * @return      void
*/
/**
 * @brief 陀螺仪Bias修正
 * @param  all_image_frame [包含图像特征和IMU预积分的集合]
 * @param  Bgs             [陀螺仪bias]
 * 这部分可以参考文献[1]中公式(15)部分(注：'表示求逆)
 *
 * Min sum||q_cb1' × q_cb0 × γ_bkbk+1||      (1)
 *
 * Min sum||q_b1_c × q_c_b0 × γ_b0_b1||
 * 
 * 公式(15)的最小值是1(q(1,(0,0,0)，所以将其前半部分移到右边得
 * 
 *            |     1    |
 * γ_b0_b1^  ·| 1/2·J·δbw| = q_b1_c × q_c_b0        (2)
 *
 *            |     1    |
 *            | 1/2·J·δbw| = γ_b0_b1^' × q_b1_c × q_c_b0        (3)
 *
 * 只取四元数的虚部并求解，
 *            JTJ·bw = 2JT(γ_b0_b1^' × q_b1_c × q_c_b0).vec  (4)
 * 
 * 即，A·δbw=b，将多个帧综合起来为
 *            sum(A)δbw = sum(b)                              (5)
 */
// VINS MONO 论文 公式 15
// 输入 
// 输出 Bgs bias_gyr
// all_image_frame 来自于 estimator.h 中的 all_image_frame
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs) {
  Matrix3d A;
  Vector3d b;
  Vector3d delta_bg;
  A.setZero();
  b.setZero();

  // 遍历所有帧
  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) 
  {
    // 当前帧 frame_i
    // 下一帧 frame_j
    frame_j = next(frame_i);
    MatrixXd tmp_A(3, 3);
    tmp_A.setZero();
    VectorXd tmp_b(3);
    tmp_b.setZero();

    // q_ij：纯视觉得到的 i 帧系下的 j 帧姿态 
    Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
    
    //tmp_A = J_j_bw
    tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
    
    //tmp_b = 2 * (r^bk_{bk+1})^-1 * (q^c0_bk)^-1 * (q^c0_{bk+1})
    //      = 2 * (r^bk_bk+1)^-1 * q_ij
    // frame_j->second.pre_integration->delta_q.inverse()：j 系下 i 系的姿态
    // frame_j->second.pre_integration->delta_q ： IMU 预积分得到的 i 帧系下的 j 帧姿态
    tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
    
    //tmp_A * delta_bg = tmp_b
    A += tmp_A.transpose() * tmp_A;
    b += tmp_A.transpose() * tmp_b;
  }

  // Eigen ldlt 线性方程求解 Ax=b
  delta_bg = A.ldlt().solve(b);
  ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

  // 因为求解出的Bias是变化量，所以要累加
  // question：这个地方滑窗内的累加深入理解
  for (int i = 0; i <= WINDOW_SIZE; i++)
    Bgs[i] += delta_bg;

  // 利用新的Bias重新repropagate
  for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) {
    frame_j = next(frame_i);
    frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
  }
}

//Algorithm 1 to find b1 b2
//在半径为G的半球找到切面的一对正交基
//求法跟论文不太一致，但是没影响
MatrixXd TangentBasis(Vector3d &g0) {
  Vector3d b, c;
  Vector3d a = g0.normalized();
  Vector3d tmp(0, 0, 1);
  if (a == tmp)
    tmp << 1, 0, 0;
  b = (tmp - a * (a.transpose() * tmp)).normalized();
  c = a.cross(b);
  MatrixXd bc(3, 2);
  bc.block<3, 1>(0, 0) = b;
  bc.block<3, 1>(0, 1) = c;
  return bc;
}

//see V-B-3 in Paper
//1.按照论文思路，重力向量是由重力大小所约束的，论文中使用半球加上半球切面来参数化重力
//2.然后迭代求得w1,w2
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x) {
  Vector3d g0 = g.normalized() * G.norm();
  Vector3d lx, ly;
  //VectorXd x;
  int all_frame_count = all_image_frame.size();
  int n_state         = all_frame_count * 3 + 2 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  for (int k = 0; k < 4; k++) {
    MatrixXd lxly(3, 2);
    lxly  = TangentBasis(g0);
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) {
      frame_j = next(frame_i);

      MatrixXd tmp_A(6, 9);
      tmp_A.setZero();
      VectorXd tmp_b(6);
      tmp_b.setZero();

      double dt = frame_j->second.pre_integration->sum_dt;

      tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
      tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
      tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
      tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

      tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
      tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
      tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
      tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;

      Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
      //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
      //MatrixXd cov_inv = cov.inverse();
      cov_inv.setIdentity();

      MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
      VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

      A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
      b.segment<6>(i * 3) += r_b.head<6>();

      A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
      b.tail<3>() += r_b.tail<3>();

      A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
      A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    A           = A * 1000.0;
    b           = b * 1000.0;
    x           = A.ldlt().solve(b);
    VectorXd dg = x.segment<2>(n_state - 3);
    g0          = (g0 + lxly * dg).normalized() * G.norm();
    //double s = x(n_state - 1);
  }
  g = g0;
}

//初始化滑动窗口中每帧的 速度V[0:n] Gravity Vectorg,尺度s -> 对应论文的V-B-2
//重力修正RefineGravity -> 对应论文的V-B-3
//重力方向跟世界坐标的Z轴对齐
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x) {
  int all_frame_count = all_image_frame.size();
  int n_state         = all_frame_count * 3 + 3 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  int i = 0;
  for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) {
    frame_j = next(frame_i);

    MatrixXd tmp_A(6, 10);
    tmp_A.setZero();
    VectorXd tmp_b(6);
    tmp_b.setZero();

    double dt = frame_j->second.pre_integration->sum_dt;

    tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
    tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
    tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
    //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
    tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
    tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
    tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
    //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

    Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
    //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
    //MatrixXd cov_inv = cov.inverse();
    cov_inv.setIdentity();

    MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
    VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

    A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
    b.segment<6>(i * 3) += r_b.head<6>();

    A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
    b.tail<4>() += r_b.tail<4>();

    A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
    A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
  }
  A        = A * 1000.0;
  b        = b * 1000.0;
  x        = A.ldlt().solve(b);
  double s = x(n_state - 1) / 100.0;
  ROS_DEBUG("estimated scale: %f", s);
  g = x.segment<3>(n_state - 4);
  ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
  if (fabs(g.norm() - G.norm()) > 1.0 || s < 0) {
    return false;
  }

  RefineGravity(all_image_frame, g, x);  // 在正切空间微调重力向量
  s                = (x.tail<1>())(0) / 100.0;
  (x.tail<1>())(0) = s;
  ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
  if (s < 0.0)
    return false;
  else
    return true;
}

// visual-inertial alignment：视觉SFM的结果与IMU预积分结果对齐
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x) {
  //估测陀螺仪的Bias，对应论文V-B-1
  solveGyroscopeBias(all_image_frame, Bgs);

  //求解V 重力向量g和 尺度s
  if (LinearAlignment(all_image_frame, g, x))
    return true;
  else
    return false;
}
