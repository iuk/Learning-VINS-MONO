#include "estimator.h"

Estimator::Estimator() : f_manager{Rs} {
  ROS_INFO("init begins");
  clearState();
}

void Estimator::setParameter() {
  for (int i = 0; i < NUM_OF_CAM; i++) {
    tic[i] = TIC[i];
    ric[i] = RIC[i];
  }
  f_manager.setRic(ric);
  ProjectionFactor::sqrt_info   = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  td                            = TD;
}

void Estimator::clearState() {
  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    Rs[i].setIdentity();
    Ps[i].setZero();
    Vs[i].setZero();
    Bas[i].setZero();
    Bgs[i].setZero();
    dt_buf[i].clear();
    linear_acceleration_buf[i].clear();
    angular_velocity_buf[i].clear();

    if (pre_integrations[i] != nullptr)
      delete pre_integrations[i];
    pre_integrations[i] = nullptr;
  }

  for (int i = 0; i < NUM_OF_CAM; i++) {
    tic[i] = Vector3d::Zero();
    ric[i] = Matrix3d::Identity();
  }

  for (auto &it : all_image_frame) {
    if (it.second.pre_integration != nullptr) {
      delete it.second.pre_integration;
      it.second.pre_integration = nullptr;
    }
  }

  solver_flag       = INITIAL;
  first_imu         = false,  // ？？？逗号是shenmegui
      sum_of_back   = 0;
  sum_of_front      = 0;
  frame_count       = 0;
  solver_flag       = INITIAL;
  initial_timestamp = 0;
  all_image_frame.clear();
  td = TD;

  if (tmp_pre_integration != nullptr)
    delete tmp_pre_integration;
  if (last_marginalization_info != nullptr)
    delete last_marginalization_info;

  tmp_pre_integration       = nullptr;
  last_marginalization_info = nullptr;
  last_marginalization_parameter_blocks.clear();

  f_manager.clearState();

  failure_occur       = 0;
  relocalization_info = 0;

  drift_correct_r = Matrix3d::Identity();
  drift_correct_t = Vector3d::Zero();
}

// 1. IMU 预积分，等图像帧间的相对 PVQ
// 2. IMU 积分，得世界系下图像帧绝对 PVQ
void Estimator::processIMU(double dt,
                           const Vector3d &linear_acceleration,
                           const Vector3d &angular_velocity) {
  if (!first_imu)  // first_imu == false：未获取第一帧IMU数据
  {
    first_imu = true;
    // 将第一帧IMU数据记录下来
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
  }

  if (!pre_integrations[frame_count]) {
    pre_integrations[frame_count] =
        new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
  }
  if (frame_count != 0)  // 在初始化时，第一帧图像特征点数据没有对应的预积分
  {
     // =================== 图像帧间的 IMU 预积分，得图像帧间的相对 PVQ ============== 
    // 需要处理三件事情。1 更新当前的预积分量。2 更新IMU残差的协方差矩阵。
    // 3 更新IMU残差对于bias的雅克比矩阵。
    pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
    //if(solver_flag != NON_LINEAR)
    tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

    dt_buf[frame_count].push_back(dt);
    linear_acceleration_buf[frame_count].push_back(linear_acceleration);
    angular_velocity_buf[frame_count].push_back(angular_velocity);
    
    // =========== IMU 中值积分 更新窗口中当前帧的 PQR Bias等===============
    // 用IMU数据进行积分，当积完一个measurement中所有IMU数据后，
    // 就得到了对应图像帧在世界坐标系中的Ps、Vs、Rs
    // 下面这一部分的积分，在没有成功完成初始化时似乎是没有意义的，
    // 因为在没有成功初始化时，对IMU数据来说是没有世界坐标系的
    // 当成功完成了初始化后，下面这一部分积分才有用，
    // 它可以通过IMU积分得到滑动窗口中最新帧在世界坐标系中的
    // Rs[j] 等 在slideWindow()中赋初值
    int j             = frame_count;
    Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
    Vector3d un_gyr   = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
    Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
    Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
    Vector3d un_acc   = 0.5 * (un_acc_0 + un_acc_1);
    Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
    Vs[j] += dt * un_acc;
  }
  // 更新上一次的 acc gyr 测量值
  acc_0 = linear_acceleration;
  gyr_0 = angular_velocity;
}

// map：输入
// feature_id,{camera_id,[x,y,z,u,v,vx,vy]}
void Estimator::processImage(
    const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
    const std_msgs::Header &header) {
  ROS_DEBUG("new image coming ------------------------------------------");
  ROS_DEBUG("Adding feature points %lu", image.size());

  // 把当前帧图像（frame_count）的特征点添加到f_manager.feature容器中
  // 计算第2最新帧与第3最新帧之间的平均视差（当前帧是第1最新帧），然后判断是否把第2最新帧添加为关键帧
  // 在未完成初始化时，如果窗口没有塞满，那么是否添加关键帧的判定结果不起作用，滑动窗口要塞满
  // 只有在滑动窗口塞满后，或者初始化完成之后，才需要滑动窗口，
  // 此时才需要做关键帧判别，根据第2最新关键帧是否未关键帧选择相应的边缘化策略
  if (f_manager.addFeatureCheckParallax(frame_count, image, td))
  {
    // 二新帧 是关键帧
    marginalization_flag = MARGIN_OLD;
  }
  else
  {
    // 二新帧 不是关键帧
    marginalization_flag = MARGIN_SECOND_NEW;
  }

  ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
  ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
  ROS_DEBUG("Solving %d", frame_count);
  ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
  
  Headers[frame_count] = header;


  ImageFrame imageframe(image, header.stamp.toSec());
  imageframe.pre_integration = tmp_pre_integration;
  // 每读取一帧图像特征点数据，都会存入all_image_frame
  all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));  
  
  tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

  // 相机与IMU外参的在线标定
  if (ESTIMATE_EXTRINSIC == 2) 
  {
    ROS_INFO("calibrating extrinsic param, rotation movement is needed");
    if (frame_count != 0) {
      vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
      Matrix3d calib_ric;
      if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric)) {
        ROS_WARN("initial extrinsic rotation calib success");
        ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                       << calib_ric);
        ric[0]             = calib_ric;
        RIC[0]             = calib_ric;
        ESTIMATE_EXTRINSIC = 1;
      }
    }
  }

  // 需要初始化
  if (solver_flag == INITIAL) 
  {
    // 滑动窗口中塞满了才进行初始化
    if (frame_count == WINDOW_SIZE)  
    {
      bool result = false;
      // 在上一次初始化失败后至少0.1秒才进行下一次初始化
      if (ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)  
      {
        // 执行初始化操作
        result            = initialStructure();
        initial_timestamp = header.stamp.toSec();
      }
      if (result)  // 初始化操作成功
      {
        solver_flag = NON_LINEAR;
        solveOdometry();             // 紧耦合优化
        slideWindow();               // 对窗口进行滑动
        f_manager.removeFailures();  // 去除滑出了滑动窗口的特征点
        ROS_INFO("Initialization finish!");
        last_R  = Rs[WINDOW_SIZE];
        last_P  = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];

      } else
        slideWindow();  // 初始化不成功，对窗口进行滑动
    }
    // 滑动窗口没塞满，接着塞
    else
      frame_count++;  
  } 
  // 已经成功初始化，进行正常的VIO紧耦合优化
  else
  {
    TicToc t_solve;
    solveOdometry();  // 紧耦合优化的入口函数
    ROS_DEBUG("solver costs: %fms", t_solve.toc());

    // 失效检测，如果失效则重启VINS系统
    if (failureDetection()) {
      ROS_WARN("failure detection!");
      failure_occur = 1;
      clearState();
      setParameter();
      ROS_WARN("system reboot!");
      return;
    }

    // 对窗口进行滑动
    TicToc t_margin;
    slideWindow();
    f_manager.removeFailures();
    ROS_DEBUG("marginalization costs: %fms", t_margin.toc());

    // prepare output of VINS
    key_poses.clear();
    for (int i = 0; i <= WINDOW_SIZE; i++)
      key_poses.push_back(Ps[i]);

    last_R  = Rs[WINDOW_SIZE];
    last_P  = Ps[WINDOW_SIZE];
    last_R0 = Rs[0];
    last_P0 = Ps[0];
  }
}

/**
 * vins系统初始化
 * 1.确保IMU有足够的excitation
 * 2.检查当前帧（滑动窗口中的最新帧）与滑动窗口中所有图像帧之间的特征点匹配关系，
 *   选择跟当前帧中有足够多数量的特征点（30个）被跟踪，且由足够视差（2o pixels）的某一帧，利用五点法恢复相对旋转和平移量。
 *   如果找不到，则在滑动窗口中保留当前帧，然后等待新的图像帧
 * 3.sfm.construct 全局SFM 恢复滑动窗口中所有帧的位姿，以及特特征点三角化
 * 4.利用pnp恢复其他帧
 * 5.visual-inertial alignment：视觉SFM的结果与IMU预积分结果对齐
 * 6.给滑动窗口中要优化的变量一个合理的初始值以便进行非线性优化
 */
// /*
// 视觉的结构初始化
// 视觉结构初始化过程至关重要，多传感器融合过程中，当单个传感器数据不确定性较高，需要依赖其他传感器降低不确定性。
// 先对纯视觉SFM初始化相机位姿，再和IMU对齐。
// 主要分为 1:纯视觉SFM估计滑动窗内相机位姿和路标点逆深度。
//         2:视觉惯性联合校准，SFM与IMU积分对齐。
// 采用松耦合的传感器融合方法得到初始值。首先用SFM进行纯视觉估计滑动窗内所有帧的位姿以及路标点逆深度，
// 然后与IMU预积分对齐，继而恢复对齐尺度S，重力g，imu速度v，和陀螺仪偏置bg
// */
/**
 * @brief   视觉的结构初始化
 * @Description 确保IMU有充分运动激励
 *              relativePose()找到具有足够视差的两帧,由F矩阵恢复R、t作为初始值
 *              sfm.construct() 全局纯视觉SFM 恢复滑动窗口帧的位姿
 *              visualInitialAlign()视觉惯性联合初始化
 * @return  bool true:初始化成功
*/

bool Estimator::initialStructure() {
  TicToc t_sfm;
  
  // 1.通过重力variance确保IMU有足够的excitation
  // 1. 通过加速度标准差判断IMU是否有充分运动以初始化。
  { //check imu observibility
    map<double, ImageFrame>::iterator frame_it;
    // 1.1 求平均相对加速度 aver_g
    Vector3d sum_g;
    // 遍历所有的 ImageFrame
    for (frame_it = all_image_frame.begin(), frame_it++;
         frame_it != all_image_frame.end(); frame_it++) 
    {
      // 预积分时长
      double dt      = frame_it->second.pre_integration->sum_dt;
      // 相对加速度
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      sum_g += tmp_g;
    }

    // 平均相对加速度
    Vector3d aver_g;
    aver_g     = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
    
    // 1.2 求相对加速度标准差
    double var = 0;
    for (frame_it = all_image_frame.begin(), frame_it++;
         frame_it != all_image_frame.end(); frame_it++) {
      double dt      = frame_it->second.pre_integration->sum_dt;
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
      //cout << "frame g " << tmp_g.transpose() << endl;
    }
    var = sqrt(var / ((int)all_image_frame.size() - 1));
    //ROS_WARN("IMU variation %f!", var);
    if (var < 0.25) {
      ROS_INFO("IMU excitation not enouth!");
      //return false;
    }
  }

  // global sfm
  // 世界携的 位置 姿态
  Quaterniond Q[frame_count + 1];  // 滑动窗口中每一帧的姿态
  Vector3d T[frame_count + 1];     // 滑动窗口中每一帧的位置
  map<int, Vector3d> sfm_tracked_points;

  Matrix3d relative_R;  // 历史匹配帧系下，最新帧的姿态
  Vector3d relative_T;  // 历史匹配帧系下，最新帧的位移
  int l;                // 选定帧在滑动窗口中的帧号
  
  // 2.选择跟最新帧中有足够数量的特征点跟踪和视差的某一帧，利用五点法恢复相对旋转和平移量
  // 如果找不到，则初始化失败
  if (!relativePose(relative_R, relative_T, l)) {
    ROS_INFO("Not enough features or parallax; Move device around");
    return false;
  }

  // 3.初始化滑动窗口中全部初始帧的相机位姿和特征点空间3D位置
  // 用于视觉初始化的图像特征点数据
  {
    vector<SFMFeature> sfm_f;
    {
      // 遍历所有的 Feature ID
      for (auto &it_per_id : f_manager.feature) {
        int imu_j = it_per_id.start_frame - 1;

        // 新建一个 SMF Feature 对象
        SFMFeature tmp_feature;
        tmp_feature.state = false;  // 该特征点的初始状态为：未被三角化
        tmp_feature.id    = it_per_id.feature_id;

        // 遍历该特征点在所有 frame 中的 7 维信息
        for (auto &it_per_frame : it_per_id.feature_per_frame) {
          imu_j++;  // 观测到该特征点的图像帧的帧号
          // 归一化平面 x y 1
          Vector3d pts_j = it_per_frame.point;
          tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }

        sfm_f.push_back(tmp_feature);
      }
    }

    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points)) {
      ROS_DEBUG("global SFM failed!");
      marginalization_flag = MARGIN_OLD;
      return false;
    }
  }

  //solve pnp for all frame
  // 由于并不是第一次视觉初始化就能成功，此时图像帧数目有可能会超过滑动窗口的大小
  // 所以再视觉初始化的最后，要求出滑动窗口外的帧的位姿
  // 4.对于非滑动窗口的所有帧，提供一个初始的R,T，然后solve pnp求解pose
  map<double, ImageFrame>::iterator frame_it;
  map<int, Vector3d>::iterator it;
  // 定义 map<double, ImageFrame> all_image_frame; 
  frame_it = all_image_frame.begin();
  // 遍历所有图像帧
  for (int i = 0; frame_it != all_image_frame.end(); frame_it++) 
  {
    // provide initial guess
    cv::Mat r, rvec, t, D, tmp_r;

    // 对于滑动窗口内的帧：

    // all_image_frame与滑动窗口中对应的帧
    if ((frame_it->first) == Headers[i].stamp.toSec())  
    {
      // 滑动窗口中所有帧都是关键帧
      frame_it->second.is_key_frame = true;
      // 根据各帧相机坐标系的姿态和外参，得到用各帧IMU坐标系的姿态
      //（对应VINS Mono论文(2018年的期刊版论文)中的公式（6））。
      // 本帧姿态变为：l 帧camera系下的 本帧 imu 系位姿
      frame_it->second.R            = Q[i].toRotationMatrix() * RIC[0].transpose();  
      // 姿态改变，位置却维持
      frame_it->second.T            = T[i];
      i++;
      continue;
    }

    // 对于滑动窗口外的帧：

    if ((frame_it->first) > Headers[i].stamp.toSec()) {
      i++;
    }

    // 为滑动窗口外的帧提供一个初始位姿
    Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
    Vector3d P_inital = -R_inital * T[i];
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    frame_it->second.is_key_frame = false;        // 初始化时位于滑动窗口外的帧是非关键帧
    vector<cv::Point3f> pts_3_vector;             // 用于pnp解算的3D点
    vector<cv::Point2f> pts_2_vector;             // 用于pnp解算的2D点
    for (auto &id_pts : frame_it->second.points)  // 对于该帧中的特征点
    {
      int feature_id = id_pts.first;   // 特征点id
      for (auto &i_p : id_pts.second)  // 由于可能有多个相机，所以需要遍历。i_p对应着一个相机所拍图像帧的特征点信息
      {
        it = sfm_tracked_points.find(feature_id);
        // 如果it不是尾部迭代器，说明在sfm_tracked_points中找到了相应的3D点
        if (it != sfm_tracked_points.end()) {
          // 记录该id特征点的3D位置
          Vector3d world_pts = it->second;
          cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
          pts_3_vector.push_back(pts_3);

          // 记录该id的特征点在该帧图像中的2D位置
          Vector2d img_pts = i_p.second.head<2>();
          cv::Point2f pts_2(img_pts(0), img_pts(1));
          pts_2_vector.push_back(pts_2);
        }
      }
    }
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    if (pts_3_vector.size() < 6)  // 如果匹配到的3D点数量少于6个，则认为初始化失败
    {
      cout << "pts_3_vector size " << pts_3_vector.size() << endl;
      ROS_DEBUG("Not enough points for solve pnp !");
      return false;
    }

    // 使用 pnp 解算窗口外帧的位姿
    // 但是没有 三角化 也没有 BA 优化
    if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))  // pnp求解失败
    {
      ROS_DEBUG("solve pnp fail!");
      return false;
    }

    // pnp求解成功
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp, tmp_R_pnp;
    cv::cv2eigen(r, tmp_R_pnp);
    R_pnp = tmp_R_pnp.transpose();
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    T_pnp              = R_pnp * (-T_pnp);
    frame_it->second.R = R_pnp * RIC[0].transpose();  // 根据各帧相机坐标系的姿态和外参，得到用各帧IMU坐标系的姿态。
    frame_it->second.T = T_pnp;
  } // finish  遍历所有图像帧

  //camera与IMU对齐
  if (visualInitialAlign())
    return true;
  else {
    ROS_INFO("misalign visual structure with IMU");
    return false;
  }
}

// visual-inertial alignment：视觉SFM的结果与IMU预积分结果对齐
/**
 * @brief   视觉惯性联合初始化
 * @Description 陀螺仪的偏置校准(加速度偏置没有处理) 计算速度V[0:n] 重力g 尺度s
 *              更新了Bgs后，IMU测量量需要repropagate  
 *              得到尺度s和重力g的方向后，需更新所有图像帧在世界坐标系下的Ps、Rs、Vs
 * @return  bool true：成功
 */
bool Estimator::visualInitialAlign() {
  TicToc t_g;
  VectorXd x;
  //solve scale
  // 调用initial_aligment.cpp中的VisualIMUAlignment函数，完成视觉SFM的结果与IMU预积分结果对齐
  //计算陀螺仪偏置，尺度，重力加速度和速度
  bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
  if (!result) {
    ROS_DEBUG("solve g failed!");
    return false;
  }

  // change state
  for (int i = 0; i <= frame_count; i++) {
    // 滑动窗口中各图像帧在世界坐标系下的旋转和平移
    Matrix3d Ri                                            = all_image_frame[Headers[i].stamp.toSec()].R;
    Vector3d Pi                                            = all_image_frame[Headers[i].stamp.toSec()].T;
    Ps[i]                                                  = Pi;
    Rs[i]                                                  = Ri;
    all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;  // 滑动窗口中所有初始帧都是关键帧
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < dep.size(); i++)
    dep[i] = -1;
  f_manager.clearDepth(dep);

  //triangulat on cam pose , no tic
  Vector3d TIC_TMP[NUM_OF_CAM];
  for (int i = 0; i < NUM_OF_CAM; i++)
    TIC_TMP[i].setZero();
  ric[0] = RIC[0];
  f_manager.setRic(ric);
  f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

  double s = (x.tail<1>())(0);
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
  }
  for (int i = frame_count; i >= 0; i--)
    Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
  int kv = -1;
  map<double, ImageFrame>::iterator frame_i;
  for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++) {
    if (frame_i->second.is_key_frame) {
      kv++;
      Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
    }
  }
  for (auto &it_per_id : f_manager.feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    it_per_id.estimated_depth *= s;
  }

  Matrix3d R0 = Utility::g2R(g);  // 当前参考坐标系与世界坐标系（依靠g构建的坐标系）的旋转矩阵，暂时每搞清楚从谁转到谁？？？
  double yaw  = Utility::R2ypr(R0 * Rs[0]).x();
  R0          = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  g           = R0 * g;
  //Matrix3d rot_diff = R0 * Rs[0].transpose();
  Matrix3d rot_diff = R0;
  for (int i = 0; i <= frame_count; i++) {
    // 似乎是把Ps、Rs、Vs转到世界坐标系下
    // ？？？但是为什么只转化了滑动窗口中的变量啊，all_image_frame中的非滑动窗口变量怎么办啊？？？不转换了吗？？？
    Ps[i] = rot_diff * Ps[i];
    Rs[i] = rot_diff * Rs[i];
    Vs[i] = rot_diff * Vs[i];
  }
  ROS_DEBUG_STREAM("g0     " << g.transpose());
  ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

  return true;
}

// 在滑动窗口中，寻找与最新帧有足够多数量的特征点对应关系和视差的帧，然后用5点法恢复相对位姿
/**
 * @brief   判断两帧有足够视差30且内点数目大于12则可进行初始化，同时得到R和T
 * @Description    判断每帧到窗口最后一帧对应特征点的平均视差是否大于30
                solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
 * @param[out]   relative_R 当前帧到第l帧之间的旋转矩阵R
 * @param[out]   relative_T 当前帧到第l帧之间的平移向量T
 * @param[out]   L 保存滑动窗口中与当前帧满足初始化条件的那一帧
 * @return  bool 1:可以进行初始化;0:不满足初始化条件
*/
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l) {
  // 遍历窗口所有帧
  for (int i = 0; i < WINDOW_SIZE; i++) 
  {
    //寻找第i帧到窗口最后一帧的对应特征点
    // 获取 第 i 和 最新帧之间的 3D 点对
    vector<pair<Vector3d, Vector3d>> corres;
    corres = f_manager.getCorresponding(i, WINDOW_SIZE);
    // 如果点对数 >20
    if (corres.size() > 20) {
      //计算平均视差
      double sum_parallax = 0;
      double average_parallax;
      // 遍历所有点对
      for (int j = 0; j < int(corres.size()); j++) {
        //第j个对应点在第i帧和最后一帧的(x,y)
        Vector2d pts_0(corres[j].first(0), corres[j].first(1));
        Vector2d pts_1(corres[j].second(0), corres[j].second(1));
        double parallax = (pts_0 - pts_1).norm();
        // 计算时差和
        sum_parallax    = sum_parallax + parallax;
      }
      // 计算平均时差
      average_parallax = 1.0 * sum_parallax / int(corres.size());

      //判断是否满足初始化条件：视差>30和内点数满足要求
      //同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的Rt
      // 如果时差够大，且求解相对 RT 成功
      if (average_parallax * 460 > 30 &&
          m_estimator.solveRelativeRT(corres, relative_R, relative_T)) {
        l = i;
        ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
        return true;
      }
    }
    // 如果点对数 <20，啥也不干
  }
  return false;
}

void Estimator::solveOdometry() {
  if (frame_count < WINDOW_SIZE)
    return;
  if (solver_flag == NON_LINEAR) {
    // 三角化
    TicToc t_tri;
    f_manager.triangulate(Ps, tic, ric);
    ROS_DEBUG("triangulation costs %f", t_tri.toc());

    // 滑动窗口紧耦合优化
    optimization();
  }
}

void Estimator::vector2double() {
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    para_Pose[i][0] = Ps[i].x();
    para_Pose[i][1] = Ps[i].y();
    para_Pose[i][2] = Ps[i].z();
    Quaterniond q{Rs[i]};
    para_Pose[i][3] = q.x();
    para_Pose[i][4] = q.y();
    para_Pose[i][5] = q.z();
    para_Pose[i][6] = q.w();

    para_SpeedBias[i][0] = Vs[i].x();
    para_SpeedBias[i][1] = Vs[i].y();
    para_SpeedBias[i][2] = Vs[i].z();

    para_SpeedBias[i][3] = Bas[i].x();
    para_SpeedBias[i][4] = Bas[i].y();
    para_SpeedBias[i][5] = Bas[i].z();

    para_SpeedBias[i][6] = Bgs[i].x();
    para_SpeedBias[i][7] = Bgs[i].y();
    para_SpeedBias[i][8] = Bgs[i].z();
  }
  for (int i = 0; i < NUM_OF_CAM; i++) {
    para_Ex_Pose[i][0] = tic[i].x();
    para_Ex_Pose[i][1] = tic[i].y();
    para_Ex_Pose[i][2] = tic[i].z();
    Quaterniond q{ric[i]};
    para_Ex_Pose[i][3] = q.x();
    para_Ex_Pose[i][4] = q.y();
    para_Ex_Pose[i][5] = q.z();
    para_Ex_Pose[i][6] = q.w();
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < f_manager.getFeatureCount(); i++)
    para_Feature[i][0] = dep(i);
  if (ESTIMATE_TD)
    para_Td[0][0] = td;
}

void Estimator::double2vector() {
  Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
  Vector3d origin_P0 = Ps[0];

  if (failure_occur) {
    origin_R0     = Utility::R2ypr(last_R0);
    origin_P0     = last_P0;
    failure_occur = 0;
  }
  Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                   para_Pose[0][3],
                                                   para_Pose[0][4],
                                                   para_Pose[0][5])
                                           .toRotationMatrix());
  double y_diff       = origin_R0.x() - origin_R00.x();
  //TODO
  Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
  if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
    ROS_DEBUG("euler singular point!");
    rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                   para_Pose[0][3],
                                   para_Pose[0][4],
                                   para_Pose[0][5])
                           .toRotationMatrix()
                           .transpose();
  }

  for (int i = 0; i <= WINDOW_SIZE; i++) {
    Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

    Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) +
            origin_P0;

    Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                para_SpeedBias[i][1],
                                para_SpeedBias[i][2]);

    Bas[i] = Vector3d(para_SpeedBias[i][3],
                      para_SpeedBias[i][4],
                      para_SpeedBias[i][5]);

    Bgs[i] = Vector3d(para_SpeedBias[i][6],
                      para_SpeedBias[i][7],
                      para_SpeedBias[i][8]);
  }

  for (int i = 0; i < NUM_OF_CAM; i++) {
    tic[i] = Vector3d(para_Ex_Pose[i][0],
                      para_Ex_Pose[i][1],
                      para_Ex_Pose[i][2]);
    ric[i] = Quaterniond(para_Ex_Pose[i][6],
                         para_Ex_Pose[i][3],
                         para_Ex_Pose[i][4],
                         para_Ex_Pose[i][5])
                 .toRotationMatrix();
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < f_manager.getFeatureCount(); i++)
    dep(i) = para_Feature[i][0];
  f_manager.setDepth(dep);
  if (ESTIMATE_TD)
    td = para_Td[0][0];

  // relative info between two loop frame
  if (relocalization_info) {
    Matrix3d relo_r;
    Vector3d relo_t;
    relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
    relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                 relo_Pose[1] - para_Pose[0][1],
                                 relo_Pose[2] - para_Pose[0][2]) +
             origin_P0;
    double drift_correct_yaw;
    drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
    drift_correct_r   = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
    drift_correct_t   = prev_relo_t - drift_correct_r * relo_t;
    relo_relative_t   = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
    relo_relative_q   = relo_r.transpose() * Rs[relo_frame_local_index];
    relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
    //cout << "vins relo " << endl;
    //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
    //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
    relocalization_info = 0;
  }
}

bool Estimator::failureDetection() {
  if (f_manager.last_track_num < 2) {
    ROS_INFO(" little feature %d", f_manager.last_track_num);
    //return true;
  }
  if (Bas[WINDOW_SIZE].norm() > 2.5) {
    ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
    return true;
  }
  if (Bgs[WINDOW_SIZE].norm() > 1.0) {
    ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
    return true;
  }
  /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
  Vector3d tmp_P = Ps[WINDOW_SIZE];
  if ((tmp_P - last_P).norm() > 5) {
    ROS_INFO(" big translation");
    return true;
  }
  if (abs(tmp_P.z() - last_P.z()) > 1) {
    ROS_INFO(" big z translation");
    return true;
  }
  Matrix3d tmp_R   = Rs[WINDOW_SIZE];
  Matrix3d delta_R = tmp_R.transpose() * last_R;
  Quaterniond delta_Q(delta_R);
  double delta_angle;
  delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
  if (delta_angle > 50) {
    ROS_INFO(" big delta_angle ");
    //return true;
  }
  return false;
}

void Estimator::optimization() {
  ceres::Problem problem;
  ceres::LossFunction *loss_function;
  //loss_function = new ceres::HuberLoss(1.0);
  loss_function = new ceres::CauchyLoss(1.0);
  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
    problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
  }
  for (int i = 0; i < NUM_OF_CAM; i++) {
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
    if (!ESTIMATE_EXTRINSIC) {
      ROS_DEBUG("fix extinsic param");
      problem.SetParameterBlockConstant(para_Ex_Pose[i]);
    } else
      ROS_DEBUG("estimate extinsic param");
  }
  if (ESTIMATE_TD) {
    problem.AddParameterBlock(para_Td[0], 1);
    //problem.SetParameterBlockConstant(para_Td[0]);
  }

  TicToc t_whole, t_prepare;
  vector2double();

  if (last_marginalization_info) {
    // construct new marginlization_factor
    MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
    problem.AddResidualBlock(marginalization_factor, NULL,
                             last_marginalization_parameter_blocks);
  }

  for (int i = 0; i < WINDOW_SIZE; i++) {
    int j = i + 1;
    if (pre_integrations[j]->sum_dt > 10.0)
      continue;
    IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
    problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
  }
  int f_m_cnt       = 0;
  int feature_index = -1;
  for (auto &it_per_id : f_manager.feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;

    ++feature_index;

    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

    Vector3d pts_i = it_per_id.feature_per_frame[0].point;

    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      if (imu_i == imu_j) {
        continue;
      }
      Vector3d pts_j = it_per_frame.point;
      if (ESTIMATE_TD) {
        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
        problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
        /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
      } else {
        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
        problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
      }
      f_m_cnt++;
    }
  }

  ROS_DEBUG("visual measurement count: %d", f_m_cnt);
  ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

  if (relocalization_info) {
    //printf("set relocalization factor! \n");
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
    int retrive_feature_index = 0;
    int feature_index         = -1;
    for (auto &it_per_id : f_manager.feature) {
      it_per_id.used_num = it_per_id.feature_per_frame.size();
      if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
        continue;
      ++feature_index;
      int start = it_per_id.start_frame;
      if (start <= relo_frame_local_index) {
        while ((int)match_points[retrive_feature_index].z() < it_per_id.feature_id) {
          retrive_feature_index++;
        }
        if ((int)match_points[retrive_feature_index].z() == it_per_id.feature_id) {
          Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
          Vector3d pts_i = it_per_id.feature_per_frame[0].point;

          ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
          problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
          retrive_feature_index++;
        }
      }
    }
  }

  ceres::Solver::Options options;

  options.linear_solver_type = ceres::DENSE_SCHUR;
  //options.num_threads = 2;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations         = NUM_ITERATIONS;
  //options.use_explicit_schur_complement = true;
  //options.minimizer_progress_to_stdout = true;
  //options.use_nonmonotonic_steps = true;
  if (marginalization_flag == MARGIN_OLD)
    options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
  else
    options.max_solver_time_in_seconds = SOLVER_TIME;
  TicToc t_solver;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //cout << summary.BriefReport() << endl;
  ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
  ROS_DEBUG("solver costs: %f", t_solver.toc());

  double2vector();

  TicToc t_whole_marginalization;
  if (marginalization_flag == MARGIN_OLD) {
    MarginalizationInfo *marginalization_info = new MarginalizationInfo();
    vector2double();

    if (last_marginalization_info) {
      vector<int> drop_set;
      for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
        if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
            last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
          drop_set.push_back(i);
      }
      // construct new marginlization_factor
      MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
      ResidualBlockInfo *residual_block_info        = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                     last_marginalization_parameter_blocks,
                                                                     drop_set);

      marginalization_info->addResidualBlockInfo(residual_block_info);
    }

    {
      if (pre_integrations[1]->sum_dt < 10.0) {
        IMUFactor *imu_factor                  = new IMUFactor(pre_integrations[1]);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                       vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                       vector<int>{0, 1});
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }

    {
      int feature_index = -1;
      for (auto &it_per_id : f_manager.feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
          continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        if (imu_i != 0)
          continue;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame) {
          imu_j++;
          if (imu_i == imu_j)
            continue;

          Vector3d pts_j = it_per_frame.point;
          if (ESTIMATE_TD) {
            ProjectionTdFactor *f_td               = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                              it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                              it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                           vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual_block_info);
          } else {
            ProjectionFactor *f                    = new ProjectionFactor(pts_i, pts_j);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                           vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual_block_info);
          }
        }
      }
    }

    TicToc t_pre_margin;
    marginalization_info->preMarginalize();
    ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

    TicToc t_margin;
    marginalization_info->marginalize();
    ROS_DEBUG("marginalization %f ms", t_margin.toc());

    std::unordered_map<long, double *> addr_shift;
    for (int i = 1; i <= WINDOW_SIZE; i++) {
      addr_shift[reinterpret_cast<long>(para_Pose[i])]      = para_Pose[i - 1];
      addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
      addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
    if (ESTIMATE_TD) {
      addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
    }
    vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

    if (last_marginalization_info)
      delete last_marginalization_info;
    last_marginalization_info             = marginalization_info;
    last_marginalization_parameter_blocks = parameter_blocks;

  } else {
    if (last_marginalization_info &&
        std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1])) {
      MarginalizationInfo *marginalization_info = new MarginalizationInfo();
      vector2double();
      if (last_marginalization_info) {
        vector<int> drop_set;
        for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
          ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
          if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
            drop_set.push_back(i);
        }
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        ResidualBlockInfo *residual_block_info        = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                       last_marginalization_parameter_blocks,
                                                                       drop_set);

        marginalization_info->addResidualBlockInfo(residual_block_info);
      }

      TicToc t_pre_margin;
      ROS_DEBUG("begin marginalization");
      marginalization_info->preMarginalize();
      ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

      TicToc t_margin;
      ROS_DEBUG("begin marginalization");
      marginalization_info->marginalize();
      ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

      std::unordered_map<long, double *> addr_shift;
      for (int i = 0; i <= WINDOW_SIZE; i++) {
        if (i == WINDOW_SIZE - 1)
          continue;
        else if (i == WINDOW_SIZE) {
          addr_shift[reinterpret_cast<long>(para_Pose[i])]      = para_Pose[i - 1];
          addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        } else {
          addr_shift[reinterpret_cast<long>(para_Pose[i])]      = para_Pose[i];
          addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
        }
      }
      for (int i = 0; i < NUM_OF_CAM; i++)
        addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
      if (ESTIMATE_TD) {
        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
      }

      vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
      if (last_marginalization_info)
        delete last_marginalization_info;
      last_marginalization_info             = marginalization_info;
      last_marginalization_parameter_blocks = parameter_blocks;
    }
  }
  ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());

  ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

void Estimator::slideWindow() {
  TicToc t_margin;
  // 如果要边缘化 最老的
  if (marginalization_flag == MARGIN_OLD) 
  {
    double t_0 = Headers[0].stamp.toSec();
    back_R0    = Rs[0];
    back_P0    = Ps[0];
    // 如果窗口满了
    if (frame_count == WINDOW_SIZE) 
    {
      // 遍历窗口
      // 经历完这个 for 循环后，原来 [0] 位置的元素交换到 [SIZE] 位置
      // 并且其他元素的整体左移
      for (int i = 0; i < WINDOW_SIZE; i++) 
      {
        Rs[i].swap(Rs[i + 1]);

        std::swap(pre_integrations[i], pre_integrations[i + 1]);

        dt_buf[i].swap(dt_buf[i + 1]);
        linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
        angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

        Headers[i] = Headers[i + 1];
        Ps[i].swap(Ps[i + 1]);
        Vs[i].swap(Vs[i + 1]);
        Bas[i].swap(Bas[i + 1]);
        Bgs[i].swap(Bgs[i + 1]);
      }

      // 修改 [SIZE] 位置的数据为最新(此时的[SIZE-1])的数据
      // 此步是为新一帧图像准备位姿初值
      // 滑动窗口中最新帧的时间戳、位姿和bias，用滑动窗口中第2最新帧的数据来初始化
      Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];  // ？？？前一帧的时间戳好像没什么用啊，因为在真正接收新的图像帧是，会重新对这个量进行赋值
      Ps[WINDOW_SIZE]      = Ps[WINDOW_SIZE - 1];
      Vs[WINDOW_SIZE]      = Vs[WINDOW_SIZE - 1];
      Rs[WINDOW_SIZE]      = Rs[WINDOW_SIZE - 1];
      Bas[WINDOW_SIZE]     = Bas[WINDOW_SIZE - 1];
      Bgs[WINDOW_SIZE]     = Bgs[WINDOW_SIZE - 1];

      // 删除[SIZE] 位置的预积分对象，并再新建一个
      delete pre_integrations[WINDOW_SIZE];
      pre_integrations[WINDOW_SIZE] =
          new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

      dt_buf[WINDOW_SIZE].clear();
      linear_acceleration_buf[WINDOW_SIZE].clear();
      angular_velocity_buf[WINDOW_SIZE].clear();

      // 一个必然进入的 if
      if (true || solver_flag == INITIAL) {
        map<double, ImageFrame>::iterator it_0;
        it_0 = all_image_frame.find(t_0);
        delete it_0->second.pre_integration;
        it_0->second.pre_integration = nullptr;

        for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it) {
          if (it->second.pre_integration)
            delete it->second.pre_integration;
          it->second.pre_integration = NULL;
        }

        all_image_frame.erase(all_image_frame.begin(), it_0);
        all_image_frame.erase(t_0);
      }
      slideWindowOld();
    }
    // 如果窗口没满，啥也不干
  } 
  // 如果要边缘化第二新的
  else {
    // 如果窗口满了
    if (frame_count == WINDOW_SIZE) 
    {
      for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++) {
        double tmp_dt                    = dt_buf[frame_count][i];
        Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
        Vector3d tmp_angular_velocity    = angular_velocity_buf[frame_count][i];

        pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

        dt_buf[frame_count - 1].push_back(tmp_dt);
        linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
        angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
      }

      Headers[frame_count - 1] = Headers[frame_count];
      Ps[frame_count - 1]      = Ps[frame_count];
      Vs[frame_count - 1]      = Vs[frame_count];
      Rs[frame_count - 1]      = Rs[frame_count];
      Bas[frame_count - 1]     = Bas[frame_count];
      Bgs[frame_count - 1]     = Bgs[frame_count];

      delete pre_integrations[WINDOW_SIZE];
      pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

      dt_buf[WINDOW_SIZE].clear();
      linear_acceleration_buf[WINDOW_SIZE].clear();
      angular_velocity_buf[WINDOW_SIZE].clear();

      slideWindowNew();
    }
    // 如果窗口没满，啥也不干
  }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew() {
  sum_of_front++;
  f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld() {
  sum_of_back++;

  bool shift_depth = solver_flag == NON_LINEAR ? true : false;
  if (shift_depth) {
    Matrix3d R0, R1;
    Vector3d P0, P1;
    R0 = back_R0 * ric[0];
    R1 = Rs[0] * ric[0];
    P0 = back_P0 + back_R0 * tic[0];
    P1 = Ps[0] + Rs[0] * tic[0];
    f_manager.removeBackShiftDepth(R0, P0, R1, P1);
  } else
    f_manager.removeBack();
}
/**
 * @brief   进行重定位
 * @optional    
 * @param[in]   _frame_stamp    重定位帧时间戳
 * @param[in]   _frame_index    重定位帧索引值
 * @param[in]   _match_points   重定位帧的所有匹配点
 * @param[in]   _relo_t     重定位帧平移向量
 * @param[in]   _relo_r     重定位帧旋转矩阵
 * @return      void
*/
void Estimator::setReloFrame(double _frame_stamp,
                             int _frame_index,
                             vector<Vector3d> &_match_points, 
                             Vector3d _relo_t, 
                             Matrix3d _relo_r) {
  relo_frame_stamp = _frame_stamp;
  relo_frame_index = _frame_index;
  match_points.clear();
  match_points = _match_points;
  prev_relo_t  = _relo_t;
  prev_relo_r  = _relo_r;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    if (relo_frame_stamp == Headers[i].stamp.toSec()) {
      relo_frame_local_index = i;
      relocalization_info    = 1;
      for (int j = 0; j < SIZE_POSE; j++)
        relo_Pose[j] = para_Pose[i][j];
    }
  }
}
