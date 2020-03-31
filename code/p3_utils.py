import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from scipy.linalg import expm

def load_data(file_name):
  '''
  function to read visual features, IMU measurements and calibration parameters
  Input:
      file_name: the input data file. Should look like "XXX_sync_KLT.npz"
  Output:
      t: time stamp
          with shape 1 * N_t
      features: visual feature point coordinates in stereo images, 
          with shape 4 * M * N_t, where M is number of features
      linear_velocity: IMU measurements in IMU frame
          with shape 3 * N_t
      rotational_velocity: IMU measurements in IMU frame
          with shape 3 * N_t
      K: (left)camera intrinsic matrix
          [fx  0 cx
            0 fy cy
            0  0  1]
          with shape 3*3
      b: stereo camera baseline
          with shape 1
      cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
          close to 
          [ 0 -1  0 t1
            0  0 -1 t2
            1  0  0 t3
            0  0  0  1]
          with shape 4*4
  '''
  with np.load(file_name) as data:
      t = data["time_stamps"] # time_stamps
      features = data["features"] # 4 x num_features : pixel coordinates of features
      linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
      rotational_velocity = data["rotational_velocity"] # rotational velocity measured in the body frame
      K = data["K"] # intrindic calibration matrix
      b = data["b"] # baseline
      cam_T_imu = data["cam_T_imu"] # Transformation from imu to camera frame
  return t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu

def visualize_trajectory_2d(pose, landmarks, better_pose, better_landmarks, timestamp, path_name="Unknown", show_ori=False, show_grid=False):
  '''
  function to visualize the trajectory in 2D
  Input:
      pose:   4*4*N matrix representing the camera pose, 
              where N is the number of pose, and each
              4*4 matrix is in SE(3)
  '''
  fig,ax = plt.subplots(figsize=(5,5))
  n_pose = pose.shape[2]
  ax.plot(landmarks[0,:],landmarks[1,:],'g.',markersize=1.5,label='landmarks')
  ax.plot(better_landmarks[0,:],better_landmarks[1,:],'c.',markersize=1.5,label='landmarks_VI')
  ax.plot(pose[0,3,:],pose[1,3,:],'r-',markersize=6,label=path_name)
  ax.plot(better_pose[0,3,:],better_pose[1,3,:],'b-',markersize=6,label=path_name + "_VI")
  ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
  ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  if show_ori:
      select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
      yaw_list = []
      for i in select_ori_index:
          _,_,yaw = mat2euler(pose[:3,:3,i])
          yaw_list.append(yaw)
      dx = np.cos(yaw_list)
      dy = np.sin(yaw_list)
      dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
      ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
          color="b",units="xy",width=1)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_title('timestamp ' + timestamp)
  ax.axis('equal')
  ax.grid(show_grid)
  ax.legend()
  fig.savefig("d" + path_name + "t" + timestamp, dpi = 300)
  plt.show(block=True)
  return fig, ax

# form the skew-symmetric matrix from a given vector x
def hat_map_3(x):
    hat_map = np.array([[   0,  -x[2],  x[1]],
                        [x[2],      0, -x[0]], 
                        [-x[1],  x[0],    0]])
    return hat_map

def hat_map_6(u):
    theta = u[3:, np.newaxis]
    p = u[:3, np.newaxis]
    
    hat_map = np.block([[hat_map_3(theta), -p],
                         [np.zeros((1, 4))]])
    
    return hat_map

def projection(q):
    return q / q[2]

def projection_derivative(q):
    derivative = np.array([[1, 0, -q[0]/q[2], 0],
                           [0, 1, -q[1]/q[2], 0],
                           [0, 0,          0, 0],
                           [0, 0, -q[3]/q[2], 1]])
    return derivative / q[2]

# K is the calibration matrix, b is the baseline
def stereo_camera_model(K, b):
    M = np.array([[K[0, 0],         0, K[0, 2],            0],
                  [      0,   K[1, 1], K[1, 2],            0],
                  [K[0, 0],         0, K[0, 2], -K[0, 0] * b],
                  [      0,   K[1, 1], K[1, 2],            0]])
    return M
    
# Converts car's current inverse pose (U) to world-frame
def world_T_imu(mean_pose):
    R_T = np.transpose(mean_pose[:3, :3])
    p = mean_pose[:3, 3].reshape(3, 1)
    
    U_inv = np.vstack((np.hstack((R_T, -np.dot(R_T, p))), np.array([0, 0, 0, 1])))
    
    return U_inv

def EKF_inertial_prediction(Car, v, omega, tau, weight_v, weight_omega):
    # covariance for movement noise
    W = np.block([[weight_v * np.eye(3),          np.zeros((3,3))],
                  [    np.zeros((3, 3)), weight_omega * np.eye(3)]])
    
    tau = -(tau)
    
    u_hat = np.vstack((np.hstack((hat_map_3(omega), v.reshape(3, 1))), np.zeros((1, 4))))    
    u_curlyhat = np.block([[  hat_map_3(omega),     hat_map_3(v)], 
                           [  np.zeros((3, 3)), hat_map_3(omega)]])
    
    Car['mean'] = expm(tau * u_hat) @ Car['mean']
    Car['covariance'] = expm(tau * u_curlyhat) @ Car['covariance'] @ np.transpose(expm(tau * u_curlyhat)) + W
    
def EKF_visual_update(Car, Landmarks, curr_features, K, b, cam_T_imu, weight = 1000):
    # covariance for measurement noise
    V = weight * np.eye(4)
    
    P = np.eye(3, 4)
    M = stereo_camera_model(K, b)
    
    for i in range(curr_features.shape[1]):
        
        z = curr_features[:, i][:]
        
        # only operate for landmarks present in current timestep
        if (np.all(z == -1)):
            continue
        
        # else if we make it here, that means the current landmark is present in the camera frame.
        # if, in the previous timestep, the landmark wasn't present, initialize the landmark now
        # using the car's pose
        if (np.all(np.isnan(Landmarks['mean'][:, i]))):
            d = (z[0] - z[2])
            Z_0 = (K[0, 0] * b) / d
            
            world_T_cam = world_T_imu(Car['mean']) @ np.linalg.inv(cam_T_imu)
            
            camera_frame_coords = np.hstack((Z_0 * np.linalg.inv(K) @ np.hstack((z[:2], 1)), 1))
            
            Landmarks['mean'][:, i] = world_T_cam @ camera_frame_coords
            
            continue 
        
        # else if landmark is present in the current timestamp, and has been seen before
        # create predicted z_tilde from previous z (in camera-frame coordinates)
        cam_T_world = cam_T_imu @ Car['mean']
        curr_landmark = cam_T_world @ Landmarks['mean'][:, i]
        z_tilde = M @ projection(curr_landmark) # remove depth information via projection, and project to pixels
        
        # form H; the Jacobian of z_tilde w.r.t. current feature m evaluated at car's current position
        H = M @ projection_derivative(curr_landmark) @ cam_T_world @ P.T
        
        # perform the EKF update
        KG = Landmarks['covariance'][:, :, i] @ H.T @ np.linalg.inv(H @ Landmarks['covariance'][:, :, i] @ H.T + V)
        
        Landmarks['mean'][:, i] = Landmarks['mean'][:, i] + P.T @ KG @ (z - z_tilde)
        Landmarks['covariance'][:, :, i] = (np.eye(3) - KG @ H) @ Landmarks['covariance'][:, :, i]
        
def EKF_visual_inertial_prediction(Car, v, omega, tau, weight_v, weight_omega):
    # movement noise
#    noise = np.block([[np.random.normal(0, weight_omega, (3, 3)), np.random.normal(0, weight_v, (3, 1))],
#                      [np.zeros((1, 4))]])
    
    # covariance for movement noise
    W = np.block([[weight_v * np.eye(3), np.zeros((3,3))],
                  [    np.zeros((3, 3)), weight_omega * np.eye(3)]])
    
    tau = -(tau)
    
    u_hat = np.vstack((np.hstack((hat_map_3(omega), v.reshape(3, 1))), np.zeros((1, 4))))    
    u_curlyhat = np.block([[hat_map_3(omega),     hat_map_3(v)], 
                           [np.zeros((3, 3)), hat_map_3(omega)]])
    
    Car['mean_vi'] = expm(tau * u_hat) @ Car['mean_vi'] # + noise
    Car['covariance_vi'] = expm(tau * u_curlyhat) @ Car['covariance_vi'] @ np.transpose(expm(tau * u_curlyhat)) + W

def EKF_visual_inertial_update(Car, Landmarks, curr_features, K, b, cam_T_imu, weight):
    # covariance for measurement noise
    V = weight * np.eye(4)
    P = np.eye(3, 4)
    M = stereo_camera_model(K, b)
    
    for i in range(curr_features.shape[1]):
        
        z = curr_features[:, i][:]
        
        # only operate for landmarks present in current timestep
        if (np.all(z == -1)):
            continue
        
        # else if we make it here, that means the current landmark is present in the camera frame.
        # if, in the previous timestep, the landmark wasn't present, initialize the landmark now
        # using the car's pose
        if (np.all(np.isnan(Landmarks['mean_vi'][:, i]))):
            d = (z[0] - z[2])
            Z_0 = (K[0, 0] * b) / d
            
            world_T_cam = world_T_imu(Car['mean_vi']) @ np.linalg.inv(cam_T_imu)
            
            camera_frame_coords = np.hstack((Z_0 * np.linalg.inv(K) @ np.hstack((z[:2], 1)), 1))
            
            Landmarks['mean_vi'][:, i] = world_T_cam @ camera_frame_coords
    
            continue 
        
        # else if landmark is present in the current timestamp, and has been seen before
        # create predicted z_tilde from previous z (in camera-frame coordinates)
        cam_T_world = cam_T_imu @ Car['mean_vi']
        curr_landmark = cam_T_world @ Landmarks['mean_vi'][:, i]
        z_tilde = M @ projection(curr_landmark) # remove depth information via projection, and project to pixels
        
        # form H; the Jacobian of z_tilde w.r.t. current feature m evaluated at car's current position
        H = M @ projection_derivative(curr_landmark) @ cam_T_world @ P.T
        
        # perform the visual EKF update
        KG = Landmarks['covariance_vi'][:, :, i] @ H.T @ np.linalg.inv(H @ Landmarks['covariance_vi'][:, :, i] @ H.T + V)
        
        Landmarks['mean_vi'][:, i] = Landmarks['mean_vi'][:, i] + P.T @ KG @ (z - z_tilde)
        Landmarks['covariance_vi'][:, :, i] = (np.eye(3) - KG @ H) @ Landmarks['covariance_vi'][:, :, i]
        
        curr_landmark = Car['mean_vi'] @ Landmarks['mean_vi'][:, i]
        # form H; the Jacobian of z_tilde w.r.t. current car's inverse pose evaluated at car's current position
        H = M @ projection_derivative(cam_T_imu @ curr_landmark) @ cam_T_imu @ np.block([[np.eye(3), -hat_map_3(curr_landmark[:3])],
                                                                                         [np.zeros((1, 6))]])
        # perform the inertial EKF update
        KG = Car['covariance_vi'] @ H.T @ np.linalg.inv(H @ Car['covariance_vi'] @ H.T + V)
        
        Car['mean_vi'] = expm(hat_map_6(KG @ (z - z_tilde))) @ Car['mean_vi']
        Car['covariance_vi'] = (np.eye(6) - KG @ H) @ Car['covariance_vi']
