import numpy as np
from p3_utils import *

def initCar(num_timesteps, weight = 0.01):
    Car = {}
    Car['mean'] = np.eye(4) # in SE(3)
    Car['trajectory'] = np.zeros((4, 4, num_timesteps)) # running history
    Car['covariance'] = weight * np.eye(6)
    
    # visual inertial updated
    Car['mean_vi'] = np.eye(4) # in SE(3)
    Car['trajectory_vi'] = np.zeros((4, 4, num_timesteps)) # running history
    Car['covariance_vi'] = weight * np.eye(6)
    
    return Car

def initLandmarks(num_features, num_timesteps, weight = 0.01):
    Landmarks = {}
    Landmarks['mean'] = np.empty((4, num_features))
    Landmarks['mean'].fill(np.nan)
    Landmarks['trajectory'] = np.empty((4, num_features, num_timesteps))
    Landmarks['covariance'] = np.zeros((3, 3, num_features))

    # visual inertial updated
    Landmarks['mean_vi'] = np.empty((4, num_features))
    Landmarks['mean_vi'].fill(np.nan)
    Landmarks['trajectory_vi'] = np.empty((4, num_features, num_timesteps))
    Landmarks['covariance_vi'] = np.zeros((3, 3, num_features))    

    for i in range(num_features):
        Landmarks['covariance'][:, :, i] = weight * np.eye(3)
        Landmarks['covariance_vi'][:, :, i] = weight * np.eye(3)
    
    return Landmarks

if __name__ == '__main__':
    filename = "./data/0027.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)

    # initialize robots and landmarks
    Car = initCar(t.shape[1])
    Landmarks = initLandmarks(features.shape[1], t.shape[1])
    
    for i in range(t.shape[1]):
        if i == 0:
            continue
        
        tau = np.abs(t[0, i] - t[0, i - 1])

    	# (a) IMU Localization via EKF Prediction
        EKF_inertial_prediction(Car, linear_velocity[:, i], rotational_velocity[:, i], tau, 0.00001, 0.0001)
        EKF_visual_inertial_prediction(Car, linear_velocity[:, i], rotational_velocity[:, i], tau, 0.00001, 0.0001)

        # record current poses (landmark trajectory for debugging)
        Car['trajectory'][:, :, i] = world_T_imu(Car['mean']) # inv(inv pose)
        Landmarks['trajectory'][:, :, i - 1] = Landmarks['mean'][:]

        Car['trajectory_vi'][:, :, i] = world_T_imu(Car['mean_vi']) 
        Landmarks['trajectory_vi'][:, :, i - 1] = Landmarks['mean_vi'][:]

    	# (b) Landmark Mapping via EKF Update
        EKF_visual_update(Car, Landmarks, features[:, :, i], K, b, cam_T_imu, 3500)
         
    	# (c) Visual-Inertial SLAM
        EKF_visual_inertial_update(Car, Landmarks, features[:, :, i], K, b, cam_T_imu, 3500)
        
        # plotting
        if ((i - 1) % 100 == 0 or i == t.shape[1] - 1):
        	# You can use the function below to visualize the robot pose over time
            visualize_trajectory_2d(Car['trajectory'], Landmarks['mean'], Car['trajectory_vi'], Landmarks['mean_vi'], timestamp = str(i), path_name = filename[7:-4], show_ori = True, show_grid = True)
            