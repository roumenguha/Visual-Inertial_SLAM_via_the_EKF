This project is composed of the following files
	- p3_utils.py
	- hw3_main.py

p3_utils.py
	- Contains several functions. 
	- The load_data() function is untouched from the default given to us. 
	- The visualize_trajectory_2d() function was modified to be able to plot the most probable landmark locations, and both the EKF vehicle location and landmark locations.
	- hat_map_3() returns the skew-symmetric matrix for a given 3d vector
	- hat_map_6() returns the skew-symmetric matrix for a given 6d vector composed of the linear and angular velocities.
	- projection() performs the projection function of our observation model
	- projection_derivative() performs the necessary operation for our Jacobian
	- world_T_imu essentially returns the inverse of the tracked variable U_t, the inverse pose of the car
	- EKF_inertial_prediction() implements part (a) of the project, by applying the control input to the motion model.
	- EKF_visual_update() implements part (b) of the project, simply updating and tracking landmarks over timesteps.
	- EKF_visual_inertial_prediction() performs exactly the same computations as EKF_inertial_prediction() but on a different set of variables for part (c).
	- EKF_visual_inertial_update() performs part of what EKF_visual_update() does, with the added ability to update our car's location. With this, the EKF is complete.	
	
hw3_main.py
	- contains two functions initCar() and initLandmarks(), and main()
	- initCar() and initLandmarks() aren't very interesting. They simply initialize the dictionaries for the vehicle being localized, and the landmarks being observed. 
	- main() is where the magic happens. The Car and Landmarks dictionaries are initialized, and we begin looping over all the timesteps. At every timestep, the corresponding control inputs were applied, the trajectories were recorded, and the updates via the observation model were applied. Finally, we plot a figure every 100 timesteps.