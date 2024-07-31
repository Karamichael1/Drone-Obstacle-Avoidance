import airsim
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from fastslam import Particle, fast_slam1, calc_input, calc_final_state

# AirSim setup
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# FastSLAM parameters
N_PARTICLE = 100
n_landmark = 0  # We'll discover landmarks as we go
DT = 0.1
particles = [Particle(n_landmark) for _ in range(N_PARTICLE)]

# Initialize plot
plt.ion()
fig, ax = plt.subplots()

def get_drone_state():
    state = client.getMultirotorState()
    return np.array([
        state.kinematics_estimated.position.x_val,
        state.kinematics_estimated.position.y_val,
        state.kinematics_estimated.orientation.z_val
    ]).reshape(3, 1)

def get_lidar_data():
    lidar_data = client.getLidarData()
    if len(lidar_data.point_cloud) < 3:
        return None
    points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
    return np.reshape(points, (int(points.shape[0]/3), 3))

def lidar_to_observations(lidar_points, x_true):
    z = np.zeros((3, len(lidar_points)))
    for i, point in enumerate(lidar_points):
        dx = point[0] - x_true[0, 0]
        dy = point[1] - x_true[1, 0]
        d = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) - x_true[2, 0]
        z[:, i] = [d, angle, i]
    return z

try:
    client.takeoffAsync().join()
    
    while True:
        # Get drone state and LiDAR data
        x_true = get_drone_state()
        lidar_points = get_lidar_data()
        
        if lidar_points is not None:
            # Convert LiDAR data to FastSLAM observations
            z = lidar_to_observations(lidar_points, x_true)
            
            # Get drone movement
            u = calc_input(time.time())  # Replace with actual drone movement data
            
            # Run FastSLAM update
            try:
                particles = fast_slam1(particles, u, z)
            except Exception as e:
                print(f"Error during FastSLAM update: {e}")
                continue
            
            x_est = calc_final_state(particles)
            
            # Update plot
            ax.clear()
            ax.scatter(lidar_points[:, 0], lidar_points[:, 1], c='b', s=1)
            for p in particles:
                ax.plot(p.x, p.y, '.r')
            ax.plot(x_est[0], x_est[1], 'xk')
            plt.pause(0.001)
        
        # Move drone
        client.moveByVelocityAsync(1, 0, 0, 0.1).join()
        
        time.sleep(DT)

except KeyboardInterrupt:
    print("Program interrupted. Landing...")
    client.landAsync().join()

finally:
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Flight completed.")
