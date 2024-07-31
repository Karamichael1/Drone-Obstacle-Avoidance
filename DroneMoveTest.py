import airsim
import numpy as np
import time
from slam import fast_slam2, Particle, calc_final_state, N_PARTICLE
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# List of LiDAR sensor names
lidar_names = [
    "LidarFrontRight", "LidarFrontLeft", "LidarBackLeft", "LidarBackRight",
    "LidarTop", "LidarBottom"
]

# SLAM initialization
n_landmark = 100  # Assuming a maximum of 100 landmarks
particles = [Particle(n_landmark) for _ in range(N_PARTICLE)]
x_est = np.zeros((3, 1))  # State estimate [x, y, yaw]
def init_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('SLAM Map')
    return fig, ax

def update_plot(ax, particles, x_est, landmarks):
    ax.clear()
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('SLAM Map')

    # Plot particles
    for p in particles:
        ax.plot(p.x, p.y, '.r', markersize=1)

    # Plot landmarks
    if landmarks is not None:
        landmark_x = landmarks[:, 0] * np.cos(landmarks[:, 1])
        landmark_y = landmarks[:, 0] * np.sin(landmarks[:, 1])
        ax.scatter(landmark_x, landmark_y, c='b', marker='*', s=50, label='Landmarks')

    # Plot estimated position
    ax.plot(x_est[0], x_est[1], 'go', markersize=10, label='Estimated Position')

    ax.legend()
    ax.grid(True)
    plt.draw()
    plt.pause(0.001)
def collect_lidar_data():
    points_list = []
    for lidar_name in lidar_names:
        lidar_data = client.getLidarData(lidar_name=lidar_name)
        if len(lidar_data.point_cloud) >= 3:
            points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            points_list.append(points)
    if points_list:
        return np.concatenate(points_list, axis=0)
    return None

def extract_landmarks(points, max_landmarks=10):
    # Simple landmark extraction (using random points as landmarks)
    if points is None or len(points) == 0:
        return None
    
    n_points = min(max_landmarks, len(points))
    landmark_indices = np.random.choice(len(points), n_points, replace=False)
    landmarks = points[landmark_indices]
    
    # Convert to polar coordinates relative to drone's position
    landmarks_polar = []
    for lm in landmarks:
        r = np.linalg.norm(lm[:2])  # Range
        bearing = np.arctan2(lm[1], lm[0])  # Bearing
        landmarks_polar.append([r, bearing])
    
    return np.array(landmarks_polar)

def update_slam(u, z):
    global particles, x_est
    particles = fast_slam2(particles, u, z)
    x_est = calc_final_state(particles)
    return x_est

def check_all_lidars(ax):
    print("\nChecking LiDAR sensors and updating SLAM:")
    current_points = collect_lidar_data()
    landmarks = None
    if current_points is not None:
        landmarks = extract_landmarks(current_points)
        if landmarks is not None:
            z = np.vstack((landmarks.T, np.arange(len(landmarks))))
            u = np.array([[1.0], [0.1]])  # [velocity, yaw_rate]
            x_est = update_slam(u, z)
            print("Estimated position:", x_est[:2].T)
            print("Estimated yaw:", np.rad2deg(x_est[2])[0])
            update_plot(ax, particles, x_est, landmarks)
        else:
            print("No landmarks detected")
    else:
        print("No LiDAR data available")
    print("------------------------")
try:
    fig, ax = init_plot()

    # Takeoff
    print("Taking off...")
    client.takeoffAsync().join()
    check_all_lidars(ax)
    
    # Move to a position
    print("Moving to position...")
    client.moveToPositionAsync(10, 0, 0, 2).join()  # Note: negative z is up
    check_all_lidars(ax)
    
    # Move in a square pattern
    for _ in range(4):
        client.moveToPositionAsync(10, 0, 0, 2).join()
        check_all_lidars(ax)
     
    # Return to start
    print("Returning to start...")
    client.moveToPositionAsync(5, 0, 0, 2).join()
    check_all_lidars(ax)
    
    # Land
    print("Landing...")
    client.landAsync().join()
    check_all_lidars(ax)

    plt.ioff()
    plt.show()

except KeyboardInterrupt:
    print("Program interrupted. Landing...")
    client.landAsync().join()

finally:
    # Disarm and release control
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Flight completed.")