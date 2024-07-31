import airsim
import numpy as np
import time

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

def check_lidar(lidar_name):
    lidar_data = client.getLidarData(lidar_name=lidar_name)
    
    if len(lidar_data.point_cloud) < 3:
        print(f"{lidar_name}: No points detected.")
        return
    
    points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0]/3), 3))
    
    print(f"{lidar_name}:")
    print(f"  Point count: {points.shape[0]}")
    print(f"  Min distance: {np.min(np.linalg.norm(points, axis=1)):.2f}")
    print(f"  Max distance: {np.max(np.linalg.norm(points, axis=1)):.2f}")

def check_all_lidars():
    print("\nChecking LiDAR sensors:")
    for lidar_name in lidar_names:
        check_lidar(lidar_name)
    print("------------------------")

try:
    # Takeoff
    print("Taking off...")
    client.takeoffAsync().join()
    check_all_lidars()
    
    # Move to a position
    print("Moving to position...")
    client.moveToPositionAsync(10, 0, 0, 5).join()  # Note: negative z is up
    check_all_lidars()
    
    # Hover for a few seconds
    print("Hovering...")
    for _ in range(5):
        time.sleep(1)
        check_all_lidars()
    
    # Land
    print("Landing...")
    client.landAsync().join()
    check_all_lidars()

except KeyboardInterrupt:
    print("Program interrupted. Landing...")
    client.landAsync().join()

finally:
    # Disarm and release control
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Flight completed.")