import airsim
import time

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
print("Taking off...")
client.takeoffAsync().join()

print("Moving to position...")
target_x, target_y, target_z = 20, 0, 0 