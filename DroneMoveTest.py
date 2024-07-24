import airsim
import time

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

try:
    # Takeoff
    print("Taking off...")
    client.takeoffAsync().join()
    
    # Move to a position
    print("Moving to position...")
    client.moveToPositionAsync(-10, 10, -10, 5).join()
    
    # Hover for a few seconds
    print("Hovering...")
    time.sleep(5)
    
    # Move to another position
    print("Moving to another position...")
    client.moveToPositionAsync(0, 0, -10, 5).join()
    
    # Hover for a few seconds
    print("Hovering...")
    time.sleep(5)
    
    # Land
    print("Landing...")
    client.landAsync().join()

except KeyboardInterrupt:
    print("Program interrupted. Landing...")
    client.landAsync().join()

finally:
    # Disarm and release control
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Flight completed.")