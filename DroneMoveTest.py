import airsim
import numpy as np
import time

class QuadrotorPotentialField:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.lidar_name = "LidarFront"

       
        self.q_repel = 50  
        self.q_attract = 10  
        self.dist_threshold = 10  
        self.speed = 2
        self.min_distance = 3 
      
        self.min_height = -1.5 
        self.max_height = -3.0 

    def get_lidar_data(self):
        return self.client.getLidarData(lidar_name=self.lidar_name)

    def process_lidar_data(self, lidar_data):
        if lidar_data.point_cloud:
            points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            
           
            distances = np.linalg.norm(points[:, :2], axis=1) 
            mask = (distances > 2) & (distances < self.dist_threshold) & (points[:, 2] > -1)
            points = points[mask]
            
            return points
        return np.array([])

    def compute_force(self, point_cloud, vehicle_coord, target_coord):
        
        if len(point_cloud) == 0:
            repelling_force = np.zeros(3)
        else:
            diff = point_cloud - vehicle_coord.reshape(1, 3)
            dist = np.linalg.norm(diff[:, :2], axis=1, keepdims=True)  
            repel_directions = -diff / np.repeat(dist, 3, axis=1)
            repel_directions[:, 2] = 0 
            distance_component = (1 / dist - 1 / self.dist_threshold) ** 2
            repelling_force = np.sum(repel_directions * self.q_repel * distance_component, axis=0)

        attraction_vector = target_coord - vehicle_coord
        attraction_dist = np.linalg.norm(attraction_vector[:2])  
        attraction_force = self.q_attract * attraction_vector / attraction_dist
        attraction_force[2] = 0  
     
        min_attraction = 0.5 
        if np.linalg.norm(attraction_force) < min_attraction:
            attraction_force = min_attraction * attraction_force / np.linalg.norm(attraction_force)

        total_force = repelling_force + attraction_force

        
        max_force = 5
        if np.linalg.norm(total_force) > max_force:
            total_force = max_force * total_force / np.linalg.norm(total_force)

        return total_force

    def run(self):
        self.client.takeoffAsync().join()
        time.sleep(2)  
        goal = np.array([20, 0, -2]) 
        
        while True:
            state = self.client.getMultirotorState()
            vehicle_coord = np.array([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                state.kinematics_estimated.position.z_val
            ])

            lidar_data = self.get_lidar_data()
            point_cloud = self.process_lidar_data(lidar_data)

            if len(point_cloud) > 0:
                closest_point = np.min(np.linalg.norm(point_cloud[:, :2], axis=1))
                print(f"Detected {len(point_cloud)} points. Closest point at {closest_point:.2f} m")
            else:
                print("No obstacles detected")

            print(f"Current position: {vehicle_coord}")

            force = self.compute_force(point_cloud, vehicle_coord, goal)
            
            velocity = self.speed * force / np.linalg.norm(force)

         
            if vehicle_coord[2] <= self.max_height:
                velocity[2] = max(0, velocity[2])
            elif vehicle_coord[2] >= self.min_height:
                velocity[2] = min(0, velocity[2])
            else:
                velocity[2] = 0

            print(f"Velocity: {velocity}")

           
            self.client.moveByVelocityAsync(velocity[0], velocity[1], velocity[2], duration=0.5)

            
            if np.linalg.norm(vehicle_coord[:2] - goal[:2]) < 0.5:
                print("Goal reached!")
                break

            time.sleep(0.1)

        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

if __name__ == "__main__":
    quad_pf = QuadrotorPotentialField()
    quad_pf.run()