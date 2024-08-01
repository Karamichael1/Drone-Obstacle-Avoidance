import airsim
import numpy as np
import math

class QuadrotorDWA:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # LiDAR sensor names
        self.lidar_names = ["LidarFront", "LidarBack", "LidarTop", "LidarBottom"]

        # DWA parameters
        self.max_speed = 5.0  # m/s
        self.max_yawrate = 1.0  # rad/s
        self.max_accel = 2.0  # m/s^2
        self.max_dyawrate = 1.0  # rad/s^2
        self.v_resolution = 0.1  # m/s
        self.yawrate_resolution = 0.1  # rad/s
        self.dt = 0.1  # s, prediction time
        self.predict_time = 3.0  # s, total prediction time
        self.obstacle_radius = 1.0  # m

    def get_lidar_data(self):
        lidar_data = []
        for lidar_name in self.lidar_names:
            lidar_data.append(self.client.getLidarData(lidar_name=lidar_name))
        return lidar_data

    def dwa_control(self, x, goal, obstacles):
        # Dynamic Window
        dw = self.calc_dynamic_window(x)

        # Evaluate all possible velocities
        best_u = None
        best_score = float('-inf')
        
        for v in np.arange(dw[0], dw[1], self.v_resolution):
            for y in np.arange(dw[2], dw[3], self.yawrate_resolution):
                trajectory = self.predict_trajectory(x, v, y)
                
                # Scoring
                heading_score = self.calc_heading_score(trajectory, goal)
                dist_score = self.calc_dist_score(trajectory, obstacles)
                velocity_score = self.calc_velocity_score(v)
                
                score = heading_score + dist_score + velocity_score
                
                if score > best_score:
                    best_u = [v, y]
                    best_score = score

        return best_u

    def calc_dynamic_window(self, x):
        # Dynamic window from robot specification
        Vs = [0.0, self.max_speed, -self.max_yawrate, self.max_yawrate]

        # Dynamic window from motion model
        Vd = [x[3] - self.max_accel * self.dt,
              x[3] + self.max_accel * self.dt,
              x[4] - self.max_dyawrate * self.dt,
              x[4] + self.max_dyawrate * self.dt]

        # Final dynamic window
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def predict_trajectory(self, x, v, y):
        trajectory = []
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, y], self.dt)
            trajectory.append(x)
            time += self.dt
        return trajectory

    def motion(self, x, u, dt):
        # Simple motion model for quadrotor
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[2] += u[1] * dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    def calc_heading_score(self, trajectory, goal):
        dx = goal[0] - trajectory[-1][0]
        dy = goal[1] - trajectory[-1][1]
        error_angle = math.atan2(dy, dx)
        cost = error_angle - trajectory[-1][2]
        return math.pi - abs(cost)

    def calc_dist_score(self, trajectory, obstacles):
        min_dist = float('inf')
        for x in trajectory:
            for obstacle in obstacles:
                d = math.sqrt((x[0] - obstacle[0])**2 + (x[1] - obstacle[1])**2)
                if d <= self.obstacle_radius:
                    return float('-inf')  # Collision
                min_dist = min(min_dist, d)
        return min_dist

    def calc_velocity_score(self, v):
        return v  # Prefer higher speeds

    def run(self):
        self.client.takeoffAsync().join()

        goal = [10, 10, -5]  # Example goal position
        
        while True:
            state = self.client.getMultirotorState()
            x = [state.kinematics_estimated.position.x_val,
                 state.kinematics_estimated.position.y_val,
                 state.kinematics_estimated.orientation.z_val,
                 state.kinematics_estimated.linear_velocity.x_val,
                 state.kinematics_estimated.angular_velocity.z_val]

            lidar_data = self.get_lidar_data()
            obstacles = self.process_lidar_data(lidar_data)

            u = self.dwa_control(x, goal, obstacles)

            if u is not None:
                self.client.moveByVelocityAsync(u[0]*math.cos(x[2]), u[0]*math.sin(x[2]), 0, 0.1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, u[1])).join()
            
            # Check if goal is reached
            if math.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2) < 0.5:
                print("Goal reached!")
                break

    def process_lidar_data(self, lidar_data):
        obstacles = []
        for i, lidar in enumerate(lidar_data):
            if lidar.point_cloud:
                points = np.array(lidar.point_cloud, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0]/3), 3))
                
                # Transform points to global coordinate system
                lidar_pose = self.client.simGetVehiclePose(self.lidar_names[i])
                lidar_orientation = airsim.to_eularian_angles(lidar_pose.orientation)
                
                R = self.rotation_matrix(lidar_orientation[0], lidar_orientation[1], lidar_orientation[2])
                T = np.array([lidar_pose.position.x_val, lidar_pose.position.y_val, lidar_pose.position.z_val])
                
                global_points = np.dot(points, R.T) + T
                
                for point in global_points:
                    if np.linalg.norm(point) < 20:  # Consider points within 20m
                        obstacles.append(point)
        return obstacles

    def rotation_matrix(self, roll, pitch, yaw):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(roll), -math.sin(roll)],
                        [0, math.sin(roll), math.cos(roll)]])
        
        R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                        [0, 1, 0],
                        [-math.sin(pitch), 0, math.cos(pitch)]])
        
        R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                        [math.sin(yaw), math.cos(yaw), 0],
                        [0, 0, 1]])
        
        return np.dot(R_z, np.dot(R_y, R_x))

    def run(self):
        self.client.takeoffAsync().join()

        goal = [10, 10, -5]  # Example goal position
        
        while True:
            state = self.client.getMultirotorState()
            x = [state.kinematics_estimated.position.x_val,
                 state.kinematics_estimated.position.y_val,
                 state.kinematics_estimated.position.z_val,
                 state.kinematics_estimated.linear_velocity.x_val,
                 state.kinematics_estimated.linear_velocity.y_val,
                 state.kinematics_estimated.linear_velocity.z_val,
                 *airsim.to_eularian_angles(state.kinematics_estimated.orientation)]

            lidar_data = self.get_lidar_data()
            obstacles = self.process_lidar_data(lidar_data)

            u = self.dwa_control(x, goal, obstacles)

            if u is not None:
                self.client.moveByVelocityAsync(u[0]*math.cos(x[6]), u[0]*math.sin(x[6]), 0, 0.1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, u[1])).join()
            
            # Check if goal is reached
            if math.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2 + (x[2] - goal[2])**2) < 0.5:
                print("Goal reached!")
                break

if __name__ == "__main__":
    quad_dwa = QuadrotorDWA()
    quad_dwa.run()