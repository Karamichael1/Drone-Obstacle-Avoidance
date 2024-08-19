import airsim
import numpy as np
import math

class QuadrotorDWA3D:
    class Config:
        def __init__(self):
            self.max_speed = 5.0  # m/s
            self.min_speed = -2.0  # m/s
            self.max_yaw_rate = 1.0  # rad/s
            self.max_pitch_rate = 0.5  # rad/s
            self.max_roll_rate = 0.5  # rad/s
            self.max_accel = 2.0  # m/s^2
            self.max_angular_accel = 1.0  # rad/s^2
            self.v_resolution = 0.1  # m/s
            self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # rad/s
            self.dt = 0.1  # s
            self.predict_time = 3.0  # s
            self.to_goal_cost_gain = 1.0
            self.speed_cost_gain = 1.0
            self.obstacle_cost_gain = 1.0
            self.robot_radius = 0.5  # m

    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.config = self.Config()

        # Set up camera
        camera_name = "0"
        camera_pose = airsim.Pose(airsim.Vector3r(0.5, 0, 0.1), airsim.to_quaternion(0, 0, 0))
        self.client.simSetCameraPose(camera_name, camera_pose)

        self.fov = 90  # Field of view in degrees
        self.height = 144
        self.width = 256

    def get_state(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        linear_vel = state.kinematics_estimated.linear_velocity
        angular_vel = state.kinematics_estimated.angular_velocity

        x = np.array([
            pos.x_val, pos.y_val, pos.z_val,
            *airsim.to_eularian_angles(orientation),
            linear_vel.x_val, linear_vel.y_val, linear_vel.z_val,
            angular_vel.x_val, angular_vel.y_val, angular_vel.z_val
        ])
        return x

    def set_goal(self, x, y, z):
        self.goal = np.array([x, y, z])

    def motion(self, x, u, dt):
        # Simplified 3D motion model
        new_x = np.array(x)
        new_x[0] += u[0] * dt
        new_x[1] += u[1] * dt
        new_x[2] += u[2] * dt
        new_x[3] += u[3] * dt
        new_x[4] += u[4] * dt
        new_x[5] += u[5] * dt
        new_x[6:] = u
        return new_x

    def calc_dynamic_window(self, x):
        vs = [self.config.min_speed, self.config.max_speed] * 3 + \
             [-self.config.max_yaw_rate, self.config.max_yaw_rate] * 3
        vd = [
            x[6] - self.config.max_accel * self.config.dt,
            x[6] + self.config.max_accel * self.config.dt,
            x[7] - self.config.max_accel * self.config.dt,
            x[7] + self.config.max_accel * self.config.dt,
            x[8] - self.config.max_accel * self.config.dt,
            x[8] + self.config.max_accel * self.config.dt,
            x[9] - self.config.max_angular_accel * self.config.dt,
            x[9] + self.config.max_angular_accel * self.config.dt,
            x[10] - self.config.max_angular_accel * self.config.dt,
            x[10] + self.config.max_angular_accel * self.config.dt,
            x[11] - self.config.max_angular_accel * self.config.dt,
            x[11] + self.config.max_angular_accel * self.config.dt
        ]
        dw = [max(vs[i], vd[i]) for i in range(0, 12, 2)] + [min(vs[i], vd[i]) for i in range(1, 12, 2)]
        return dw

    def predict_trajectory(self, x_init, v, y):
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.config.predict_time:
            x = self.motion(x, [v[0], v[1], v[2], y[0], y[1], y[2]], self.config.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.config.dt
        return trajectory

    def calc_obstacle_cost(self, trajectory, obstacles):
        min_dist = float('inf')
        for point in trajectory:
            for obstacle in obstacles:
                d = np.linalg.norm(point[:3] - obstacle)
                if d <= self.config.robot_radius:
                    return float('inf')
                min_dist = min(min_dist, d - self.config.robot_radius)
        return 1.0 / min_dist if min_dist > 0 else float('inf')

    def calc_to_goal_cost(self, trajectory):
        dx = self.goal[0] - trajectory[-1, 0]
        dy = self.goal[1] - trajectory[-1, 1]
        dz = self.goal[2] - trajectory[-1, 2]
        return np.linalg.norm([dx, dy, dz])

    def calc_control_and_trajectory(self, x, dw, ob):
        x_init = x[:]
        min_cost = float('inf')
        best_u = [0.0] * 6
        best_trajectory = np.array([x])

        for vx in np.arange(dw[0], dw[1], self.config.v_resolution):
            for vy in np.arange(dw[2], dw[3], self.config.v_resolution):
                for vz in np.arange(dw[4], dw[5], self.config.v_resolution):
                    for yaw_rate in np.arange(dw[6], dw[7], self.config.yaw_rate_resolution):
                        v = [vx, vy, vz]
                        y = [0, 0, yaw_rate]  # Simplified: only considering yaw rate
                        trajectory = self.predict_trajectory(x_init, v, y)

                        to_goal_cost = self.config.to_goal_cost_gain * self.calc_to_goal_cost(trajectory)
                        speed_cost = self.config.speed_cost_gain * (self.config.max_speed - trajectory[-1, 6])
                        ob_cost = self.config.obstacle_cost_gain * self.calc_obstacle_cost(trajectory, ob)

                        final_cost = to_goal_cost + speed_cost + ob_cost

                        if min_cost >= final_cost:
                            min_cost = final_cost
                            best_u = [*v, *y]
                            best_trajectory = trajectory

        return best_u, best_trajectory

    def dwa_control(self, x, ob):
        dw = self.calc_dynamic_window(x)
        u, trajectory = self.calc_control_and_trajectory(x, dw, ob)
        return u, trajectory

    def get_obstacles(self):
        # Get depth image
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
        depth_image = airsim.list_to_2d_float_array(responses[0].image_data_float, self.width, self.height)
        
        # Calculate obstacle positions
        obstacles = []
        vehicle_pose = self.client.simGetVehiclePose()
        vehicle_position = np.array([vehicle_pose.position.x_val, vehicle_pose.position.y_val, vehicle_pose.position.z_val])

        for i in range(self.height):
            for j in range(self.width):
                depth = depth_image[i][j]
                if 1 < depth < 15:  # Consider obstacles between 1 and 15 meters
                    # Calculate ray direction
                    x = j - self.width / 2
                    y = i - self.height / 2
                    z = self.width / (2 * np.tan(self.fov * np.pi / 360))
                    ray = np.array([x, y, z])
                    ray = ray / np.linalg.norm(ray)

                    # Calculate obstacle position in world coordinates
                    obstacle_position = vehicle_position + depth * ray
                    obstacles.append(obstacle_position)

        # Downsample obstacles to reduce computation time
        if len(obstacles) > 100:
            obstacles = obstacles[::len(obstacles)//100]

        return obstacles

    def run(self):
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.takeoffAsync().join()

        x = self.get_state()
        self.set_goal(20, 10, -2)  # Example goal

        while True:
            ob = self.get_obstacles()
            u, predicted_trajectory = self.dwa_control(x, ob)

            self.client.moveByVelocityZAsync(u[0], u[1], u[2], 
                                             duration=self.config.dt,
                                             yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=u[5]))

            x = self.get_state()

            # Check if goal is reached
            if np.linalg.norm(x[:3] - self.goal) < self.config.robot_radius:
                print("Goal reached!")
                break

        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

if __name__ == "__main__":
    quad_dwa = QuadrotorDWA3D()
    quad_dwa.run()