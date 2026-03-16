import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import math
import numpy as np

# Config

NUMBER_OF_PARTICLES = 1000
MAP_DIMENSIONS = (144, 144)
PLOT_PADDING = 15
ROBOT_SIZE = (16, 18)

# setting up the initial robot position & variables to keep track of both real and predicted pos of bot

robot_x = 72.0
robot_y = 40
robot_theta = 70

initial_x = 40.0
initial_y = 40.0
initial_robot_theta = 70.0

initial_x = robot_x
initial_y = robot_y
initial_robot_theta = robot_theta

predicted_x = initial_x
predicted_y = initial_y
predicted_theta = initial_robot_theta

# define the amount of noise for the sensors and the initial spread of the particles

distance_sensor_noise = 0.5 # in inches
odometry_noise = 3 # in inches

odometry_theta_noise = 1 # in degrees
theta_noise = 0.2 # in degrees

# simulate big slip (rare)

big_slip_chance = 0.03
big_slip_distance_upper_limit = 6 # in inches
big_slip_distance_lower_limit = 2 # in inches

big_slip_theta_change_upper_limit = 15 # in degrees
big_slip_theta_change_lower_limit = 5 # in degrees

# spawn particles around the initial position with some noise
# ensure they are within the map boundaries

spawn_x = np.random.normal(initial_x, 20.0, NUMBER_OF_PARTICLES)
spawn_y = np.random.normal(initial_y, 20.0, NUMBER_OF_PARTICLES)
spawn_theta = np.random.normal(initial_robot_theta, 20.0, NUMBER_OF_PARTICLES) % 360
spawn_x = np.clip(spawn_x, 0, MAP_DIMENSIONS[0])
spawn_y = np.clip(spawn_y, 0, MAP_DIMENSIONS[1])

particles = np.column_stack((spawn_x, spawn_y, spawn_theta, np.ones(NUMBER_OF_PARTICLES) / NUMBER_OF_PARTICLES))

particle_vector_length = 2.5
particle_theta_rad = np.radians(particles[:, 2])

particle_dx = particle_vector_length * np.sin(particle_theta_rad)
particle_dy = particle_vector_length * np.cos(particle_theta_rad)

# setup and define the distance sensors (up, down, left, right) with their positions and directions relative to the robot

distance_sensor_available = ["up", "down", "left", "right"]
distance_sensors = {
    "up": ((0, 0), (0.0, 0.0)),
    "down": ((0, 0), (0.0, 0.0)),
    "left": ((0, 0), (0.0, 0.0)),
    "right": ((0, 0), (0.0, 0.0)),
}

distance_sensor_distances = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}

keys = {"up": False, "down": False, "left": False, "right": False}




# Visualization setup

fig, ax = plt.subplots(figsize=(8, 8))

field_shape = patches.Rectangle(
    (0, 0),
    MAP_DIMENSIONS[0],
    MAP_DIMENSIONS[1],
    linewidth=1,
    edgecolor="black",
    facecolor="none",
)

robot_shape = patches.Rectangle(
    (0, 0),
    ROBOT_SIZE[0],
    ROBOT_SIZE[1],
    rotation_point="center",
    color="blue",
    alpha=0.5,
)

# self-explanatory, just the robot's center and heading line for visualization

(center_dot,) = ax.plot([], [], marker="o", color="blue", markersize=5)  # Center dot
(heading_line,) = ax.plot([], [], color="red", linewidth=2, zorder=5)
(particles_scatter,) = ax.plot([], [], "r.", markersize=2, alpha=0.5, zorder=2)
(up_distance_line,) = ax.plot([], [], color="green", linewidth=2, zorder=5)
(down_distance_line,) = ax.plot([], [], color="green", linewidth=2, zorder=5)
(left_distance_line,) = ax.plot([], [], color="green", linewidth=2, zorder=5)
(right_distance_line,) = ax.plot([], [], color="green", linewidth=2, zorder=5)

particles_heading_vectors = ax.quiver(

    particles[:, 0],
    particles[:, 1],
    particle_dx,
    particle_dy,
    angles="xy",
    scale_units="xy",
    scale=1,
    color="orange",
    alpha=0.35,
    width=0.002,
    zorder=1,

)

# where the logic for mcl will go, we will read the sensor values, update the particles' weights based on how well they match the sensor readings, resample the particles based on their weights, and then calculate the new predicted position of the robot based on the particles' distribution
class mcl:

    def __init__(self):
        
        self.imu = robot_theta + np.random.normal(0, theta_noise)  # Simulated IMU reading with noise

        # distance sensors(distances to wall)

        self.sensors["top"] = distance_sensor_distances["up"] + np.random.normal(
            0, distance_sensor_noise
        )
        self.sensors["bottom"] = distance_sensor_distances["down"] + np.random.normal(
            0, distance_sensor_noise
        )
        self.sensors["left"] = distance_sensor_distances["left"] + np.random.normal(
            0, distance_sensor_noise
        )
        self.sensors["right"] = distance_sensor_distances["right"] + np.random.normal(
            0, distance_sensor_noise
        )

        # odometry readings (change in position and orientation since last update)

        self.last_odometry_x = robot_x + np.random.normal(0, odometry_noise)
        self.last_odometry_y = robot_y + np.random.normal(0, odometry_noise)
        self.last_odometry_theta = self.imu

    def update_odometry_and_sensor_readings(self):

        # meant to simulate the odometry readings we would get from the robot's encoders and IMU, which will be used to predict the new position of the particles before we update their weights based on the sensor readings

        self.odometry_x = robot_x + np.random.normal(0, odometry_noise)
        self.odometry_y = robot_y + np.random.normal(0, odometry_noise)
        self.odometry_theta = self.imu

        if np.random.rand() < big_slip_chance:

            slip_distance = np.random.uniform(
                big_slip_distance_lower_limit, big_slip_distance_upper_limit
            )

            slip_theta_change = np.random.uniform(
                big_slip_theta_change_lower_limit, big_slip_theta_change_upper_limit
                )
            
            self.odometry_x += slip_distance * math.sin(math.radians(robot_theta))
            self.odometry_y += slip_distance * math.cos(math.radians(robot_theta))
            self.odometry_theta = (self.odometry_theta + slip_theta_change) % 360

        # update the distance sensor readings with noise, which will be used to update the weights of the particles based on how well their predicted sensor readings match these actual noisy readings

        self.sensors["top"] = distance_sensor_distances["up"] + np.random.normal(
            0, distance_sensor_noise
        )
        self.sensors["bottom"] = distance_sensor_distances["down"] + np.random.normal(
            0, distance_sensor_noise
        )
        self.sensors["left"] = distance_sensor_distances["left"] + np.random.normal(
            0, distance_sensor_noise
        )
        self.sensors["right"] = distance_sensor_distances["right"] + np.random.normal(
            0, distance_sensor_noise
        )


    # update happens here or sum like that

    def predict(self):

        self.update_odometry_and_sensor_readings()
        self.imu = robot_theta + np.random.normal(0, theta_noise)
        
        # get the change in position and orientation from the odometry readings, which will be used to predict the new position of the particles

        delta_x = self.odometry_x - self.last_odometry_x
        delta_y = self.odometry_y - self.last_odometry_y
        delta_theta = (self.odometry_theta - self.last_odometry_theta) % 360

        self.last_odometry_x = self.odometry_x
        self.last_odometry_y = self.odometry_y
        self.last_odometry_theta = self.odometry_theta

        for i in range(NUMBER_OF_PARTICLES):

            # update the particle's position based on the odometry readings, adding some noise to simulate uncertainty

            particles[i, 0] += delta_x + np.random.normal(0, odometry_noise)
            particles[i, 1] += delta_y + np.random.normal(0, odometry_noise)
            particles[i, 2] = (particles[i, 2] + delta_theta + np.random.normal(0, theta_noise)) % 360

            # keep the particles within the map boundaries

            particles[i, 0] = np.clip(particles[i, 0], 0, MAP_DIMENSIONS[0])
            particles[i, 1] = np.clip(particles[i, 1], 0, MAP_DIMENSIONS[1])
            particles[i, 2] = np.clip(particles[i, 2], 0, 360)



    def calculate_expected_sensor_reading(self, particle_x, particle_y, particle_theta, sensor_name):

        # calculate the expected sensor reading for a given particle based on its position and orientation, which will be compared to the actual sensor reading to update the particle's weight

        theta_rad = math.radians(particle_theta)

        if sensor_name == "up":
            direction_x = math.sin(theta_rad)
            direction_y = math.cos(theta_rad)
        elif sensor_name == "down":
            direction_x = -math.sin(theta_rad)
            direction_y = -math.cos(theta_rad)
        elif sensor_name == "left":
            direction_x = math.cos(theta_rad)
            direction_y = -math.sin(theta_rad)
        elif sensor_name == "right":
            direction_x = -math.cos(theta_rad)
            direction_y = math.sin(theta_rad)

        sensor_end_x, sensor_end_y, expected_distance = get_sensor_ray(
            particle_x, particle_y, direction_x, direction_y
        )

        return expected_distance

            
    def update_weights(self):

        # update the weights of the particles based on how similar their predicted sensor readings are to the actual sensor readings, which will be used to resample the particles and calculate the new predicted position of the robot
        
        for i in range(NUMBER_OF_PARTICLES):

            sensor_weight_results = []

            for sensor_name in distance_sensor_available:

                expected_distance = self.sensors[sensor_name]
                particle_x, particle_y, particle_theta = particles[i]

                # calculate the expected sensor reading for this particle based on its position and orientation, which will be compared to the actual sensor reading to update the particle's weight

                predicted_distance = self.calculate_expected_sensor_reading(
                    particle_x, particle_y, particle_theta, sensor_name)
                
                sensor_weight_results.append(
                    np.exp(-0.5 * ((predicted_distance - expected_distance) / distance_sensor_noise) ** 2)
                )

            particles[i, 3] = np.prod(sensor_weight_results)


                






    def update(self):

        self.predict()

        self.update_weights()        

mcl_instance = mcl()



# function to calculate the endpoint of a distance sensor ray based on its origin and direction
# + ensuring it intersects with the field boundaries

def get_sensor_ray(origin_x, origin_y, direction_x, direction_y):
    
    distances_to_wall = []

    if direction_x > 0:
        distances_to_wall.append((MAP_DIMENSIONS[0] - origin_x) / direction_x)
    elif direction_x < 0:
        distances_to_wall.append((0 - origin_x) / direction_x)

    if direction_y > 0:
        distances_to_wall.append((MAP_DIMENSIONS[1] - origin_y) / direction_y)
    elif direction_y < 0:
        distances_to_wall.append((0 - origin_y) / direction_y)

    ray_distance = min(distance for distance in distances_to_wall if distance >= 0)

    return (
        origin_x + direction_x * ray_distance,
        origin_y + direction_y * ray_distance,
        ray_distance,
    )

# function to update the distance sensors' positions, directions, and visual lines based on the robot's current orientation

def update_distance_sensors(theta_rad):
    
    forward_x = math.sin(theta_rad)
    forward_y = math.cos(theta_rad)

    right_x = math.cos(theta_rad)
    right_y = -math.sin(theta_rad)

    left_x = -right_x
    left_y = -right_y

    half_width = ROBOT_SIZE[0] / 2
    half_length = ROBOT_SIZE[1] / 2

    # defining the sensor origins and directions based on the robot's current position and orientation

    sensor_definitions = {
        "up": {
            "origin": (
                robot_x + forward_x * half_length,
                robot_y + forward_y * half_length,
            ),
            "direction": (forward_x, forward_y),
            "line": up_distance_line,
        },
        "down": {
            "origin": (
                robot_x - forward_x * half_length,
                robot_y - forward_y * half_length,
            ),
            "direction": (-forward_x, -forward_y),
            "line": down_distance_line,
        },
        "left": {
            "origin": (robot_x + left_x * half_width, robot_y + left_y * half_width),
            "direction": (left_x, left_y),
            "line": left_distance_line,
        },
        "right": {
            "origin": (robot_x + right_x * half_width, robot_y + right_y * half_width),
            "direction": (right_x, right_y),
            "line": right_distance_line,
        },
    }

    for sensor_name, sensor_definition in sensor_definitions.items():
        sensor_x, sensor_y = sensor_definition["origin"]
        direction_x, direction_y = sensor_definition["direction"]

        sensor_end_x, sensor_end_y, sensor_distance = get_sensor_ray(
            sensor_x, sensor_y, direction_x, direction_y
        )

        distance_sensors[sensor_name] = (
            (sensor_x, sensor_y),
            (sensor_end_x, sensor_end_y),
        )
        distance_sensor_distances[sensor_name] = sensor_distance
        sensor_definition["line"].set_data(
            [sensor_x, sensor_end_x], [sensor_y, sensor_end_y]
        )

# just making sure any part of the robot doesn't go out of bounds when it moves or rotates, considering its size and orientation

def keep_robot_in_bounds(theta_rad):
    global robot_x, robot_y

    half_width = ROBOT_SIZE[0] / 2
    half_length = ROBOT_SIZE[1] / 2
    abs_cos = abs(math.cos(theta_rad))
    abs_sin = abs(math.sin(theta_rad))

    x_extent = half_width * abs_cos + half_length * abs_sin
    y_extent = half_width * abs_sin + half_length * abs_cos

    robot_x = max(x_extent, min(MAP_DIMENSIONS[0] - x_extent, robot_x))
    robot_y = max(y_extent, min(MAP_DIMENSIONS[1] - y_extent, robot_y))

# initialization function for the animation, setting up the plot limits, aspect ratio, and initial positions of all elements

def init():

    
    theta_rad = math.radians(robot_theta)
    keep_robot_in_bounds(theta_rad)
    update_distance_sensors(theta_rad)

    ax.set_xlim(-PLOT_PADDING, MAP_DIMENSIONS[0] + PLOT_PADDING)
    ax.set_ylim(-PLOT_PADDING, MAP_DIMENSIONS[1] + PLOT_PADDING)

    ax.set_aspect("equal")
    ax.add_patch(field_shape)
    ax.add_patch(robot_shape)

    particles_scatter.set_data(particles[:, 0], particles[:, 1])
    particle_theta_rad = np.radians(particles[:, 2])
    particle_dx = particle_vector_length * np.sin(particle_theta_rad)
    particle_dy = particle_vector_length * np.cos(particle_theta_rad)
    particles_heading_vectors.set_offsets(particles[:, :2])
    particles_heading_vectors.set_UVC(particle_dx, particle_dy)

    return (
        field_shape,
        robot_shape,
        center_dot,
        heading_line,
        particles_scatter,
        particles_heading_vectors,
        up_distance_line,
        down_distance_line,
        left_distance_line,
        right_distance_line,
    )


def on_press(event):
    if event.key in keys:
        keys[event.key] = True


def on_release(event):
    if event.key in keys:
        keys[event.key] = False


# update

def update(frame):
    global robot_x, robot_y, robot_theta

    if keys["left"]:
        robot_theta = (robot_theta - 3) % 360
    if keys["right"]:
        robot_theta = (robot_theta + 3) % 360

    theta_rad = math.radians(robot_theta)

    if keys["up"]:

        robot_x += 1.5 * math.sin(theta_rad)
        robot_y += 1.5 * math.cos(theta_rad)

    if keys["down"]:
        robot_x -= 1.5 * math.sin(theta_rad)
        robot_y -= 1.5 * math.cos(theta_rad)

    keep_robot_in_bounds(theta_rad)

    update_distance_sensors(theta_rad)

    mcl_instance.update()
    

    particles_scatter.set_data(particles[:, 0], particles[:, 1])

    particle_theta_rad = np.radians(particles[:, 2])
    particle_dx = particle_vector_length * np.sin(particle_theta_rad)
    particle_dy = particle_vector_length * np.cos(particle_theta_rad)
    particles_heading_vectors.set_offsets(particles[:, :2])
    particles_heading_vectors.set_UVC(particle_dx, particle_dy)

    # displaying robot

    corner_x = robot_x - ROBOT_SIZE[0] / 2
    corner_y = robot_y - ROBOT_SIZE[1] / 2

    robot_shape.set_xy((corner_x, corner_y))
    robot_shape.set_angle(-robot_theta)

    center_dot.set_data([robot_x], [robot_y])

    # thetha calculation w arrow

    arrow_length = 15.0

    dx = arrow_length * math.sin(theta_rad)
    dy = arrow_length * math.cos(theta_rad)

    heading_line.set_data([robot_x, robot_x + dx], [robot_y, robot_y + dy])

    ax.set_title(f"VEX Virtual Field (Theta: {robot_theta:.1f}°)")

    return (
        field_shape,
        robot_shape,
        center_dot,
        heading_line,
        particles_scatter,
        particles_heading_vectors,
        up_distance_line,
        down_distance_line,
        left_distance_line,
        right_distance_line,
    )

# connecting the key press and release events to their respective handlers

fig.canvas.mpl_connect("key_press_event", on_press)
fig.canvas.mpl_connect("key_release_event", on_release)

ani = animation.FuncAnimation(fig, update, init_func=init, interval=5, blit=True)

plt.show()
