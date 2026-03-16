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

distance_sensor_noise = 0.5  # in inches
odometry_noise = 3  # in inches

odometry_theta_noise = 1  # in degrees
theta_noise = 0.2  # in degrees

# simulate big slip (rare)

big_slip_chance = 0.03
big_slip_distance_upper_limit = 6  # in inches
big_slip_distance_lower_limit = 2  # in inches

big_slip_theta_change_upper_limit = 15  # in degrees
big_slip_theta_change_lower_limit = 5  # in degrees

# spawn particles around the initial position with some noise
# ensure they are within the map boundaries

spawn_x = np.random.normal(initial_x, 20.0, NUMBER_OF_PARTICLES)
spawn_y = np.random.normal(initial_y, 20.0, NUMBER_OF_PARTICLES)
spawn_theta = np.random.normal(initial_robot_theta, 5.0, NUMBER_OF_PARTICLES) % 360
spawn_x = np.clip(spawn_x, 0, MAP_DIMENSIONS[0])
spawn_y = np.clip(spawn_y, 0, MAP_DIMENSIONS[1])

particles = np.column_stack(
    (spawn_x, spawn_y, spawn_theta, np.ones(NUMBER_OF_PARTICLES) / NUMBER_OF_PARTICLES)
)

particle_vector_length = 2.5
particle_theta_rad = np.radians(particles[:, 2])

particle_dx = particle_vector_length * np.sin(particle_theta_rad)
particle_dy = particle_vector_length * np.cos(particle_theta_rad)

# setup and define the distance sensors (top, bottom, left, right) with their positions and directions relative to the robot

distance_sensor_available = ["top", "bottom", "left", "right"]
distance_sensors = {
    "top": ((0, 0), (0.0, 0.0)),
    "bottom": ((0, 0), (0.0, 0.0)),
    "left": ((0, 0), (0.0, 0.0)),
    "right": ((0, 0), (0.0, 0.0)),
}

distance_sensor_distances = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}

keys = {"up": False, "down": False, "left": False, "right": False}


# Visualization setup

fig, ax = plt.subplots(figsize=(8, 8))
error_text = ax.text(
    0.02,
    0.98,
    "",
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)

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

(center_dot,) = ax.plot(
    [], [], marker="o", color="blue", markersize=6, linestyle="None", label="Actual"
)  # Center dot
(mcl_estimated_dot,) = ax.plot(
    [],
    [],
    marker="o",
    color="green",
    markersize=6,
    linestyle="None",
    label="MCL estimated",
)
(odometry_dot,) = ax.plot(
    [],
    [],
    marker="o",
    color="yellow",
    markersize=6,
    markeredgecolor="black",
    linestyle="None",
    label="Odometry",
)
(actual_heading_line,) = ax.plot(
    [], [], color="blue", linewidth=2, zorder=5, label="Actual heading"
)
(mcl_heading_line,) = ax.plot(
    [], [], color="green", linewidth=2, linestyle="--", zorder=5, label="MCL heading"
)
(odometry_heading_line,) = ax.plot(
    [],
    [],
    color="goldenrod",
    linewidth=2,
    linestyle=":",
    zorder=5,
    label="Odometry heading",
)
(particles_scatter,) = ax.plot([], [], "r.", markersize=2, alpha=0.5, zorder=2)
(top_distance_line,) = ax.plot([], [], color="green", linewidth=2, zorder=5)
(bottom_distance_line,) = ax.plot([], [], color="green", linewidth=2, zorder=5)
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

        self.imu = robot_theta + np.random.normal(
            0, theta_noise
        )  # Simulated IMU reading with noise

        self.sensors = {}

        # distance sensors(distances to wall)

        self.sensors["top"] = distance_sensor_distances["top"] + np.random.normal(
            0, distance_sensor_noise
        )
        self.sensors["bottom"] = distance_sensor_distances["bottom"] + np.random.normal(
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
        self.last_odometry_theta = self.imu + np.random.normal(0, odometry_theta_noise)
        self.odometry_x = self.last_odometry_x
        self.odometry_y = self.last_odometry_y
        self.odometry_theta = self.last_odometry_theta

    def update_odometry_and_sensor_readings(self):

        # meant to simulate the odometry readings we would get from the robot's encoders and IMU, which will be used to predict the new position of the particles before we update their weights based on the sensor readings

        self.odometry_x = robot_x + np.random.normal(0, odometry_noise)
        self.odometry_y = robot_y + np.random.normal(0, odometry_noise)
        self.odometry_theta = self.imu + np.random.normal(0, odometry_theta_noise)

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

        self.sensors["top"] = distance_sensor_distances["top"] + np.random.normal(
            0, distance_sensor_noise
        )
        self.sensors["bottom"] = distance_sensor_distances["bottom"] + np.random.normal(
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

        self.imu = robot_theta + np.random.normal(0, theta_noise)
        self.update_odometry_and_sensor_readings()

        # get the change in position and orientation from the odometry readings, which will be used to predict the new position of the particles

        delta_x = self.odometry_x - self.last_odometry_x
        delta_y = self.odometry_y - self.last_odometry_y
        delta_theta = self.odometry_theta - self.last_odometry_theta
        delta_theta = (delta_theta + 180) % 360 - 180

        self.last_odometry_x = self.odometry_x
        self.last_odometry_y = self.odometry_y
        self.last_odometry_theta = self.odometry_theta

        for i in range(NUMBER_OF_PARTICLES):

            # update the particle's position based on the odometry readings, adding some noise to simulate uncertainty

            particles[i, 0] += delta_x + np.random.normal(0, odometry_noise)
            particles[i, 1] += delta_y + np.random.normal(0, odometry_noise)
            particles[i, 2] = (
                particles[i, 2] + delta_theta + np.random.normal(0, theta_noise)
            ) % 360

            # keep the particles within the map boundaries

            particles[i, 0] = np.clip(particles[i, 0], 0, MAP_DIMENSIONS[0])
            particles[i, 1] = np.clip(particles[i, 1], 0, MAP_DIMENSIONS[1])
            particles[i, 2] %= 360

    def calculate_expected_sensor_reading(
        self, particle_x, particle_y, particle_theta, sensor_name
    ):

        # calculate the expected sensor reading for a given particle based on its position and orientation, which will be compared to the actual sensor reading to update the particle's weight

        theta_rad = math.radians(particle_theta)

        forward_x = math.sin(theta_rad)
        forward_y = math.cos(theta_rad)
        right_x = math.cos(theta_rad)
        right_y = -math.sin(theta_rad)
        left_x = -right_x
        left_y = -right_y

        half_width = ROBOT_SIZE[0] / 2
        half_length = ROBOT_SIZE[1] / 2

        if sensor_name == "top":
            sensor_x = particle_x + forward_x * half_length
            sensor_y = particle_y + forward_y * half_length
            direction_x = forward_x
            direction_y = forward_y
        elif sensor_name == "bottom":
            sensor_x = particle_x - forward_x * half_length
            sensor_y = particle_y - forward_y * half_length
            direction_x = -forward_x
            direction_y = -forward_y
        elif sensor_name == "left":
            sensor_x = particle_x + left_x * half_width
            sensor_y = particle_y + left_y * half_width
            direction_x = left_x
            direction_y = left_y
        elif sensor_name == "right":
            sensor_x = particle_x + right_x * half_width
            sensor_y = particle_y + right_y * half_width
            direction_x = right_x
            direction_y = right_y

        sensor_end_x, sensor_end_y, expected_distance = get_sensor_ray(
            sensor_x, sensor_y, direction_x, direction_y
        )

        return expected_distance

    def update_weights(self):

        # update the weights of the particles based on how similar their predicted sensor readings are to the actual sensor readings, which will be used to resample the particles and calculate the new predicted position of the robot

        for i in range(NUMBER_OF_PARTICLES):

            sensor_weight_results = []

            for sensor_name in distance_sensor_available:

                expected_distance = self.sensors[sensor_name]
                particle_x, particle_y, particle_theta, weight = particles[i]

                # calculate the expected sensor reading for this particle based on its position and orientation, which will be compared to the actual sensor reading to update the particle's weight

                predicted_distance = self.calculate_expected_sensor_reading(
                    particle_x, particle_y, particle_theta, sensor_name
                )

                # calculate the weight contribution from this sensor using a "Gaussian probability density function"(copy pasted this forumala, i cant lie), which will be higher if the predicted distance is close to the actual distance and lower if it is far away, taking into account the noise of the sensor

                sensor_weight_results.append(
                    np.exp(
                        -0.5
                        * (
                            (predicted_distance - expected_distance)
                            / distance_sensor_noise
                        )
                        ** 2
                    )
                )

            particles[i, 3] = np.prod(sensor_weight_results)

    def resample_particles(self):

        # resample the particles based on their weights, which will create a new set of particles that are more likely to be close to the robot's actual position, and then calculate the new predicted position of the robot based on the distribution of the resampled particles

        # built the "pie chart" where each particle's weight determines the size of its slice, and then randomly sample from this distribution to create a new set of particles

        weights = particles[:, 3]

        # normalize weights?

        weights += 1e-300
        weights /= np.sum(weights)

        # spin wheel

        indices = np.random.choice(
            range(NUMBER_OF_PARTICLES), size=NUMBER_OF_PARTICLES, p=weights
        )

        new_generation = particles[indices]

        # reset weights to uniform after resampling, since we have a new generation of particles and we will update their weights based on the sensor readings in the next iteration
        # replaces old particles with the new resampled particles, and resets their weights to uniform since we will update them based on the sensor readings in the next iteration

        particles[:, :3] = new_generation[:, :3]
        particles[:, 3] = 1.0 / NUMBER_OF_PARTICLES

    def update(self):

        self.predict()

        self.update_weights()

        self.resample_particles()

    def get_estimated_pose(self):

        # calculate the average position and orientation of the particles, which will be used as the predicted position of the robot, since after resampling we expect most of the particles to be clustered around the robot's actual position

        avg_x = np.mean(particles[:, 0])
        avg_y = np.mean(particles[:, 1])

        thetas_rad = np.radians(particles[:, 2])
        avg_sin = np.mean(np.sin(thetas_rad))
        avg_cos = np.mean(np.cos(thetas_rad))

        avg_theta_rad = math.atan2(avg_sin, avg_cos)
        avg_theta = math.degrees(avg_theta_rad) % 360

        return avg_x, avg_y, avg_theta


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
        "top": {
            "origin": (
                robot_x + forward_x * half_length,
                robot_y + forward_y * half_length,
            ),
            "direction": (forward_x, forward_y),
            "line": top_distance_line,
        },
        "bottom": {
            "origin": (
                robot_x - forward_x * half_length,
                robot_y - forward_y * half_length,
            ),
            "direction": (-forward_x, -forward_y),
            "line": bottom_distance_line,
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


def angle_error_deg(reference_theta, compared_theta):
    return abs((compared_theta - reference_theta + 180) % 360 - 180)


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
    ax.legend(loc="upper right")

    particles_scatter.set_data(particles[:, 0], particles[:, 1])
    particle_theta_rad = np.radians(particles[:, 2])
    particle_dx = particle_vector_length * np.sin(particle_theta_rad)
    particle_dy = particle_vector_length * np.cos(particle_theta_rad)
    particles_heading_vectors.set_offsets(particles[:, :2])
    particles_heading_vectors.set_UVC(particle_dx, particle_dy)

    estimated_x, estimated_y, estimated_theta = mcl_instance.get_estimated_pose()
    center_dot.set_data([robot_x], [robot_y])
    mcl_estimated_dot.set_data([estimated_x], [estimated_y])
    odometry_dot.set_data([mcl_instance.odometry_x], [mcl_instance.odometry_y])

    arrow_length = 15.0
    robot_theta_rad = math.radians(robot_theta)
    estimated_theta_rad = math.radians(estimated_theta)
    odometry_theta_rad = math.radians(mcl_instance.odometry_theta)

    actual_heading_line.set_data(
        [robot_x, robot_x + arrow_length * math.sin(robot_theta_rad)],
        [robot_y, robot_y + arrow_length * math.cos(robot_theta_rad)],
    )
    mcl_heading_line.set_data(
        [estimated_x, estimated_x + arrow_length * math.sin(estimated_theta_rad)],
        [estimated_y, estimated_y + arrow_length * math.cos(estimated_theta_rad)],
    )
    odometry_heading_line.set_data(
        [
            mcl_instance.odometry_x,
            mcl_instance.odometry_x + arrow_length * math.sin(odometry_theta_rad),
        ],
        [
            mcl_instance.odometry_y,
            mcl_instance.odometry_y + arrow_length * math.cos(odometry_theta_rad),
        ],
    )

    mcl_pos_error = math.hypot(robot_x - estimated_x, robot_y - estimated_y)
    odom_pos_error = math.hypot(
        robot_x - mcl_instance.odometry_x, robot_y - mcl_instance.odometry_y
    )
    mcl_heading_error = angle_error_deg(robot_theta, estimated_theta)
    odom_heading_error = angle_error_deg(robot_theta, mcl_instance.odometry_theta)

    error_text.set_text(
        f"MCL pos err: {mcl_pos_error:.2f} in | MCL heading err: {mcl_heading_error:.1f}°\n"
        f"Odom pos err: {odom_pos_error:.2f} in | Odom heading err: {odom_heading_error:.1f}°"
    )

    ax.set_title(f"VEX Virtual Field (Theta: {robot_theta:.1f}°)")

    return (
        field_shape,
        robot_shape,
        error_text,
        center_dot,
        mcl_estimated_dot,
        odometry_dot,
        actual_heading_line,
        mcl_heading_line,
        odometry_heading_line,
        particles_scatter,
        particles_heading_vectors,
        top_distance_line,
        bottom_distance_line,
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

    moved_or_turned = False

    if keys["left"]:
        moved_or_turned = True
        robot_theta = (robot_theta - 3) % 360
    if keys["right"]:

        moved_or_turned = True
        robot_theta = (robot_theta + 3) % 360

    theta_rad = math.radians(robot_theta)

    if keys["up"]:
        moved_or_turned = True
        robot_x += 1.5 * math.sin(theta_rad)
        robot_y += 1.5 * math.cos(theta_rad)

    if keys["down"]:
        moved_or_turned = True
        robot_x -= 1.5 * math.sin(theta_rad)
        robot_y -= 1.5 * math.cos(theta_rad)

    keep_robot_in_bounds(theta_rad)

    update_distance_sensors(theta_rad)

    if moved_or_turned:

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

    estimated_x, estimated_y, estimated_theta = mcl_instance.get_estimated_pose()

    center_dot.set_data([robot_x], [robot_y])
    mcl_estimated_dot.set_data([estimated_x], [estimated_y])
    odometry_dot.set_data([mcl_instance.odometry_x], [mcl_instance.odometry_y])

    arrow_length = 15.0
    estimated_theta_rad = math.radians(estimated_theta)
    odometry_theta_rad = math.radians(mcl_instance.odometry_theta)

    actual_heading_line.set_data(
        [robot_x, robot_x + arrow_length * math.sin(theta_rad)],
        [robot_y, robot_y + arrow_length * math.cos(theta_rad)],
    )
    mcl_heading_line.set_data(
        [estimated_x, estimated_x + arrow_length * math.sin(estimated_theta_rad)],
        [estimated_y, estimated_y + arrow_length * math.cos(estimated_theta_rad)],
    )
    odometry_heading_line.set_data(
        [
            mcl_instance.odometry_x,
            mcl_instance.odometry_x + arrow_length * math.sin(odometry_theta_rad),
        ],
        [
            mcl_instance.odometry_y,
            mcl_instance.odometry_y + arrow_length * math.cos(odometry_theta_rad),
        ],
    )

    mcl_pos_error = math.hypot(robot_x - estimated_x, robot_y - estimated_y)
    odom_pos_error = math.hypot(
        robot_x - mcl_instance.odometry_x, robot_y - mcl_instance.odometry_y
    )
    mcl_heading_error = angle_error_deg(robot_theta, estimated_theta)
    odom_heading_error = angle_error_deg(robot_theta, mcl_instance.odometry_theta)

    error_text.set_text(
        f"MCL pos err: {mcl_pos_error:.2f} in | MCL heading err: {mcl_heading_error:.1f}°\n"
        f"Odom pos err: {odom_pos_error:.2f} in | Odom heading err: {odom_heading_error:.1f}°"
    )

    ax.set_title(f"VEX Virtual Field (Theta: {robot_theta:.1f}°)")

    return (
        field_shape,
        robot_shape,
        error_text,
        center_dot,
        mcl_estimated_dot,
        odometry_dot,
        actual_heading_line,
        mcl_heading_line,
        odometry_heading_line,
        particles_scatter,
        particles_heading_vectors,
        top_distance_line,
        bottom_distance_line,
        left_distance_line,
        right_distance_line,
    )


# connecting the key press and release events to their respective handlers

fig.canvas.mpl_connect("key_press_event", on_press)
fig.canvas.mpl_connect("key_release_event", on_release)

ani = animation.FuncAnimation(fig, update, init_func=init, interval=5, blit=True)

plt.show()
