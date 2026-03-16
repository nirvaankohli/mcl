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

spawn_x = np.random.normal(initial_x, 20.0, NUMBER_OF_PARTICLES)
spawn_y = np.random.normal(initial_y, 20.0, NUMBER_OF_PARTICLES)
spawn_theta = np.random.normal(initial_robot_theta, 20.0, NUMBER_OF_PARTICLES) % 360
spawn_x = np.clip(spawn_x, 0, MAP_DIMENSIONS[0])
spawn_y = np.clip(spawn_y, 0, MAP_DIMENSIONS[1])

particles = np.column_stack((spawn_x, spawn_y, spawn_theta))

particle_vector_length = 2.5
particle_theta_rad = np.radians(particles[:, 2])
particle_dx = particle_vector_length * np.sin(particle_theta_rad)
particle_dy = particle_vector_length * np.cos(particle_theta_rad)

distance_sensor_noise = 0.5
distance_sensor_available = ["up", "down", "left", "right"]
distance_sensors = {
    "up": ((0, 0), (0.0, 0.0)),
    "down": ((0, 0), (0.0, 0.0)),
    "left": ((0, 0), (0.0, 0.0)),
    "right": ((0, 0), (0.0, 0.0)),
}

distance_sensor_distances = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}

print(f"Initial particles:\n{particles}")
print(f"Initial robot position: ({robot_x}, {robot_y}), Theta: {robot_theta}°")

keys = {"up": False, "down": False, "left": False, "right": False}

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


def update_distance_sensors(theta_rad):
    forward_x = math.sin(theta_rad)
    forward_y = math.cos(theta_rad)

    right_x = math.cos(theta_rad)
    right_y = -math.sin(theta_rad)

    left_x = -right_x
    left_y = -right_y

    half_width = ROBOT_SIZE[0] / 2
    half_length = ROBOT_SIZE[1] / 2

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


def mcl():

    # Main loop for the MCL simulation
    # Lets start by defining all the sensors to our use

    imu = robot_theta + np.random.normal(0, 1.0)  # Simulated IMU reading with noise

    # distance sensors(distances to wall)

    top_distance = distance_sensor_distances["up"] + np.random.normal(
        0, distance_sensor_noise
    )
    bottom_distance = distance_sensor_distances["down"] + np.random.normal(
        0, distance_sensor_noise
    )
    left_distance = distance_sensor_distances["left"] + np.random.normal(
        0, distance_sensor_noise
    )
    right_distance = distance_sensor_distances["right"] + np.random.normal(
        0, distance_sensor_noise
    )

    last_state = (predicted_x, predicted_y, predicted_theta)


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


fig.canvas.mpl_connect("key_press_event", on_press)
fig.canvas.mpl_connect("key_release_event", on_release)

ani = animation.FuncAnimation(fig, update, init_func=init, interval=5, blit=True)

plt.show()
