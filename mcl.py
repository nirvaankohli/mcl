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

initial_x = 72.0
initial_y = 40.0
initial_robot_theta = 70.0

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


def init():
    ax.set_xlim(-PLOT_PADDING, MAP_DIMENSIONS[0] + PLOT_PADDING)
    ax.set_ylim(-PLOT_PADDING, MAP_DIMENSIONS[1] + PLOT_PADDING)
    ax.set_aspect("equal")
    ax.add_patch(field_shape)
    ax.add_patch(robot_shape)
    return field_shape, robot_shape, center_dot, heading_line


def on_press(event):
    if event.key in keys:
        keys[event.key] = True


def on_release(event):
    if event.key in keys:
        keys[event.key] = False


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

    return field_shape, robot_shape, center_dot, heading_line


fig.canvas.mpl_connect("key_press_event", on_press)
fig.canvas.mpl_connect("key_release_event", on_release)

ani = animation.FuncAnimation(fig, update, init_func=init, interval=5, blit=True)

plt.show()
