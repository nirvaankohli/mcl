import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

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

fig, ax = plt.subplots(figsize=(8, 8))


def draw_field():

    ax.clear()

    ax.set_xlim(-PLOT_PADDING, MAP_DIMENSIONS[0] + PLOT_PADDING)
    ax.set_ylim(-PLOT_PADDING, MAP_DIMENSIONS[1] + PLOT_PADDING)
    ax.set_aspect("equal")

    field_shape = patches.Rectangle(
        (0, 0),
        MAP_DIMENSIONS[0],
        MAP_DIMENSIONS[1],
        linewidth=1,
        edgecolor="black",
        facecolor="none",
    )

    ax.add_patch(field_shape)

    # displaying robot

    corner_x = robot_x - ROBOT_SIZE[0] / 2
    corner_y = robot_y - ROBOT_SIZE[1] / 2

    robot_shape = patches.Rectangle(
        (corner_x, corner_y),
        ROBOT_SIZE[0],
        ROBOT_SIZE[1],
        angle=-robot_theta,
        rotation_point="center",
        color="blue",
        alpha=0.5,
    )

    ax.add_patch(robot_shape)
    ax.plot(robot_x, robot_y, marker="o", color="blue", markersize=5)  # Center dot

    # thetha calculation w arrow

    theta_rad = math.radians(robot_theta)

    arrow_length = 15.0

    dx = arrow_length * math.sin(theta_rad)
    dy = arrow_length * math.cos(theta_rad)

    ax.arrow(
        robot_x,
        robot_y,
        dx,
        dy,
        head_width=3,
        head_length=4,
        fc="red",
        ec="red",
        linewidth=2,
        zorder=5,
    )

    ax.set_title(f"VEX Virtual Field (Theta: {robot_theta}°)")
    fig.canvas.draw()


def on_key_press(event):

    global robot_x, robot_y, robot_theta
    theta_rad = math.radians(robot_theta)

    if event.key == "left":
        robot_theta = (robot_theta - 10) % 360
    elif event.key == "right":
        robot_theta = (robot_theta + 10) % 360
    elif event.key == "up":
        robot_x += 5 * math.sin(theta_rad)
        robot_y += 5 * math.cos(theta_rad)
    elif event.key == "down":
        robot_x -= 5 * math.sin(theta_rad)
        robot_y -= 5 * math.cos(theta_rad)

    robot_theta %= 360

    draw_field()


fig.canvas.mpl_connect("key_press_event", on_key_press)

draw_field()
plt.show()
