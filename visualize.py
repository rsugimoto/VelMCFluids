import sys
import numpy as np
import matplotlib.pylab as plt
import os
import json

src_folder = sys.argv[1] + "/raw/"
dst_folder = sys.argv[1] + "/png/"
overwrite = True
draw_obstacles = True
os.makedirs(dst_folder, exist_ok=True)

float_type = np.float32
float_size = 4

with open(os.path.join(src_folder, "config.json"), "r") as f:
    config = json.load(f)

vertices = []
polylines = []
domain_size = config["domain_size"]
if type(domain_size) is int or type(domain_size) is float:
    domain_size = [domain_size, domain_size]
is_bounded_domain = config["domain_type"] == "BoundedDomain"
grid_res = config["grid_res"]
if type(grid_res) is int or type(grid_res) is float:
    grid_res = [grid_res, grid_res]
dt = float(config["dt"])
solid_velocity = (
    (float(config["solid_velocity"][0]), float(config["solid_velocity"][1]))
    if "solid_velocity" in config
    else (0.0, 0.0)
)
obstacle_scale = float(config["obstacle_scale"]) if "obstacle_scale" in config else 1.0
obstacle_shift = (
    (float(config["obstacle_shift"][0]), float(config["obstacle_shift"][1]))
    if "obstacle_shift" in config
    else (0.0, 0.0)
)
sourcesinks = []
if "velocity_sources" in config:
    sourcesinks = config["velocity_sources"]

if "obj_file" in config:
    obj_file = config["obj_file"]

    obj_file_path = os.path.abspath(os.path.join("objs", obj_file))
    with open(obj_file_path, "r") as f:
        for line in f:
            if len(line) == 0:
                continue
            if line[0] == "v":
                x, y, z = line.split(" ")[1:]
                x, y, z = float(x), float(y), float(z)
                x = obstacle_scale * x + obstacle_shift[0]
                y = obstacle_scale * y + obstacle_shift[1]
                x = (grid_res[0] / domain_size[0]) * x + (0.5 * grid_res[0] - 0.5)
                y = (grid_res[1] / domain_size[1]) * -y + (0.5 * grid_res[1] - 0.5)
                # z = (grid_res[2] / domain_size[2]) * z + (0.5 * grid_res[2] - 0.5)
                vertices.append((x, y))
            if line[0] == "l":
                polyline = line.split(" ")[1:]
                polyline = [int(x) - 1 for x in polyline]
                polylines.append(polyline)


def draw_polygons(translation, linewidth=1):
    if len(polylines) == 0:
        return

    translation = (
        (grid_res[0] / domain_size[0]) * translation[0],
        (grid_res[1] / domain_size[1]) * -translation[1],
    )
    # translation = (0, 0)
    for polyline in polylines:
        X = []
        Y = []
        for vert_idx in polyline:
            x, y = vertices[vert_idx]
            x += translation[0]
            y += translation[1]
            X.append(x)
            Y.append(y)
        # plt.fill(X, Y, facecolor="tab:blue", edgecolor=None, linewidth=0)
        # plt.fill(X, Y, facecolor="white", edgecolor="black", linewidth=1)
        plt.plot(X, Y, color="black", linewidth=linewidth)


def draw_source_sink():
    if len(sourcesinks) == 0:
        return

    for sourcesink in sourcesinks:
        x, y, strength = sourcesink
        x = (grid_res[0] / domain_size[0]) * x + (0.5 * grid_res[0] - 0.5)
        y = (grid_res[1] / domain_size[1]) * -y + (0.5 * grid_res[1] - 0.5)
        plt.plot(x, y, color="crimson", marker="o")


def save_scalar_field(
    input_file_path,
    output_file_path,
    is_optional,
    draw_obstacles,
    translation,
    v_min=0.0,
    v_max=1.0,
):
    if overwrite or not os.path.exists(input_file_path):
        try:
            with open(input_file_path, "rb") as file:
                size_x = int.from_bytes(file.read(4), "little")
                size_y = int.from_bytes(file.read(4), "little")
                # size_z = int.from_bytes(file.read(4), "little")
                buf = file.read(size_x * size_y * float_size)
                ten = np.frombuffer(buf, dtype=float_type)
                ten = ten.reshape(size_y, size_x)  # in the order of y x
                mat = ten[::-1]  # flip y
                plt.close()
                fig = plt.figure(figsize=(size_x / 100.0, size_y / 100.0), dpi=100.0)
                ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
                ax.set_xlim(0, size_x - 1)
                ax.set_ylim(size_y - 1, 0)
                ax.set_axis_off()
                fig.add_axes(ax)
                plt.imshow(
                    mat,
                    cmap="Greys" if v_min >= 0.0 else "bwr",
                    vmin=v_min,
                    vmax=v_max,
                )
                if draw_obstacles:
                    draw_polygons(translation)
                draw_source_sink()
                plt.savefig(output_file_path, dpi=100.0)
        except FileNotFoundError:
            if not is_optional:
                exit(0)


def save_vector_field(
    input_file_path, output_file_path, is_optional, scale, draw_obstacles, translation, winding_number_file_path=None
):
    mask = None
    if winding_number_file_path is not None and os.path.exists( winding_number_file_path):
        try:
            with open( winding_number_file_path, "rb") as file:
                size_x = int.from_bytes(file.read(4), "little")
                size_y = int.from_bytes(file.read(4), "little")
                # size_z = int.from_bytes(file.read(4), "little")
                buf = file.read(size_x * size_y * float_size)
                ten = np.frombuffer(buf, dtype=float_type)
                ten = ten.reshape(size_y, size_x)  # in the order of y x
                mask = ten[::-1]  # flip y
                if is_bounded_domain:
                    mask = (-0.5 < mask) & (mask < 0.5)
                else:
                    mask = mask < -0.5
        except FileNotFoundError:
            mask = None

    if overwrite or not os.path.exists(output_file_path):
        try:
            with open(input_file_path, "rb") as file:
                size_x = int.from_bytes(file.read(4), "little")
                size_y = int.from_bytes(file.read(4), "little")
                # size_z = int.from_bytes(file.read(4), "little")
                buf = file.read(size_x * size_y * 2 * float_size)
                ten = np.frombuffer(buf, dtype=float_type)
                ten = ten.reshape(size_y, size_x, 2)  # in the order of y x
                mat = ten[::-1]  # flip y

                plt.close()
                fig = plt.figure(
                    figsize=(4 * size_x / 100.0, 4 * size_y / 100.0), dpi=100.0
                )
                ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
                ax.set_xlim(0, size_x - 1)
                ax.set_ylim(size_y - 1, 0)
                ax.set_axis_off()
                fig.add_axes(ax)

                U = mat[:, :, 0]  # x
                V = mat[:, :, 1]  # y
                speed = np.sqrt(U**2 + V**2)
                if scale == None:
                    scale = speed.max()
                UN = U / scale
                VN = V / scale

                if mask is not None:
                    speed[mask] = np.nan
                    UN[mask] = np.nan
                    VN[mask] = np.nan
                scalar = plt.imshow(speed, cmap="Reds", vmin=0.0, vmax=scale)
                frequency = 4
                Y, X = np.mgrid[
                    frequency : size_y : 2 * frequency,
                    frequency : size_x : 2 * frequency,
                ]
                vector = plt.quiver(
                    X,
                    Y,
                    UN[frequency :: 2 * frequency, frequency :: 2 * frequency],
                    VN[frequency :: 2 * frequency, frequency :: 2 * frequency],
                    color="blue",
                    scale=10,
                )
                if draw_obstacles:
                    draw_polygons(translation, 4)
                plt.savefig(output_file_path, dpi=100.0)
        except FileNotFoundError:
            if not is_optional:
                exit(0)
    if scale == 0.0:
        return None
    return scale


velocity_scale = 0.88 #None

i = -1
while True:
    i += 1
    if i > 0 and velocity_scale == 0.0:
        velocity_scale = 0.2

    translation = (solid_velocity[0] * dt * i, solid_velocity[1] * dt * i)

    save_scalar_field(
        src_folder + f"winding_number_{i}.scalar",
        dst_folder + f"winding_number_{i}.png",
        True,
        False,
        translation,
        -1.0,
        1.0,
    )

    # save_scalar_field(
    #     src_folder + f"concentration_prediffuse_{i}.scalar",
    #     dst_folder + f"concentration_prediffuse_{i}.png",
    #     True,
    #     draw_obstacles,
    #     translation,
    # )

    # save_scalar_field(
    #     src_folder + f"concentration_prediffuse_{i}.scalar",
    #     dst_folder + f"concentration_prediffuse_{i}.png",
    #     True,
    #     draw_obstacles,
    #     translation,
    # )

    save_scalar_field(
        src_folder + f"concentration_{i}.scalar",
        dst_folder + f"concentration_{i}.png",
        False,
        draw_obstacles,
        translation,
        -0.5,
        0.5,
    )

    # save_scalar_field(
    #     src_folder + f"temperature_prediffuse_{i}.scalar",
    #     dst_folder + f"temperature_prediffuse_{i}.png",
    #     True,
    #     draw_obstacles,
    #     translation,
    # )

    # save_scalar_field(
    #     src_folder + f"temperature_{i}.scalar",
    #     dst_folder + f"temperature_{i}.png",
    #     True,
    #     draw_obstacles,
    #     translation,
    # )

    # velocity_scale = save_vector_field(
    #     src_folder + f"velocity_prefirstproject_{i}.vector",
    #     dst_folder + f"velocity_prefirstproject_{i}.png",
    #     True,
    #     velocity_scale,
    #     draw_obstacles,
    #     translation,
    # )

    # velocity_scale = save_vector_field(
    #     src_folder + f"velocity_prediffuse_{i}.vector",
    #     dst_folder + f"velocity_prediffuse_{i}.png",
    #     True,
    #     velocity_scale,
    #     draw_obstacles,
    #     translation,
    # )

    # velocity_scale = save_vector_field(
    #     src_folder + f"velocity_preproject_{i}.vector",
    #     dst_folder + f"velocity_preproject_{i}.png",
    #     True,
    #     velocity_scale,
    #     draw_obstacles,
    #     translation,
    # )

    velocity_scale = save_vector_field(
        src_folder + f"velocity_{i}.vector",
        dst_folder + f"velocity_{i}.png",
        False,
        velocity_scale,
        draw_obstacles,
        translation,
        src_folder + f"winding_number_0.scalar",
    )

    if i==0:
        print("velocity scale:", velocity_scale)
    print("frame:", i)
