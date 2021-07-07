#-*-coding:utf-8-*-

import matplotlib.pyplot as plt
import cv2


def plot_xyz_curve(x_coord, y_coord, z_coord, curr_frame, max_draw=100, title="", save_path="tmp.jpg"):
    plt.figure()
    t = list(range(curr_frame+1))
    if curr_frame > max_draw:
        x_plot, y_plot, z_plot = x_coord[-max_draw:], y_coord[-max_draw:], z_coord[-max_draw:]
        t = t[:max_draw]
    else:
        x_plot, y_plot, z_plot = x_coord, y_coord, z_coord
    plt.plot(t, x_plot, linewidth=4, label="x")
    plt.plot(t, y_plot, linewidth=4, label="y")
    plt.plot(t, z_plot, linewidth=4, label="z")
    plt.legend(loc="upper left")

    plt.title(title, fontsize=20)
    plt.xlabel("frame", fontsize=12)
    plt.ylabel("movement(mm)", fontsize=12)
    plt.savefig(save_path)
    plt.close()

    return cv2.imread(save_path)


def plot_xyz_curves(all_kps_moves, hip_moves, curr_frame, max_draw=100, title="", save_path="tmp.jpg"):
    plt.figure()
    t = list(range(curr_frame+1))
    if curr_frame > max_draw:
        all_kps_moves, hip_moves = all_kps_moves[:,-max_draw:], hip_moves[:,-max_draw:]
        t = t[:max_draw]

    plt.plot(t, all_kps_moves[0], linewidth=2, label="x-all")
    plt.plot(t, all_kps_moves[1], linewidth=2, label="y-all")
    plt.plot(t, all_kps_moves[2], linewidth=2, label="z-all")
    plt.plot(t, hip_moves[0], linewidth=2, label="x-hip")
    plt.plot(t, hip_moves[1], linewidth=2, label="y-hip")
    plt.plot(t, hip_moves[2], linewidth=2, label="z-hip")

    plt.legend(loc="upper left")

    plt.title(title, fontsize=20)
    plt.xlabel("frame", fontsize=12)
    plt.ylabel("movement(mm)", fontsize=12)
    plt.savefig(save_path)
    plt.close()

    return cv2.imread(save_path)


