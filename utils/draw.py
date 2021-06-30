#-*-coding:utf-8-*-

import matplotlib.pyplot as plt
import cv2


def plot_xyz_curve(x_coord, y_coord, z_coord, curr_frame, max_draw=30, title="", save_path="tmp.jpg"):
    t = list(range(curr_frame+1))
    if curr_frame > max_draw:
        x_plot, y_plot, z_plot = x_coord[-max_draw:], y_coord[-max_draw:], z_coord[-max_draw:]
        t = t[:max_draw]
    else:
        x_plot, y_plot, z_plot = x_coord, y_coord, z_coord
    plt.plot(t, x_plot, linewidth=4)
    plt.plot(t, y_plot, linewidth=4)
    plt.plot(t, z_plot, linewidth=4)

    plt.title(title, fontsize=20)
    plt.xlabel("frame", fontsize=12)
    plt.ylabel("movement(mm)", fontsize=12)
    plt.savefig(save_path)
    plt.clf()

    return cv2.imread(save_path)



