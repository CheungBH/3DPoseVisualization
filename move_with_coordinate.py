#-*-coding:utf-8-*-
import cv2
import numpy as np
import mv_utils.transformations as transformations
import matplotlib.pyplot as plt


class Camera:
    def __init__(self, R, t, K, dist=None, name=""):
        self.R = R.copy()
        self.t = t.copy()
        self.K = K.copy()
        self.dist = dist

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_width, new_height = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])


def generate_grid_mesh(start, end, step=1.0):
    num_point_per_line = int((end - start) // step + 1)
    its = np.linspace(start, end, num_point_per_line)
    line = []
    color = []
    common_line_color = [192, 192, 192]
    for i in range(num_point_per_line):
        line.append([its[0], its[i], 0, its[-1], its[i], 0])
        if its[i] == 0:
            color.append([0, 255, 0])
        else:
            color.append(common_line_color)

    for i in range(num_point_per_line):
        line.append([its[i], its[-1], 0, its[i], its[0], 0])
        if its[i] == 0:
            color.append([0, 0, 255])
        else:
            color.append(common_line_color)

    return np.array(line, dtype=np.float32), np.array(color, dtype=np.uint8)


def euclidean_to_homogeneous(points):
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    else:
        raise TypeError("Works only with numpy arrays")


def homogeneous_to_euclidean(points):
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    else:
        raise TypeError("Works only with numpy arrays")


def projection_to_2d_plane(vertices, projection_matrix, view_matrix=None, scale=None):
    if view_matrix is not None:
        vertices = (homogeneous_to_euclidean(
            (euclidean_to_homogeneous(vertices) @ view_matrix.T) @ projection_matrix.T)[:, :2]) * scale

        vertices[:, 1] = scale - vertices[:, 1]
        vertices[:, 0] = vertices[:, 0] + scale
    else:
        vertices = euclidean_to_homogeneous(vertices) @ projection_matrix.T
        vertices = homogeneous_to_euclidean(vertices)
    return vertices.astype(np.int32)


def look_at(eye, center, up):
    f = unit_vector(center - eye)
    u = unit_vector(up)
    s = unit_vector(np.cross(f, u))
    u = np.cross(s, f)

    result = transformations.identity_matrix()
    result[:3, 0] = s
    result[:3, 1] = u
    result[:3, 2] = -f
    result[3, 0] = -np.dot(s, eye)
    result[3, 1] = -np.dot(u, eye)
    result[3, 2] = np.dot(f, eye)
    return result.T


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def show3Dpose(joints,
               ax,
               lcolor="#3498db",
               rcolor="#e74c3c",
               add_labels=True,
               gt=False,
               pred=False,
               plot_dot=False
               ):  # blue, orange
    """
    Visualize a 3d skeleton

    Args
      channels: 96x1 vector. The pose to plot.
      ax: matplotlib 3d axis to draw on
      lcolor: color for left part of the body
      rcolor: color for right part of the body
      add_labels: whether to add coordinate labels
    Returns
      Nothing. Draws on ax.
    """

    I = np.array([0, 1, 2, 5, 4, 3, 6, 7, 8, 9, 8, 11, 10, 8, 13, 14]) # start points
    J = np.array([1, 2, 6, 4, 3, 6, 7, 8, 16, 16, 12, 12, 11, 13, 14, 15])  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([joints[I[i], j], joints[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=4, c=lcolor if LR[i] else rcolor)
    if plot_dot:
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='k', marker='o')

    RADIUS = 750  # space around the subject
    xroot, yroot, zroot = joints[8, 0], joints[8, 1], joints[8, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    ax.set_aspect('auto')
    #  ax.set_xticks([])
    #  ax.set_yticks([])
    #  ax.set_zticks([])

    #  ax.get_xaxis().set_ticklabels([])
    #  ax.get_yaxis().set_ticklabels([])
    #  ax.set_zticklabels([])
    # Get rid of the panes (actually, make them white)
    #  white = (1.0, 1.0, 1.0, 0.0)
    #  ax.w_xaxis.set_pane_color(white)
    #  ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    #  ax.w_xaxis.line.set_color(white)
    #  ax.w_yaxis.line.set_color(white)
    #  ax.w_zaxis.line.set_color(white)
    ax.view_init(10, -60)

import os

camera_vertices = np.array([[0, 0, 0], [-1, -1, 2],
                            [0, 0, 0], [1, 1, 2],
                            [0, 0, 0], [1, -1, 2],
                            [0, 0, 0], [-1, 1, 2],
                            [-1, 1, 2], [-1, -1, 2],
                            [-1, -1, 2], [1, -1, 2],
                            [1, -1, 2], [1, 1, 2],
                            [1, 1, 2], [-1, 1, 2]], dtype=np.float32)

human36m_connectivity_dict = [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 16), (9, 16), (8, 12),
                              (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)]

multiview_data = np.load("mv_utils/3D_sence_multiview.npy", allow_pickle=True).tolist()
subject_name, camera_name, action_name, camera_configs, labels = multiview_data['subject_names'], multiview_data[
    'camera_names'], multiview_data['action_names'], multiview_data['cameras'], multiview_data['table']

camera_name = [str(i) for i, c in enumerate(camera_name)]

# subject_name ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
# action_name ['Directions-1', 'Directions-2', 'Discussion-1', 'Discussion-2', 'Eating-1', 'Eating-2', 'Greeting-1', 'Greeting-2', 'Phoning-1', 'Phoning-2', 'Posing-1', 'Posing-2', 'Purchases-1', 'Purchases-2', 'Sitting-1', 'Sitting-2', 'SittingDown-1', 'SittingDown-2', 'Smoking-1', 'Smoking-2', 'TakingPhoto-1', 'TakingPhoto-2', 'Waiting-1', 'Waiting-2', 'Walking-1', 'Walking-2', 'WalkingDog-1', 'WalkingDog-2', 'WalkingTogether-1', 'WalkingTogether-2']

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (900, 900))
img_root = "/media/hkuit155/Windows/1TB_dataset/learnable_triangle_dataset"
specific_subject = "S6"
specific_action = "Directions-1"
mask_subject = labels['subject_idx'] == subject_name.index(specific_subject)
actions = [action_name.index(specific_action)]
img_folder = os.path.join(img_root, "processed", specific_subject, specific_action, "imageSequence-undistorted")

mask_actions = np.isin(labels['action_idx'], actions)
mask_subject = mask_subject & mask_actions
indices = []
indices.append(np.nonzero(mask_subject)[0])
specific_label = labels[np.concatenate(indices)]
specific_3d_skeleton = specific_label['keypoints']

movements = []
for i in range(len(specific_3d_skeleton)-1):
    movements.append(specific_3d_skeleton[i+1]-specific_3d_skeleton[i])

a = 1
