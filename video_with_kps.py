#-*-coding:utf-8-*-
#-*-coding:utf-8-*-
import cv2
import numpy as np
import mv_utils.transformations as transformations
import matplotlib.pyplot as plt
import torch


def project_3d_points_to_image_plane_without_distortion(proj_matrix, points_3d, convert_back_to_euclidean=True):
    """Project 3D points to image plane not taking into account distortion
    Args:
        proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
        points_3d numpy array or torch tensor of shape (N, 3): 3D points
        convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
                                        NOTE: division by zero can be here if z = 0
    Returns:
        numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
    """
    if isinstance(proj_matrix, np.ndarray) and isinstance(points_3d, np.ndarray):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.T
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


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


def update_camera_vectors():
    global front
    front_temp = np.zeros((3,))
    front_temp[0] = np.cos(np.radians(yaw)) * np.cos(np.radians(pitch))
    front_temp[1] = np.sin(np.radians(pitch))
    front_temp[2] = np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
    front = unit_vector(front_temp)
    global right
    right = unit_vector(np.cross(front, world_up))


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
    LR = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([joints[I[i], j], joints[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=4, c=lcolor if LR[i] else rcolor)
    if plot_dot:
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='k', marker='o')

    RADIUS = 750  # space around the subject
    xroot, yroot, zroot = joints[2, 0], joints[2, 1], joints[2, 2]
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

multiview_data = np.load("resources/3D_sence_multiview.npy", allow_pickle=True).tolist()
subject_name, camera_name, action_name, camera_configs, labels = multiview_data['subject_names'], multiview_data[
    'camera_names'], multiview_data['action_names'], multiview_data['cameras'], multiview_data['table']

camera_name = [str(i) for i, c in enumerate(camera_name)]

# subject_name ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

# video_action_name = ['Directions 1', 'Directions', 'Discussion 1', 'Discussion 2', 'Eating', 'Eating 1', 'Greeting 1', 'Greeting 2', 'Phoning', 'Phoning 1', 'Posing', 'Posing 1', 'Purchases', 'Purchases 1', 'Sitting 1', 'Sitting', 'SittingDown', 'SittingDown 1', 'Smoking', 'Smoking 1', 'Photo', 'Photo 2', 'Waiting 1', 'Waiting 2', 'Walking', 'Walking 1', 'WalkDog', 'WalkDog 1', 'WalkingTogether', 'WalkingTogether 1']
label_action_name = ['Directions-1', 'Directions-2', 'Discussion-1', 'Discussion-2', 'Eating-1', 'Eating-2', 'Greeting-1', 'Greeting-2', 'Phoning-1', 'Phoning-2', 'Posing-1', 'Posing-2', 'Purchases-1', 'Purchases-2', 'Sitting-1', 'Sitting-2', 'SittingDown-1', 'SittingDown-2', 'Smoking-1', 'Smoking-2', 'TakingPhoto-1', 'TakingPhoto-2', 'Waiting-1', 'Waiting-2', 'Walking-1', 'Walking-2', 'WalkingDog-1', 'WalkingDog-2', 'WalkingTogether-1', 'WalkingTogether-2']

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (900, 900))
img_root = "/media/hkuit155/Windows/1TB_dataset/multi_view/h36m/images"
specific_subject = "S1"
specific_action = "WalkingDog-1"
mask_subject = labels['subject_idx'] == subject_name.index(specific_subject)
actions = [action_name.index(specific_action)]
# img_folder = os.path.join(img_root, "processed", specific_subject, specific_action, "imageSequence-undistorted")

mask_actions = np.isin(labels['action_idx'], actions)
mask_subject = mask_subject & mask_actions
indices = []
indices.append(np.nonzero(mask_subject)[0])
specific_label = labels[np.concatenate(indices)]
specific_3d_skeleton = specific_label['keypoints']

specific_camera_config = camera_configs[subject_name.index(specific_subject)]
specific_camera_config = [
    Camera(specific_camera_config["R"][i], specific_camera_config["t"][i], specific_camera_config["K"][i]) for i in
    range(len(camera_name))]

# first person setup
yaw = -125
pitch = -15
world_up = np.array([0.0, 1.0, 0.0])
position = np.array([5000, 2500, 7557])
front = np.array([0.0, 0.0, -1.0])
right = np.array([0.0, 0.0, 0.0])

grid_vertices, grid_color = generate_grid_mesh(-4, 4, step=1)
grid_vertices = grid_vertices.reshape(-1, 3)

rorate_x_90 = transformations.rotation_matrix(np.radians(-90), (1, 0, 0))

frame_size = 900
original_video_frame_size = 1000
frame = np.zeros([frame_size, frame_size])

for i in range(len(camera_name)):
    specific_camera_config[i].update_after_resize((original_video_frame_size,) * 2,
                                                  (frame_size,) * 2)

update_camera_vectors()
view_matrix = look_at(position, position + front, world_up)

projection_matrix = np.array([[2.41421, 0, 0, 0],
                              [0, 2.41421, 0, 0],
                              [0, 0, -1, -0.2],
                              [0, 0, -1, 0]], dtype=np.float32)

o_view_matrix = view_matrix.copy()
o_projection_matrix = projection_matrix.copy()

total_frame = specific_3d_skeleton.shape[0]
frame_index = 0

view_camera_index = 1
plt.figure()
# cam_dict = {0: "54138969",
#             -1: "55011271",
#             2: "58860488",
#             1: "55011271",
#             3: "60457274"}

actor_map = {"S{}".format(idx): "s_0{}".format(idx) for idx in [1,5,6,7,8,9]}
actor_map["S11"] = "s_11"
# camera_map = {0: "ca_01", 1: "ca_01", 2: "ca_01", 3: "ca_04"}
camera_idx = "ca_0{}".format(view_camera_index+1)
action_map = {action: "act_{}".format(str(int(i/2)+2).zfill(2)) for i, action in enumerate(label_action_name)}
subaction = "subact_01" if "1" in specific_action else "subact_02"

img_string = "{}_{}_{}_{}".format(actor_map[specific_subject], action_map[specific_action], subaction, camera_idx)
# img_folder = os.path.join(img_root, )

while True:

    if frame_index == total_frame:
        frame_index = 0
    frame = np.zeros([frame_size, frame_size, 3])

    view_matrix = o_view_matrix
    projection_matrix = o_projection_matrix

    grid_vertices_project = grid_vertices @ (np.eye(3) if view_matrix is None else rorate_x_90[:3, :3].T)
    grid_vertices_project = grid_vertices_project @ transformations.scale_matrix(650)[:3, :3].T
    grid_vertices_project = projection_to_2d_plane(grid_vertices_project, projection_matrix, view_matrix,
                                                   int(frame_size / 2)).reshape(-1, 4)

    # draw line
    for index, line in enumerate(grid_vertices_project):
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), grid_color[index].tolist())

    # draw camera
    for camera_index, conf in enumerate(specific_camera_config):
        if view_camera_index == camera_index:
            continue
        m_rt = transformations.identity_matrix()
        r = np.array(conf.R, dtype=np.float32).T
        m_rt[:-1, -1] = -r @ np.array(conf.t, dtype=np.float32).squeeze()
        m_rt[:-1, :-1] = r

        m_s = transformations.identity_matrix() * 200
        m_s[3, 3] = 1

        camera_vertices_convert = homogeneous_to_euclidean(
            euclidean_to_homogeneous(camera_vertices) @ (
                    (np.eye(4) if view_matrix is None else rorate_x_90) @ m_rt @ m_s).T)

        camera_vertices_convert = projection_to_2d_plane(camera_vertices_convert, projection_matrix, view_matrix,
                                                         int(frame_size / 2))
        camera_vertices_convert = camera_vertices_convert.reshape(-1, 4)
        for index, line in enumerate(camera_vertices_convert):
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 153, 255), thickness=1)
        cv2.putText(frame, camera_name[camera_index],
                    (camera_vertices_convert[1, 0], camera_vertices_convert[1, 1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

    specific_3d_skeleton_project = specific_3d_skeleton[frame_index].reshape(-1, 3)
    specific_3d_skeleton_project_2d = specific_3d_skeleton[frame_index].reshape(-1, 3)

    ax = plt.subplot(1, 1, 1, projection='3d')
    show3Dpose(specific_3d_skeleton_project, ax, plot_dot=True, add_labels=False)
    plt.tight_layout()
    plt.savefig("tmp.JPG")
    plt.clf()
    img3d = cv2.resize(cv2.imread("tmp.JPG"), (720, 540))
    cv2.imshow("3d", img3d)

    img_path = os.path.join(img_root, img_string, img_string+"_"+str(frame_index+1).zfill(6)+".jpg")
    origin_img = cv2.imread(img_path)

    specific_3d_skeleton_project = specific_3d_skeleton_project @ (
        np.eye(3) if view_matrix is None else rorate_x_90[:3, :3]).T
    specific_3d_skeleton_project = specific_3d_skeleton_project @ np.eye(3, dtype=np.float32) * 1
    specific_3d_skeleton_project = projection_to_2d_plane(specific_3d_skeleton_project, projection_matrix, view_matrix,
                                                          int(frame_size / 2)).reshape(17, 2)

    for c in human36m_connectivity_dict:
        cv2.line(frame, (*specific_3d_skeleton_project[c[0]],), (*specific_3d_skeleton_project[c[1]],),
                 (100, 155, 255), thickness=2)
        cv2.circle(frame, (*specific_3d_skeleton_project[c[0]],), 3, (0, 0, 255), -1)
        cv2.circle(frame, (*specific_3d_skeleton_project[c[1]],), 3, (0, 0, 255), -1)

    view_matrix_2d = None
    projection_matrix_2d = specific_camera_config[view_camera_index].projection
    specific_3d_skeleton_project_2d = specific_3d_skeleton_project_2d @ (
        np.eye(3) if view_matrix_2d is None else rorate_x_90[:3, :3]).T
    specific_3d_skeleton_project_2d = specific_3d_skeleton_project_2d @ np.eye(3, dtype=np.float32) * 1
    specific_3d_skeleton_project_2d = projection_to_2d_plane(specific_3d_skeleton_project_2d, projection_matrix_2d,
                                                             view_matrix_2d, int(frame_size / 2)).reshape(17, 2)

    for c in human36m_connectivity_dict:
        cv2.line(origin_img, (*specific_3d_skeleton_project_2d[c[0]],), (*specific_3d_skeleton_project_2d[c[1]],),
                 (100, 155, 255), thickness=2)
        cv2.circle(origin_img, (*specific_3d_skeleton_project_2d[c[0]],), 3, (0, 0, 255), -1)
        cv2.circle(origin_img, (*specific_3d_skeleton_project_2d[c[1]],), 3, (0, 0, 255), -1)

    frame_index += 1
    # print(frame_index)
    cv2.imshow(specific_action, frame)
    # cv2.imshow("projection", frame_projection)
    cv2.imshow("img_origin", origin_img)

    out.write(np.uint8(frame))
    frame = cv2.resize(frame, (720, 540))

    # cv2.imshow("img_origin", origin_img)
    # out_img = cv2.hconcat([origin_img,img3d,frame])
    # out_img = np.concatenate((origin_img, cv2.resize(cv2.imread("tmp.JPG"), (720, 540)), frame), axis=0)
    # cv2.imshow("result", out_img)
    # cv2.waitKey(1)

    # if cv2.waitKey(6) & 0xFF == ord('1'):
    #     view_camera_index += 1
    #     if view_camera_index == 4:
    #         view_camera_index = -1
    cv2.waitKey(1)

cv2.destroyAllWindows()
