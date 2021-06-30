#-*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

root = "resources/constraints"
static_pose_path = os.path.join(root, "staticPose.npy")
static_pose = np.load(static_pose_path, allow_pickle=True).item()
# indices used for computing bone vectors for non-torso bones
nt_parent_indices = [8, 11, 12, 14, 15, 4, 5, 1, 2]
nt_child_indices = [10, 12, 13, 15, 16, 5, 6, 2, 3]
di_indices = {2:5, 4:2, 6:13, 8:9}
di = static_pose['di']
a = static_pose['a'].reshape(3)
bone_indices = {0: [5, 6, 7, 8],
                1: [7, 8],
                2: [8],
                6: [5, 6],
                7: [6],
                13: [1, 2, 3, 4], # thorax
                17: [1, 2],
                18: [2],
                25: [3, 4],
                26: [4]
                }
# used roots for random selection
root_joints = [0, 1, 2, 6, 7, 13, 17, 18, 25, 26]
template = np.load(os.path.join(root, 'template.npy'), allow_pickle=True).reshape(32,-1)
template_bones = template[nt_parent_indices, :] - template[nt_child_indices, :]

def to_spherical(xyz):
    """
    Convert from Cartisian coordinate to spherical coordinate
    theta: [-pi, pi]
    phi: [-pi/2, pi/2]
    note that xyz should be float number
    """
    # return in r, phi, and theta (elevation angle from z axis down)
    return_value = np.zeros(xyz.shape, dtype=xyz.dtype)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    return_value[:,0] = np.sqrt(xy + xyz[:,2]**2) # r
    return_value[:,1] = np.arctan2(xyz[:,1], xyz[:,0]) # theta
    return_value[:,2] = np.arctan2(xyz[:,2], np.sqrt(xy)) # phi
    return return_value

template_bone_lengths = to_spherical(template_bones)[:, 0]


def normalize(vector):
    """
    Normalize a vector.
    """
    return vector/np.linalg.norm(vector)


def get_basis1(skeleton):
    """
    Compute local coordinate system from 3D joint positions.
    This system is used for upper-limbs.
    """
    # compute the vector from the left shoulder to the right shoulder
    left_shoulder = skeleton[11]
    right_shoulder = skeleton[14]
    v1 = normalize(right_shoulder - left_shoulder)
    # compute the backbone vector from the thorax to the spine
    thorax = skeleton[8]
    spine = skeleton[7]
    v2 = normalize(spine - thorax)
    # v3 is the cross product of v1 and v2 (front-facing vector for upper-body)
    v3 = normalize(np.cross(v1, v2))
    return v1, v2, v3

def gram_schmidt_columns(X):
    """
    Apply Gram-Schmidt orthogonalization to obtain basis vectors.
    """
    B = np.zeros(X.shape)
    B[:, 0] = (1/np.linalg.norm(X[:, 0]))*X[:, 0]
    for i in range(1, 3):
        v = X[:, i]
        U = B[:, 0:i] # subspace basis which has already been orthonormalized
        pc = U.T @ v # orthogonal projection coefficients of v onto U
        p = U@pc
        v = v - p
        if np.linalg.norm(v) < 2e-16:
            # vectors are not linearly independent!
            raise ValueError
        else:
            v = normalize(v)
            B[:, i] = v
    return B

def get_normal(x1, a, x):
    """
    Get normal vector.
    """
    nth = 1e-4
    # x and a are parallel
    if np.linalg.norm(x - a) < nth or np.linalg.norm(x + a) < nth:
        n = np.cross(x, x1)
        flag = True
    else:
        n = np.cross(a, x)
        flag = False
    return normalize(n), flag


def to_local(skeleton):
    """
    Represent the bone vectors in the local coordinate systems.
    """
    v1, v2, v3 = get_basis1(skeleton)
    # compute the vector from the left hip to the right hip
    left_hip = skeleton[4]
    right_hip = skeleton[1]
    v4 = normalize(right_hip - left_hip)
    # v5 is the cross product of v4 and v2 (front-facing vector for lower-body)
    v5 = normalize(np.cross(v4, v2))
    # compute orthogonal coordinate systems using GramSchmidt
    # for upper body, we use v1, v2 and v3
    system1 = gram_schmidt_columns(np.hstack([v1.reshape(3,1),
                                              v2.reshape(3,1),
                                              v3.reshape(3,1)]))
    # make sure the directions rougly align
    #system1 = direction_check(system1, v1, v2, v3)
    # for lower body, we use v4, v2 and v5
    system2 = gram_schmidt_columns(np.hstack([v4.reshape(3,1),
                                              v2.reshape(3,1),
                                              v5.reshape(3,1)]))
    #system2 = direction_check(system2, v4, v2, v5)

    bones = skeleton[nt_parent_indices, :] - skeleton[nt_child_indices, :]
    # convert bone vector to local coordinate system
    bones_local = np.zeros(bones.shape, dtype=bones.dtype)
    for bone_idx in range(len(bones)):
        # only compute bone vectors for non-torsos
        # the order of the non-torso bone vector is:
        # bone vector1: thorax to head top
        # bone vector2: left shoulder to left elbow
        # bone vector3: left elbow to left wrist
        # bone vector4: right shoulder to right elbow
        # bone vector5: right elbow to right wrist
        # bone vector6: left hip to left knee
        # bone vector7: left knee to left ankle
        # bone vector8: right hip to right knee
        # bone vector9: right knee to right ankle
        bone = normalize(bones[bone_idx])
        if bone_idx in [0, 1, 3, 5, 7]:
            # bones that are directly connected to the torso
            if bone_idx in [0, 1, 3]:
                # upper body
                bones_local[bone_idx] = system1.T @ bone
            else:
                # lower body
                bones_local[bone_idx] = system2.T @ bone
        else:
            if bone_idx in [2, 4]:
                parent_R = system1
            else:
                parent_R = system2
            # parent bone index is smaller than 1
            vector_u = normalize(bones[bone_idx - 1])
            di_index = di_indices[bone_idx]
            vector_v, flag = get_normal(parent_R@di[:, di_index],
                                        parent_R@a,
                                        vector_u
                                        )
            vector_w = np.cross(vector_u, vector_v)
            local_system = gram_schmidt_columns(np.hstack([vector_u.reshape(3,1),
                                                           vector_v.reshape(3,1),
                                                           vector_w.reshape(3,1)]
                                                          )
                                                )
            bones_local[bone_idx] = local_system.T @ bone
    return bones_local


def to_spherical(xyz):
    """
    Convert from Cartisian coordinate to spherical coordinate
    theta: [-pi, pi]
    phi: [-pi/2, pi/2]
    note that xyz should be float number
    """
    # return in r, phi, and theta (elevation angle from z axis down)
    return_value = np.zeros(xyz.shape, dtype=xyz.dtype)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    return_value[:,0] = np.sqrt(xy + xyz[:,2]**2) # r
    return_value[:,1] = np.arctan2(xyz[:,1], xyz[:,0]) # theta
    return_value[:,2] = np.arctan2(xyz[:,2], np.sqrt(xy)) # phi
    return return_value


def get_bone_length(skeleton):
    """
    Compute limb length for a given skeleton.
    """
    bones = skeleton[nt_parent_indices, :] - skeleton[nt_child_indices, :]
    bone_lengths = to_spherical(bones)[:, 0]
    return bone_lengths


def re_order(skeleton):
    # the ordering of coordinate used by the Prior was x,z and y
    return skeleton[:, [0,2,1]]


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

    I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]) # start points
    J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])  # end points
    LR = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([joints[I[i], j], joints[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=4, c=lcolor if LR[i] else rcolor)
    if plot_dot:
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='k', marker='o')

    RADIUS = 750  # space around the subject
    xroot, yroot, zroot = joints[0, 0], joints[0, 1], joints[0, 2]
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


def rotate_bone(bone, angles):
    r = R.from_euler('xyz', angles, degrees=True)
    bone_rot = r.as_dcm() @ bone.reshape(3,1)
    return bone_rot.reshape(3)


def rotate_bone_random(bone, sigma=10.):
    angle = np.random.normal(scale=sigma)
    axis_idx = np.random.choice(3, 1)
    angle = sigma
    if axis_idx == 0:
        r = R.from_euler('xyz', [angle, 0., 0.], degrees=True)
    elif axis_idx == 1:
        r = R.from_euler('xyz', [0., angle, 0.], degrees=True)
    else:
        r = R.from_euler('xyz', [0., 0., angle], degrees=True)
    bone_rot = r.as_dcm() @ bone.reshape(3,1)
    return bone_rot.reshape(3)


def swap_bones(bones_father, bones_mother, root_idx):
    swap_indices = bone_indices[root_idx]
    temp = bones_father.copy()
    bones_father[swap_indices] = bones_mother[swap_indices].copy()
    bones_mother[swap_indices] = temp[swap_indices].copy()
    del temp
    return bones_father, bones_mother, swap_indices


def get_skeleton(bones, pose, bone_length=template_bone_lengths):
    """
    Update the non-torso limb of a skeleton by specifying bone vectors.
    """
    new_pose = pose.copy()
    for bone_idx in [0,1,3,5,7,2,4,6,8]:
        new_pose[nt_child_indices[bone_idx]] = new_pose[nt_parent_indices[bone_idx]] \
        - bones[bone_idx]*bone_length[bone_idx]
    return new_pose


def modify_pose(skeleton, local_bones, bone_length, ro=False):
    # get a new pose by modify an existing pose with input local bone vectors
    # and bone lengths
    new_bones = to_global(skeleton, local_bones)['bg']
    new_pose = get_skeleton(new_bones, skeleton, bone_length=bone_length)
    if ro:
        new_pose = re_order(new_pose)
    return new_pose.reshape(-1)


def to_global(skeleton, bones_local, cache=False):
    """
    Convert local coordinate back into global coordinate system.
    cache: return intermeadiate results
    """
    return_value = {}
    v1, v2, v3 = get_basis1(skeleton)
    # compute the vector from the left hip to the right hip
    left_hip = skeleton[4]
    right_hip = skeleton[1]
    v4 = normalize(right_hip - left_hip)
    # v5 is the cross product of v4 and v2 (front-facing vector for lower-body)
    v5 = normalize(np.cross(v4, v2))
    # compute orthogonal coordinate systems using GramSchmidt
    # for upper body, we use v1, v2 and v3
    system1 = gram_schmidt_columns(np.hstack([v1.reshape(3, 1),
                                              v2.reshape(3, 1),
                                              v3.reshape(3, 1)]))
    # make sure the directions rougly align
    # system1 = direction_check(system1, v1, v2, v3)
    # for lower body, we use v4, v2 and v5
    system2 = gram_schmidt_columns(np.hstack([v4.reshape(3, 1),
                                              v2.reshape(3, 1),
                                              v5.reshape(3, 1)]))
    # system2 = direction_check(system2, v4, v2, v5)
    if cache:
        return_value['cache'] = [system1, system2]
        return_value['bl'] = bones_local

    bones_global = np.zeros(bones_local.shape)
    # convert bone vector to local coordinate system
    for bone_idx in [0, 1, 3, 5, 7, 2, 4, 6, 8]:
        # the indices follow the order from torso to limbs
        # only compute bone vectors for non-torsos
        bone = normalize(bones_local[bone_idx])
        if bone_idx in [0, 1, 3, 5, 7]:
            # bones that are directly connected to the torso
            if bone_idx in [0, 1, 3]:
                # upper body
                # this is the inverse transformation compared to the to_local
                # function
                bones_global[bone_idx] = system1 @ bone
            else:
                # lower body
                bones_global[bone_idx] = system2 @ bone
        else:
            if bone_idx in [2, 4]:
                parent_R = system1
            else:
                parent_R = system2
            # parent bone index is smaller than 1
            vector_u = normalize(bones_global[bone_idx - 1])
            di_index = di_indices[bone_idx]
            vector_v, flag = get_normal(parent_R @ di[:, di_index],
                                        parent_R @ a,
                                        vector_u)
            vector_w = np.cross(vector_u, vector_v)
            local_system = gram_schmidt_columns(np.hstack([vector_u.reshape(3, 1),
                                                           vector_v.reshape(3, 1),
                                                           vector_w.reshape(3, 1)]))
            if cache:
                return_value['cache'].append(local_system)
            bones_global[bone_idx] = local_system @ bone
    return_value['bg'] = bones_global
    return return_value


def set_z(pose, target):
    if pose is None:
        return None
    original_shape = pose.shape
    pose = pose.reshape(17, 3)
    min_val = pose[:, 2].min()
    pose[:, 2] -= min_val - target
    return pose.reshape(original_shape)


total_joints_num = 17


def exploration(father, angles_dict):
    """
    Produce novel data by exploring the data space with evolutionary operators.
    cross over operator in the local coordinate system
    mutation: perturb the local joint angle
    """
    # get local coordinate for each bone vector

    father = re_order(father.reshape(total_joints_num, -1))
    father_bone_length = get_bone_length(father)
    bones_father = to_local(father)

    # indices = bone_indices[root_joints[list(angles_dict.keys())[0]]]
    # values = list[angle_dicts.values()][0]

    # local mutation: apply random rotation to local limb
    for k, v in angles_dict.items():
        indices = bone_indices[root_joints[k]]
        for bone_idx in indices:
            bones_father[bone_idx] = rotate_bone(bones_father[bone_idx], v)

    modified_pose = modify_pose(father, bones_father, father_bone_length, ro=True)
    set_z(modified_pose, np.random.normal(loc=20.0, scale=3.0))

    ax1 = plt.subplot(2, 1, 1, projection='3d')
    # plt.title('father')
    show3Dpose(re_order(father), ax1, add_labels=False, plot_dot=True)
    # plt.title('mother')
    ax3 = plt.subplot(2, 1, 2, projection='3d')
    # plt.title('son: ' + str(valid_vec_fa.sum()))
    show3Dpose(modified_pose.reshape(total_joints_num, -1), ax3, add_labels=False, plot_dot=True)

    plt.tight_layout()
    plt.savefig("./EXAMPLE17.JPG")
    plt.clf()
    return modified_pose


def get_useful_idx(ls):
    useful = []
    for item in ls:
        useful.append(item*3)
        useful.append(item*3+1)
        useful.append(item*3+2)
    return useful


if __name__ == '__main__':
    import cv2
    target_frame_num = 500
    maximum_angle = 10
    useful_idx = get_useful_idx([0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27])
    plt.figure()
    out = cv2.VideoWriter('angle_random.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (900, 900))
    angle_dicts = {1: [10, 10, 10],
                   # 2: [5, 15, 10],
                   8: [20, 30, 20]}
    bone = np.array([-26.017000198364258, 659.6329956054688, 917.6480102539062, -130.90054321289062, 603.06494140625, 923.584228515625, -162.97560119628906, 484.32232666015625, 513.33984375, -276.5113220214844, 538.9822387695312, 89.21865844726562, -264.859130859375, 403.36260986328125, 34.39588928222656, -265.51898193359375, 332.306640625, 58.38550567626953, 78.86579895019531, 716.20068359375, 911.7118530273438, 109.70648193359375, 728.9241943359375, 484.7274475097656, 69.05551147460938, 862.8753051757812, 65.01152038574219, 150.8942108154297, 745.80712890625, 31.378429412841797, 200.72909545898438, 694.290283203125, 53.457279205322266, -26.011560440063477, 659.6333618164062, 917.7478637695312, -53.7849235534668, 659.4111328125, 1140.2408447265625, -88.10198211669922, 626.5931396484375, 1389.8197021484375, -66.125, 532.8142700195312, 1456.4462890625, -112.96037292480469, 602.103515625, 1535.3773193359375, -88.10198211669922, 626.5931396484375, 1389.8197021484375, 20.55335807800293, 709.119873046875, 1346.6932373046875, 241.72450256347656, 752.8900146484375, 1208.22802734375, 392.4771728515625, 560.609130859375, 1254.20361328125, 392.4771728515625, 560.609130859375, 1254.20361328125, 345.77362060546875, 546.5153198242188, 1341.496337890625, 475.9122009277344, 520.8990478515625, 1292.4317626953125, 475.9122009277344, 520.8990478515625, 1292.4317626953125, -88.10198211669922, 626.5931396484375, 1389.8197021484375, -187.40086364746094, 567.8629760742188, 1305.1588134765625, -151.79685974121094, 424.5013427734375, 1085.6495361328125, 78.09835815429688, 508.656982421875, 1128.986572265625, 78.09835815429688, 508.656982421875, 1128.986572265625, 77.14738464355469, 587.83203125, 1190.065185546875, 206.67327880859375, 479.8706970214844, 1168.303466796875, 206.67327880859375, 479.8706970214844, 1168.303466796875])
    bone = bone[useful_idx]

    exploration(bone, angle_dicts)

    #
    # for i in range(target_frame_num):
    #     random_angles_dict = {i: [np.random.normal(scale=maximum_angle), np.random.normal(scale=maximum_angle),
    #                               np.random.normal(scale=maximum_angle)] for i in range(len(root_joints))}
    #     bone = exploration(bone, random_angles_dict)
    #     frame = cv2.resize(cv2.imread("./EXAMPLE1.JPG"), (900, 900))
    #     cv2.imshow("mutation", frame)
    #     out.write(frame)
    #     cv2.waitKey(1)
    # out.release()
