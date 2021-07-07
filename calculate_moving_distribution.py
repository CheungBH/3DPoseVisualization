#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import cv2


multiview_data = np.load("resources/3D_sence_multiview.npy", allow_pickle=True).tolist()
subject_name, camera_name, action_name, camera_configs, labels = multiview_data['subject_names'], multiview_data[
    'camera_names'], multiview_data['action_names'], multiview_data['cameras'], multiview_data['table']


def get_movement(specific_subject, specific_action):
    mask_subject = labels['subject_idx'] == subject_name.index(specific_subject)
    actions = [action_name.index(specific_action)]

    mask_actions = np.isin(labels['action_idx'], actions)
    mask_subject = mask_subject & mask_actions
    indices = []
    indices.append(np.nonzero(mask_subject)[0])
    specific_label = labels[np.concatenate(indices)]
    specific_3d_skeleton = specific_label['keypoints']

    movements, abs_movements = [], []
    for i in range(len(specific_3d_skeleton) - 1):
        movements.append(specific_3d_skeleton[i + 1] - specific_3d_skeleton[i])
        abs_movements.append(abs(specific_3d_skeleton[i + 1] - specific_3d_skeleton[i]))
    return movements, abs_movements, len(specific_3d_skeleton)


def draw_gaussian(dist, num_column=20, name=""):
    mean = np.mean(dist)
    sigma = np.std(dist)
    print("{}: mean {}, sigma {}".format(name, mean, sigma))
    plt.hist(dist, bins=num_column)
    plt.title("{} movement distribution".format(name))
    plt.savefig("mean.jpg")
    return cv2.imread("mean.jpg"), mean, sigma


if __name__ == '__main__':
    subject_name = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
    action_name = ['Directions-1', 'Directions-2', 'Discussion-1', 'Discussion-2', 'Eating-1', 'Eating-2', 'Greeting-1', 'Greeting-2', 'Phoning-1', 'Phoning-2', 'Posing-1', 'Posing-2', 'Purchases-1', 'Purchases-2', 'Sitting-1', 'Sitting-2', 'SittingDown-1', 'SittingDown-2', 'Smoking-1', 'Smoking-2', 'TakingPhoto-1', 'TakingPhoto-2', 'Waiting-1', 'Waiting-2', 'Walking-1', 'Walking-2', 'WalkingDog-1', 'WalkingDog-2', 'WalkingTogether-1', 'WalkingTogether-2']
    diff = []
    for s_name in subject_name:
        for a_name in action_name:
            mvm, abs_mvm, mvl = get_movement(s_name, a_name)
            diff += mvm
            # print(sum(abs_mvm)/mvl)
    diff_np = np.array(diff)
    x_img, x_mean, x_sigma = draw_gaussian(diff_np[0], name="x")
    y_img, y_mean, y_sigma = draw_gaussian(diff_np[1], name="y")
    z_img, z_mean, z_sigma = draw_gaussian(diff_np[2], name="z")
    img = cv2.hconcat((x_img, y_img, z_img))
    cv2.imshow("img", img)
    cv2.imwrite("distribution.jpg", img)
    cv2.waitKey(0)
