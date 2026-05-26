import torch

from utils.rotation_conversion import axis_angle_to_quaternion, quaternion_to_matrix, matrix_to_quaternion, \
    euler_angles_to_matrix, quaternion_multiply, quaternion_invert, matrix_to_euler_angles, quaternion_apply, \
    matrix_to_axis_angle


def norm_amass_motion(poses_src, trans_src):
    """
    Rotate AMASS motion (z-up) to always face forward
    :param poses_src: np.array(T, J, 3)
    :param trans_src: np.array(T, 3)
    :return: poses: np.array(T, J, 3); trans: np.array(T, 3)
    """
    poses, trans = torch.from_numpy(poses_src.copy()), torch.from_numpy(trans_src.copy())

    # rotation
    poses = axis_angle_to_quaternion(poses)
    # rotate motion to always face forward
    ref_pose = matrix_to_quaternion(torch.eye(3))
    cur_pose = poses[0, 0]
    root_rotate = quaternion_multiply(ref_pose, quaternion_invert(cur_pose))
    # normalize root_rotate to only work on horizontal orientation, not elevation
    root_rotateY = matrix_to_euler_angles(quaternion_to_matrix(root_rotate), "XYZ")
    root_rotateY[0] = 0.
    root_rotateY[1] = 0.
    root_rotate = matrix_to_quaternion(euler_angles_to_matrix(root_rotateY, "XYZ"))
    poses[:, 0] = quaternion_multiply(root_rotate.unsqueeze(0), poses[:, 0])
    # trajectory
    trans[:, [0, 1]] = trans[:, [0, 1]] - trans[[0], [0, 1]]
    # rotate among z axis
    trans = quaternion_apply(root_rotate.unsqueeze(0), trans)
    poses = matrix_to_axis_angle(quaternion_to_matrix(poses))

    poses, trans = poses.numpy(), trans.numpy()
    return poses, trans


def flip_amass_motion(poses_src, trans_src, is_norm=False):
    """
    Flip AMASS motion. Must be normalized first.
    :param poses_src: np.array(T, J, 3)
    :param trans_src: np.array(T, 3)
    :param is_norm: if normalized then jump this
    :return: poses: np.array(T, J, 3); trans: np.array(T, 3)
    """

    def norm_in_quat(poses, trans):
        ref_pose = matrix_to_quaternion(torch.eye(3))
        cur_pose = poses[0, 0]
        root_rotate = quaternion_multiply(ref_pose, quaternion_invert(cur_pose))
        # normalize root_rotate to only work on horizontal orientation, not elevation
        root_rotateY = matrix_to_euler_angles(quaternion_to_matrix(root_rotate), "XYZ")
        root_rotateY[0] = 0.
        root_rotateY[1] = 0.
        root_rotate = matrix_to_quaternion(euler_angles_to_matrix(root_rotateY, "XYZ"))
        poses[:, 0] = quaternion_multiply(root_rotate.unsqueeze(0), poses[:, 0])
        # trajectory
        trans[:, [0, 1]] = trans[:, [0, 1]] - trans[[0], [0, 1]]
        # rotate among z axis
        trans = quaternion_apply(root_rotate.unsqueeze(0), trans)
        return poses, trans

    def flip_in_euler(poses_euler, trans):
        right_index = [2, 5, 8, 11, 14, 17, 19, 21]
        left_index = [1, 4, 7, 10, 13, 16, 18, 20]
        right_euler = (poses_euler[:, right_index]).clone()
        left_euler = (poses_euler[:, left_index]).clone()
        right_euler[:, :, [1]] = -right_euler[:, :, [1]]
        right_euler[:, :, [2]] = -right_euler[:, :, [2]]
        left_euler[:, :, [1]] = -left_euler[:, :, [1]]
        left_euler[:, :, [2]] = -left_euler[:, :, [2]]
        poses_euler[:, right_index] = left_euler
        poses_euler[:, left_index] = right_euler
        # spine
        spine_index = [0, 3, 6, 9, 12, 15]
        poses_euler[:, spine_index, [1]] = -poses_euler[:, spine_index, [1]]
        poses_euler[:, spine_index, [2]] = -poses_euler[:, spine_index, [2]]
        # flip on x
        trans[:, 0] = -trans[:, 0]
        return poses_euler, trans

    poses, trans = torch.from_numpy(poses_src.copy()), torch.from_numpy(trans_src.copy())
    if not is_norm:
        poses, trans = norm_in_quat(axis_angle_to_quaternion(poses),
                                    trans)
    else:
        poses = axis_angle_to_quaternion(poses)
    poses_euler, trans = flip_in_euler(matrix_to_euler_angles(quaternion_to_matrix(poses), 'XYZ'),
                                       trans)
    poses = matrix_to_axis_angle(euler_angles_to_matrix(poses_euler, 'XYZ'))
    poses, trans = poses.numpy(), trans.numpy()
    return poses, trans
