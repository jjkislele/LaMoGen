import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import iirfilter


def rotation_on_x_axis(skel, rad):
    rot_mat = np.zeros((3, 3))
    rot_mat[0, 0] = 1
    rot_mat[1, 1] = np.cos(rad)
    rot_mat[1, 2] = -np.sin(rad)
    rot_mat[2, 1] = np.sin(rad)
    rot_mat[2, 2] = np.cos(rad)
    skel = np.dot(skel, rot_mat)
    return skel


def rotation_on_y_axis(skel, rad):
    rot_mat = np.zeros((3, 3))
    rot_mat[0, 0] = np.cos(rad)
    rot_mat[0, 2] = np.sin(rad)
    rot_mat[1, 1] = 1
    rot_mat[2, 0] = -np.sin(rad)
    rot_mat[2, 2] = np.cos(rad)
    skel = np.dot(skel, rot_mat)
    return skel


def rotation_on_z_axis(skel, rad):
    rot_mat = np.zeros((3, 3))
    rot_mat[0, 0] = np.cos(rad)
    rot_mat[0, 1] = -np.sin(rad)
    rot_mat[1, 0] = np.sin(rad)
    rot_mat[1, 1] = np.cos(rad)
    rot_mat[2, 2] = 1
    skel = np.dot(skel, rot_mat)
    return skel


def smooth_joint_trajectory(src_xyz: np.ndarray, cutoff_freq=0.1, order=3, filter_type='butter'):
    """ Smooth one joint trajectory, using butter filter or other filters
    :param src_xyz: T, 3
    :return: smoothed xyz
    """
    dst_xyz = src_xyz.copy()
    # Smooth each dimension (x, y, z) separately
    try:
        for dim in range(3):
            signal = dst_xyz[:, dim]
            if filter_type.lower() == 'butter':
                b, a = butter(order, cutoff_freq)
            else:
                b, a = iirfilter(order, cutoff_freq, btype='lowpass', ftype=filter_type)
            smoothed_signal = filtfilt(b, a, signal)
            dst_xyz[:, dim] = smoothed_signal
    except Exception as e:
        pass
        # print(f"[Smooth] input len: {len(dst_xyz)} | Error msg: {e}")

    return dst_xyz


def canonicalize_amass_skels(src_skels: np.ndarray, lock_bone):
    """Lock bone (pelvis), and face y-
    :param src_skels: T, 22, 3
    :return: canonicalized skeletons
    """
    T = len(src_skels)
    dst_skels = src_skels.copy()
    # lock bone, and face y-
    if 'pelvis' == lock_bone:
        j1 = 1  # l_hip
        j2 = 2  # r_hip
        j3 = 0  # pelvis
        # all joints to local
        dst_skels[:, :] -= dst_skels[:, [0]]
    elif 'r_collar' == lock_bone:
        j1 = 13  # left_collar
        j2 = 17  # right_shoulder
        j3 = 14  # right_collar
        # all joints to local
        dst_skels[:, :] -= dst_skels[:, [14]]
    elif 'l_collar' == lock_bone:
        j1 = 16  # left_shoulder
        j2 = 14  # right_collar
        j3 = 13  # left_collar
        # all joints to local
        dst_skels[:, :] -= dst_skels[:, [13]]
    elif 'neck' == lock_bone:
        j1 = 13  # l_collar
        j2 = 14  # r_collar
        j3 = 12  # neck
        # all joints to local
        dst_skels[:, :] -= dst_skels[:, [12]]
    else:
        return NotImplementedError
    vec_dst = np.array([1, 0])
    for i in range(T):
        # xy plane
        vec1 = dst_skels[i, j1] - dst_skels[i, j2]
        rad = np.math.atan2(np.linalg.det([vec1[:2], vec_dst]), np.dot(vec1[:2], vec_dst))
        dst_skels[i] = rotation_on_z_axis(dst_skels[i], -rad)
        # xz plane
        vec1 = dst_skels[i, j1] - dst_skels[i, j2]
        rad = np.math.atan2(np.linalg.det([vec1[[0, 2]], vec_dst]), np.dot(vec1[[0, 2]], vec_dst))
        dst_skels[i] = rotation_on_y_axis(dst_skels[i], rad)
        # yz plane
        vec1 = dst_skels[i, j1] - dst_skels[i, j3]
        rad = np.math.atan2(np.linalg.det([vec1[[1, 2]], vec_dst]), np.dot(vec1[[1, 2]], vec_dst))
        dst_skels[i] = rotation_on_x_axis(dst_skels[i], np.deg2rad(90) - rad)
    return dst_skels
