import numpy as np
import torch

from utils.rotation_conversion import axis_angle_to_matrix, matrix_to_rotation_6d
from utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle


def trans_interp(trans_A, trans_B, step):
    alpha = np.linspace(0, 1, step + 1)
    last = np.einsum("l,...->l...", 1 - alpha, trans_A)
    new = np.einsum("l,...->l...", alpha, trans_B)
    chuncks = (last + new)[:-1]
    return chuncks


def poses_axis_interp_rot6d(poses_A, poses_B, step):
    # TODO: which one is better? axis-based or rot6d-based?
    poses_A = torch.from_numpy(poses_A)
    poses_B = torch.from_numpy(poses_B)
    poses_A_6d = matrix_to_rotation_6d(axis_angle_to_matrix(poses_A))
    poses_B_6d = matrix_to_rotation_6d(axis_angle_to_matrix(poses_B))
    alpha = torch.linspace(0, 1, step + 1)
    last = torch.einsum("l,...->l...", 1 - alpha, poses_A_6d)
    new = torch.einsum("l,...->l...", alpha, poses_B_6d)
    chuncks = (last + new)[:-1]
    chuncks_axis = matrix_to_axis_angle(rotation_6d_to_matrix(chuncks))
    return chuncks_axis.numpy()


def poses_axis_interp(poses_A, poses_B, step):
    # TODO: which one is better? axis-based or rot6d-based?
    alpha = np.linspace(0, 1, step + 1)
    last = np.einsum("l,...->l...", 1 - alpha, poses_A)
    new = np.einsum("l,...->l...", alpha, poses_B)
    chuncks = (last + new)[:-1]
    return chuncks
