import numpy as np
from matplotlib import pyplot as plt

from utils.smpl_utils.joint_names import SMPL_BASE_LIMB_MAP, SMPL_73_LIMB_MAP, SMPL_45_LIMB_MAP


def draw_seq_3d_in_one_scene(seq_data):
    """
    Draw a sequence motion in a 3D scene
    :param seq_data: [T, Joint, 3] or [T, Body, Joint, 3]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color_list = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w'] * 3
    plt.ion()
    plt.cla()
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    ax.set_zlim((0.1, 3))
    plt.tight_layout()
    plt.ylim((-3, 3))
    plt.xlim((-3, 3))
    # Make a 3D quiver plot
    x, y, z = np.zeros((3, 3))
    u, v, w = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1)

    if len(seq_data.shape) == 3:
        seq_data = seq_data[:, np.newaxis]
    for frame in seq_data:
        for bIdx, body in enumerate(frame):
            if len(body) == 19:
                limb_map = [[1, 1, 1, 5, 6, 0, 0, 2, 3, 5, 12, 11, 12, 7, 14, 13, 14, 4, 4],
                            [5, 6, 4, 2, 3, 2, 3, 7, 8, 11, 6, 15, 16, 13, 8, 18, 17, 9, 10], ]
            elif len(body) == 25:  # Openpose
                limb_map = [[0, 0, 0, 1, 1, 2, 2, 3, 5, 5, 6, 8, 8, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 19, 22],
                            [1, 15, 16, 2, 5, 3, 9, 4, 6, 12, 7, 9, 12, 10, 11, 22, 24, 13, 14, 19, 21, 17, 18, 20, 23]]
            elif len(body) == 15:  # Shelf
                limb_map = [[0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 2, 3, 8, 9],
                            [1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 8, 9, 12, 12]]
            elif len(body) == 17:  # Human 3.6M
                limb_map = [[0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15, 0],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 8]]
            elif len(body) == 34:
                limb_map = [[0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15, 0,
                             17, 18, 19, 17, 21, 22, 17, 24, 25, 26, 25, 28, 29, 25, 31, 32, 17],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 8,
                             18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 25]]
            elif len(body) == 22:  # SMPL base (w/o hand, w/o foot)
                limb_map = SMPL_BASE_LIMB_MAP
            elif len(body) == 21:  # KIT
                limb_map = [[0, 1, 2, 3, 3, 8, 9, 3, 5, 6, 0, 16, 17, 19, 0, 11, 12, 13, 14],
                            [1, 2, 3, 4, 8, 9, 10, 5, 6, 7, 16, 17, 18, 20, 11, 12, 13, 14, 15]]
            elif len(body) == 73:  # SMPL+H (w/ hand, w/ foot)
                limb_map = SMPL_73_LIMB_MAP
            elif len(body) == 45:  # SMPL-X base
                limb_map = SMPL_45_LIMB_MAP
            elif len(body) == 6:
                limb_map = [[0, 1, 2, 3, 2],
                            [1, 2, 3, 4, 5]]
            else:
                limb_map = None

            x = body[:, 0].squeeze()
            y = body[:, 1].squeeze()
            z = body[:, 2].squeeze()
            # draw joint
            ax.scatter(x, y, z, c=color_list[bIdx], s=10, marker='x')
            # draw limb
            if limb_map is not None:
                for limbIdx in range(len(limb_map[0])):
                    jA_x = x[limb_map[0][limbIdx]]
                    jA_y = y[limb_map[0][limbIdx]]
                    jA_z = z[limb_map[0][limbIdx]]
                    jB_x = x[limb_map[1][limbIdx]]
                    jB_y = y[limb_map[1][limbIdx]]
                    jB_z = z[limb_map[1][limbIdx]]
                    limb_x = np.linspace(jA_x, jB_x, 100)
                    limb_y = np.linspace(jA_y, jB_y, 100)
                    limb_z = np.linspace(jA_z, jB_z, 100)
                    ax.plot(limb_x, limb_y, limb_z, linewidth=1)

    plt.ioff()  # disable interactive mode
    plt.show()
