"""
Functions to visualize human poses
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from .frame import ReferenceFrame
from .utils import *


def wrap_show3d_sequence(sequence, camera_param=None, bone_type='H36M', out_path=None):
    """
    Visualize a video sequence wrapper
    :param sequence: frame_num * 96
    :return: None
    """
    print("\t-> Frame num: {}, joint num: {}".format(len(sequence), sequence.shape[1] // 3))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i = 0
    while i < sequence.shape[0]:
        cur_mot = sequence[i]

        if out_path:
            out_path_ = out_path + '/{:04d}.jpg'.format(i + 1)
            if os.path.exists(out_path_):
                print(f"\t-> {out_path_} jumped")
                i += 1
                continue

        plt.ion()
        plt.cla()
        # 1. draw skeleton
        print("\t-> {}".format(i))
        show3Dpose(cur_mot, ax, add_labels=True, bone_type=bone_type, no_trace=False)
        # 2. draw axis
        # World coordinate axis
        world_origin = np.zeros(3)
        dx, dy, dz = np.eye(3)
        world_frame = ReferenceFrame(origin=world_origin,
                                     dx=dx,
                                     dy=dy,
                                     dz=dz,
                                     name="World", )
        world_frame.draw3d()
        # Camera coordinate axis
        if camera_param is not None:
            for cIdx, camera in enumerate(camera_param):
                dx, dy, dz = camera[0]  # R
                t = camera[1].squeeze() / 1000.  # t
                camera_frame = ReferenceFrame(origin=t,
                                              dx=dx,
                                              dy=dy,
                                              dz=dz,
                                              name="C {}".format(cIdx), )
                camera_frame.draw3d()
        # 3. store if needed
        if out_path:
            # if (i - 1) % 5 == 0:
            plt.savefig(out_path + '/{:04d}.jpg'.format(i + 1),
                        dpi=320, format='jpg', transparent=False, bbox_inches='tight', pad_inches=0)
        plt.pause(0.0001)
        i += 1

    # plt.ioff()  # disable interactive mode
    # plt.show()


def wrap_show3d_sequence_gt(pred, gt, camera_param=None, bone_type='H36M', out_path=None):
    """
    Visualize a video sequence wrapper
    :param sequence: frame_num * 96
    :return: None
    """
    print("\t-> Frame num: {}, joint num: {}".format(len(pred), pred.shape[1] // 3))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i = 0
    while i < pred.shape[0]:
        plt.ion()
        plt.cla()

        # 1. draw skeleton
        print("\t-> {}".format(i))
        show3Dpose_gt(pred[i], gt[i], ax, add_labels=True, bone_type=bone_type, no_trace=False)
        # 2. draw axis
        # World coordinate axis
        world_origin = np.zeros(3)
        dx, dy, dz = np.eye(3)
        world_frame = ReferenceFrame(origin=world_origin,
                                     dx=dx,
                                     dy=dy,
                                     dz=dz,
                                     name="World", )
        # world_frame.draw3d()
        # Camera coordinate axis
        if camera_param is not None:
            for cIdx, camera in enumerate(camera_param):
                dx, dy, dz = camera[0]  # R
                t = camera[1].squeeze() / 1000.  # t
                camera_frame = ReferenceFrame(origin=t,
                                              dx=dx,
                                              dy=dy,
                                              dz=dz,
                                              name="C {}".format(cIdx), )
                camera_frame.draw3d()

        if out_path:
            if (i - 1) % 5 == 0:
                plt.savefig(out_path + '_{}.jpg'.format(i),
                            dpi=320, format='jpg', transparent=False, bbox_inches='tight', pad_inches=0)
        plt.pause(0.0001)
        i += 1
    plt.ioff()  # disable interactive mode
    plt.show()


def wrap_show3d_pose(vals3d, add_labels=True):
    """
    Simple wrapper for 3d skeleton
    :param vals3d: 96x1 vector. The pose to plot.
    :param add_labels: whether to add coordinate labels
    :return: None
    """
    fig3d = plt.figure()
    ax3d = Axes3D(fig3d)
    show3Dpose(vals3d, ax3d, add_labels=add_labels)
    plt.show()


# modified and cited from
# https://github.com/una-dinosauria/3d-pose-baseline/blob/666080d86a96666d499300719053cc8af7ef51c8/src/viz.py#L10
def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False,
               bone_type="H36M", no_trace=True, fixed_pos=True):  # blue, orange
    """
    Visualize a 3d skeleton
    Args
      channels: 96x1 vector. The pose to plot.
      ax: matplotlib 3d axis to draw on
      lcolor: color for left part of the body
      rcolor: color for right part of the body
      add_labels: whether to add coordinate labels
      bone_type: the type of the bone
    Returns
      Nothing. Draws on ax.
    """

    if bone_type is "H36M":
        assert channels.size == len(H36M_NAMES) * 3, \
            "channels should have 96 entries, it has %d instead" % channels.size
        vals = np.reshape(channels, (len(H36M_NAMES), -1))
        I = np.array(H36M_JOINT_A) - 1
        J = np.array(H36M_JOINT_B) - 1
        LR = np.array(H36M_LR, dtype=bool)
    elif bone_type is "H36M_SUB":
        # default config
        assert channels.size == len(H36M_NAMES) * 3, \
            "channels should have 96 entries, it has %d instead" % channels.size
        vals = np.reshape(channels, (len(H36M_NAMES), -1))
        I = np.array(H36M_SUB_JOINT_A) - 1
        J = np.array(H36M_SUB_JOINT_B) - 1
        LR = np.array(H36M_SUB_LR, dtype=bool)
    elif bone_type is 'BODY25':
        vals = np.reshape(channels, (len(BODY25_NAMES), -1))
        I = np.array(BODY25_JOINT_A)
        J = np.array(BODY25_JOINT_B)
        LR = np.array(BODY25_LR, dtype=bool)
    else:
        assert False, 'Unknown bone type: {}'.format(bone_type)

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

    if not fixed_pos:
        RADIUS = 5000  # space around the subject
        if no_trace:
            xroot, yroot, zroot = 0., 0., 0.
        else:
            xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
        ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
        ax.set_zlim3d([-RADIUS * 0.1 + zroot, RADIUS + zroot])
        ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    else:
        # ax.set_xlim3d([-0.8, 0.8])
        # ax.set_zlim3d([-0.8, 0.8])
        # ax.set_ylim3d([-0.8, 0.8])
        ax.set_xlim3d([-2., 2.])
        ax.set_zlim3d([0., 4.])
        ax.set_ylim3d([-2., 2.])

    add_labels = False

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        # Get rid of the ticks and tick labels
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)


def show3Dpose_gt(channels, channels_gt, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False,
                  bone_type="H36M", no_trace=True):  # blue, orange
    """
    Visualize a 3d skeleton
    Args
      channels: 96x1 vector. The pose to plot.
      ax: matplotlib 3d axis to draw on
      lcolor: color for left part of the body
      rcolor: color for right part of the body
      add_labels: whether to add coordinate labels
      bone_type: the type of the bone
    Returns
      Nothing. Draws on ax.
    """

    if bone_type is "H36M":
        assert channels.size == len(H36M_NAMES) * 3, \
            "channels should have 96 entries, it has %d instead" % channels.size
        vals = np.reshape(channels, (len(H36M_NAMES), -1))
        vals_gt = np.reshape(channels_gt, (len(H36M_NAMES), -1))
        I = np.array(H36M_JOINT_A) - 1
        J = np.array(H36M_JOINT_B) - 1
        LR = np.array(H36M_LR, dtype=bool)
    elif bone_type is "H36M_SUB":
        # default config
        assert channels.size == len(H36M_NAMES) * 3, \
            "channels should have 96 entries, it has %d instead" % channels.size
        vals = np.reshape(channels, (len(H36M_NAMES), -1))
        vals_gt = np.reshape(channels_gt, (len(H36M_NAMES), -1))
        I = np.array(H36M_SUB_JOINT_A) - 1
        J = np.array(H36M_SUB_JOINT_B) - 1
        LR = np.array(H36M_SUB_LR, dtype=bool)
    elif bone_type is 'BODY25':
        vals = np.reshape(channels, (len(BODY25_NAMES), -1))
        vals_gt = np.reshape(channels_gt, (len(BODY25_NAMES), -1))
        I = np.array(BODY25_JOINT_A)
        J = np.array(BODY25_JOINT_B)
        LR = np.array(BODY25_LR, dtype=bool)
    else:
        assert False, 'Unknown bone type: {}'.format(bone_type)

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)
        x_gt, y_gt, z_gt = [np.array([vals_gt[I[i], j], vals_gt[J[i], j]]) for j in range(3)]
        ax.plot(x_gt, y_gt, z_gt, lw=1, c='k')

    # RADIUS = 5000  # space around the subject
    # if no_trace:
    #     xroot, yroot, zroot = 0., 0., 0.
    # else:
    #     xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    # ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    # ax.set_zlim3d([-RADIUS * 0.1 + zroot, RADIUS + zroot])
    # ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_xlim3d([-0.8, 0.8])
    ax.set_zlim3d([-0.8, 0.8])
    ax.set_ylim3d([-0.8, 0.8])

    add_labels = False

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        # Get rid of the ticks and tick labels
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)
