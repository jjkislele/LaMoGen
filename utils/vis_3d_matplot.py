import numpy as np

from .matplot_vis.utils import H36M_FULL_TO_SUB
from .matplot_vis.visualization import wrap_show3d_sequence


def vis_mot_seq(mot_seq, output_root):
    """
    :param mot_seq: frame x joint (17) x 3
    :param output_root:
    :return:
    """
    # swap world space from opencv
    # mot_seq[:, :, [0, 2, 1]] = mot_seq[:, :, [0, 1, 2]]
    # mot_seq[:, :, 2] = -mot_seq[:, :, 2]

    T, J, D = mot_seq.shape
    clip_full = np.zeros((T, J, D)) if J == 32 else np.zeros((T, 32, D))
    clip_full[:, H36M_FULL_TO_SUB] = mot_seq
    wrap_show3d_sequence(clip_full.reshape(-1, 96), bone_type='H36M_SUB',
                         out_path=output_root)
