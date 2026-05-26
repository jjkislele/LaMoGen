# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
# H36M_NAMES[4] = 'RBigToe'
# H36M_NAMES[5] = 'RSmallToe'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
# H36M_NAMES[9] = 'LBigToe'
# H36M_NAMES[10] = 'LSmallToe'
# H36M_NAMES[11] = 'Hip'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
# H36M_NAMES[16] = 'Thorax'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
# H36M_NAMES[20] = 'LWrist'
# H36M_NAMES[21] = 'LHand1'
# H36M_NAMES[22] = 'LHand2'
# H36M_NAMES[23] = 'None'
# H36M_NAMES[24] = 'Thorax'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'
# H36M_NAMES[28] = 'RWrist'
# H36M_NAMES[29] = 'RHand1'
# H36M_NAMES[30] = 'RHand2'
# H36M_NAMES[31] = 'None'

H36M_JOINT_A = [1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 13, 14, 15, 14, 18, 19, 20, 20, 14, 26, 27, 28, 28]
H36M_JOINT_B = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 22, 23, 26, 27, 28, 30, 31]
H36M_LR = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

H36M_SUB_JOINT_A = [1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]
H36M_SUB_JOINT_B = [2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]
H36M_SUB_LR = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

H36M_FULL_TO_SUB = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

BODY25_H36M_map = [14, -1, 17, 18, 19, 25, 26, 27, -1, 1, 2, 3, 6, 7, 8, -1, -1, -1, -1, 8, 9, -1, 4, 5, -1]

BODY25_NAMES = [''] * 25
BODY25_JOINT_A = [0, 0, 0, 1, 1, 2, 2, 3, 5, 5, 6, 8, 8, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 19, 22]
BODY25_JOINT_B = [1, 15, 16, 2, 5, 3, 9, 4, 6, 12, 7, 9, 12, 10, 11, 22, 24, 13, 14, 19, 21, 17, 18, 20, 23]
BODY25_LR = [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0]


def update_sequence(sequence, rm_joint=True, go_root=True):
    """
    1. convert from MM to M
    2. swap z/y axis
    3. [optional] remove joints
    4. [optional] root relative
    :param sequence: a video sequence with shape T x J x 3
    :param rm_joint: flag for removing joints
    :return: updated sequence
    """
    assert len(sequence.shape) == 3
    assert sequence.shape[-1] == 3
    remove_lst = [4, 5, 9, 10, 11, 14, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]

    new_sequence = []
    for frame in sequence:
        new_frame = []
        frame /= 1000.
        if go_root:
            frame -= frame[0]  # Hip
        for jIdx, joint in enumerate(frame):
            if jIdx in remove_lst and rm_joint:
                continue
            tmp = joint[1]
            joint[1] = joint[2]
            joint[2] = -tmp
            new_frame.append(joint)
        new_sequence.append(new_frame)
    return new_sequence
