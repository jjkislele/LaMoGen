import copy

SMPL_45_JOINT_NAMES = {
    'pelvis': 0,
    'left_hip': 1,
    'right_hip': 2,
    'spine1': 3,
    'left_knee': 4,
    'right_knee': 5,
    'spine2': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'spine3': 9,
    'left_foot': 10,
    'right_foot': 11,
    'neck': 12,
    'left_collar': 13,
    'right_collar': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
    'left_hand': 22,
    'right_hand': 23,
    'nose': 24,
    'right_eye': 25,
    'left_eye': 26,
    'right_ear': 27,
    'left_ear': 28,
    'left_big_toe': 29,
    'left_small_toe': 30,
    'left_heel': 31,
    'right_big_toe': 32,
    'right_small_toe': 33,
    'right_heel': 34,
    'left_thumb': 35,
    'left_index': 36,
    'left_middle': 37,
    'left_ring': 38,
    'left_pinky': 39,
    'right_thumb': 40,
    'right_index': 41,
    'right_middle': 42,
    'right_ring': 43,
    'right_pinky': 44,
}

SMPLH_JOINT_NAMES = {
    'pelvis': 0,
    'left_hip': 1,
    'right_hip': 2,
    'spine1': 3,
    'left_knee': 4,
    'right_knee': 5,
    'spine2': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'spine3': 9,
    'left_foot': 10,
    'right_foot': 11,
    'neck': 12,
    'left_collar': 13,
    'right_collar': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
    'left_index1': 22,
    'left_index2': 23,
    'left_index3': 24,
    'left_middle1': 25,
    'left_middle2': 26,
    'left_middle3': 27,
    'left_pinky1': 28,
    'left_pinky2': 29,
    'left_pinky3': 30,
    'left_ring1': 31,
    'left_ring2': 32,
    'left_ring3': 33,
    'left_thumb1': 34,
    'left_thumb2': 35,
    'left_thumb3': 36,
    'right_index1': 37,
    'right_index2': 38,
    'right_index3': 39,
    'right_middle1': 40,
    'right_middle2': 41,
    'right_middle3': 42,
    'right_pinky1': 43,
    'right_pinky2': 44,
    'right_pinky3': 45,
    'right_ring1': 46,
    'right_ring2': 47,
    'right_ring3': 48,
    'right_thumb1': 49,
    'right_thumb2': 50,
    'right_thumb3': 51,
    'nose': 52,
    'right_eye': 53,
    'left_eye': 54,
    'right_ear': 55,
    'left_ear': 56,
    'left_big_toe': 57,
    'left_small_toe': 58,
    'left_heel': 59,
    'right_big_toe': 60,
    'right_small_toe': 61,
    'right_heel': 62,
    'left_thumb': 63,
    'left_index': 64,
    'left_middle': 65,
    'left_ring': 66,
    'left_pinky': 67,
    'right_thumb': 68,
    'right_index': 69,
    'right_middle': 70,
    'right_ring': 71,
    'right_pinky': 72,
}

# joint 22
SMPL_BASE_LIMB_MAP = [[0, 1, 4, 0, 2, 5, 10, 8, 13, 16, 18, 14, 17, 19, 15, 0, 3, 6, 9, 12],
                      [1, 4, 7, 2, 5, 8, 7, 11, 16, 18, 20, 17, 19, 21, 12, 3, 6, 9, 12, 15]]

# joint 73, SMPL+H
SMPL_73_LIMB_MAP = copy.deepcopy(SMPL_BASE_LIMB_MAP)
# hand
SMPL_73_LIMB_MAP[0] += [20, 34, 22, 25, 28, 31, 21, 49, 37, 40, 46]
SMPL_73_LIMB_MAP[1] += [34, 22, 25, 28, 31, 20, 49, 37, 40, 43, 21]
# head
SMPL_73_LIMB_MAP[0] += [52, 52, 53, 54]
SMPL_73_LIMB_MAP[1] += [53, 54, 55, 56]
# foot
SMPL_73_LIMB_MAP[0] += [10, 57, 58, 11, 60, 61]
SMPL_73_LIMB_MAP[1] += [57, 58, 59, 60, 61, 62]

# joint 45, SMPL-X no face, no fingers
SMPL_45_LIMB_MAP = copy.deepcopy(SMPL_BASE_LIMB_MAP)
# hand
SMPL_45_LIMB_MAP[0] += [20, 22, 35, 21, 23, 40]
SMPL_45_LIMB_MAP[1] += [22, 35, 20, 23, 40, 21]
# head
SMPL_45_LIMB_MAP[0] += [24, 24, 25, 26]
SMPL_45_LIMB_MAP[1] += [25, 26, 27, 28]
# foot
SMPL_45_LIMB_MAP[0] += [10, 29, 30, 11, 32, 33]
SMPL_45_LIMB_MAP[1] += [29, 30, 31, 32, 33, 34]
