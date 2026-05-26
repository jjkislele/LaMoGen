import numpy as np

codebook_range = {
    'support_l': {'basic': (0, 9),
                  'knee_bend': (9, 15),
                  'hip_bend': (15, 21),
                  'is_idle': (21, 23)},  # 0: False, 1: True
    'support_r': {'basic': (23, 32),
                  'knee_bend': (32, 38),
                  'hip_bend': (38, 44),
                  'is_idle': (44, 46)},  # 0: False, 1: True
    'support_both': {'orient_horz': (46, 54),
                     'orient_vert': (54, 62),
                     'velocity_horz': (62, 67),
                     'velocity_vert': (67, 72)},
    'upper_l': {'basic': (72, 81),
                'elbow_bend': (81, 87),
                'shoulder_bend': (87, 93),
                'is_idle': (93, 95)},  # 0: False, 1: True
    'upper_r': {'basic': (95, 104),
                'elbow_bend': (104, 110),
                'shoulder_bend': (110, 116),
                'is_idle': (116, 118)},  # 0: False, 1: True
    'torso': {'head_orient_horz': (118, 126),
              'head_orient_vert': (126, 134),
              'spine': (134, 140)},
    # 06/07 @ new code: relative wrist laban. to differentiate the waving
    'upper_l_rel': {'basic': (140, 149), },
    'upper_r_rel': {'basic': (149, 158), },
}

codebook_size = -1
to_code_range = []
for k, v in codebook_range.items():
    for kk, vv in v.items():
        codebook_size = max(codebook_size, vv[1])
        if kk == 'basic':
            bgn, end = vv
            vv = [(bgn + ii * 3, bgn + ii * 3 + 3) for ii in range(3)]
            to_code_range += vv
        else:
            to_code_range.append(vv)
cat_num = len(to_code_range)
to_code_range.append((158, 160))  # [EOS]
to_code_range.append((160, 360))  # event number


def convert_to_lbn_codebook(lbn_dict):
    length = lbn_dict['length']
    codebook = np.zeros((length, codebook_size), dtype=int)
    for k, v in lbn_dict.items():
        if k == 'length':
            continue
        for kk, vv in v.items():
            vv = np.array(vv, dtype=int)
            bgn, end = codebook_range[k][kk]
            if 'basic' == kk:
                vv += 1  # move [-1, 0, 1] to [0, 1, 2]
                vv += bgn  # add stride
                vv[:, 0] += 0  # side
                vv[:, 1] += 3  # fwd
                vv[:, 2] += 6  # level
            else:
                vv += bgn
                vv = vv[:, np.newaxis]
            codebook[np.arange(length)[:, None], vv] = 1
    return codebook


def convert_back_lbn_codebook(codebook):
    lbn_dict = {}
    length = len(codebook)
    for k, v in codebook_range.items():
        lbn_dict[k] = {}
        for kk, vv in v.items():
            bgn, end = codebook_range[k][kk]
            sel_lbn_code = codebook[:, bgn:end]
            if 'basic' == kk:
                side, fwd, lvl = sel_lbn_code[:, 0:3], sel_lbn_code[:, 3:6], sel_lbn_code[:, 6:9]
                side = np.where(side == 1)[1] - 1
                fwd = np.where(fwd == 1)[1] - 1
                lvl = np.where(lvl == 1)[1] - 1
                lbn_dict[k][kk] = np.concatenate((side[:, None], fwd[:, None], lvl[:, None]), axis=1)
            # 06/09 @ handle the boolean values
            elif 'is_idle' == kk:
                lbn_dict[k][kk] = np.where(sel_lbn_code == 1)[1] > 0
            else:
                lbn_dict[k][kk] = np.where(sel_lbn_code == 1)[1]
    lbn_dict['length'] = length
    return lbn_dict


def check_equal_lbn_dict(src_dict, dst_dict):
    is_equal = []
    for k, v in src_dict.items():
        if k == 'length':
            is_equal.append(v == dst_dict[k])
            continue
        for kk, vv in v.items():
            src_item = np.array(vv)
            dst_item = np.array(dst_dict[k][kk])
            is_equal.append(np.all(src_item == dst_item))
    is_equal = np.array(is_equal)
    return np.all(is_equal)
