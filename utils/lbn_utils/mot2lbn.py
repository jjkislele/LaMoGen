import numpy as np
from collections import Counter

from utils.skel_painter import draw_seq_3d_in_one_scene
from utils.smpl_utils.joint_names import SMPL_45_JOINT_NAMES as jNames
from utils.lbn_utils.aug import canonicalize_amass_skels, smooth_joint_trajectory


def detect_foot_contact(skels, is_smoothed=True):
    """ Foot contact by velocity factor (HumanML3D)
    NOTE: must in global!!!
    :param skels: T, J>=22, 3
    :return: foot contact
    """
    smoothed_lower_trajs = []
    for jTarget in ['left_ankle', 'left_foot', 'right_ankle', 'right_foot']:
        ends = skels[:, jNames[jTarget]]
        if is_smoothed:
            ends_smoothed = smooth_joint_trajectory(ends, cutoff_freq=0.4, order=3)
        else:
            ends_smoothed = ends
        smoothed_lower_trajs.append(ends_smoothed)
    smoothed_lower_trajs = np.stack(smoothed_lower_trajs)
    smoothed_lower_trajs = np.concatenate((smoothed_lower_trajs[:, [0]], smoothed_lower_trajs), axis=1)
    magnitudes = np.diff(smoothed_lower_trajs, axis=1) ** 2
    lower_l = (np.sum(magnitudes[[0, 1]], axis=-1)).transpose(1, 0)
    lower_r = (np.sum(magnitudes[[2, 3]], axis=-1)).transpose(1, 0)
    fc_lr = np.concatenate((np.sum(lower_l < 0.002, axis=-1, keepdims=True) > 1,
                            np.sum(lower_r < 0.002, axis=-1, keepdims=True) > 1), axis=-1)
    return fc_lr


def calc_vec_pairs_deg(vectors_a, vectors_b):
    assert len(vectors_a) == len(vectors_b) and len(vectors_a.shape) == 2
    dot_products = np.einsum('ij,ij->i', vectors_a, vectors_b)
    norms_a = np.linalg.norm(vectors_a, axis=1)
    norms_b = np.linalg.norm(vectors_b, axis=1)
    cos_thetas = dot_products / (norms_a * norms_b)
    cos_thetas = np.clip(cos_thetas, -1.0, 1.0)
    angle_radians = np.arccos(cos_thetas)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def detect_wrist_movement(skels, is_smoothed=True):
    """Detect whether wrists are moving.

    Returns a boolean array where True means **stationary** and False means **moving**.
    NOTE: This definition is opposite to the one used in foot-contact (fc) detection.
    It is kept as-is for backward compatibility.
    """
    smoothed_trajs = []
    for jTarget in ['left_wrist', 'left_elbow', 'right_wrist', 'right_elbow']:
        ends = skels[:, jNames[jTarget]]
        if is_smoothed:
            ends_smoothed = smooth_joint_trajectory(ends, cutoff_freq=0.4, order=3)
        else:
            ends_smoothed = ends
        smoothed_trajs.append(ends_smoothed)
    smoothed_trajs = np.stack(smoothed_trajs)
    smoothed_trajs = np.concatenate((smoothed_trajs[:, [0]], smoothed_trajs), axis=1)
    magnitudes = np.diff(smoothed_trajs, axis=1) ** 2
    wrist_l = (np.sum(magnitudes[[0, 1]], axis=-1)).transpose(1, 0)
    wrist_r = (np.sum(magnitudes[[2, 3]], axis=-1)).transpose(1, 0)
    mv_lr = np.concatenate((np.sum(wrist_l < 0.0005, axis=-1, keepdims=True) > 1,
                            np.sum(wrist_r < 0.0005, axis=-1, keepdims=True) > 1), axis=-1)
    return mv_lr


def detect_frame_based_lbn_lo(skels_canon):
    """
    :param skels_canon: T, 22, 3
    :return: event_label: laban symbol, event_voxel: voxel pose
    """
    assert np.sum(skels_canon[0, jNames['pelvis']]) == 0.
    event_label = {}
    # NOTE: in local, face -y, right hand -x, up +z
    for jTarget in ['left_foot', 'right_foot']:
        ends = skels_canon[:, jNames[jTarget]]
        ends_smoothed = smooth_joint_trajectory(ends, cutoff_freq=0.4, order=3)
        ends_q = np.round(np.round(np.array(ends_smoothed / 0.1)).astype(int) * 0.1, 2)
        ends_label = np.zeros_like(ends_q)
        # x (side)
        # NOTE: we assume the cylinder is 0.3m x 0.3m
        # NOTE: we assume: left: -1, medium: 0, right: +1
        if 'left' in jTarget:
            ends_label[ends_q[:, 0] < -0.1, 0] = 1
            ends_label[ends_q[:, 0] > 0.3, 0] = -1
        else:
            ends_label[ends_q[:, 0] < -0.3, 0] = 1
            ends_label[ends_q[:, 0] > 0.1, 0] = -1
        # y (forward/backward)
        # NOTE: add foot length (0.2m) as offset
        # NOTE: we assume: backward: -1, medium: 0, forward: +1
        # 06/06 @ add 0.1 offset due to canonicalization
        ends_label[ends_q[:, 1] < -0.15, 1] = 1
        ends_label[ends_q[:, 1] > -0.05, 1] = -1
        # z (height)
        # NOTE: we assume leg length is 0.9
        # NOTE: we assume: low: -1, medium: 0, high: +1
        ends_label[ends_q[:, 2] > -0.8, 2] = -1
        ends_label[ends_q[:, 2] > 0., 2] = 1
        # store
        event_label[jTarget] = ends_label
    return event_label


def detect_frame_based_lbn_up(skels_canon):
    """
    :param skels_canon: T, 22, 3
    :return: event_label: laban symbol, event_voxel: voxel pose
    """
    assert np.sum(skels_canon[:, jNames['spine3']]) == 0.
    event_label = {}
    # NOTE: in local, face -y, right hand -x, up +z
    for jTarget in ['left_wrist', 'right_wrist']:
        ends = skels_canon[:, jNames[jTarget]]
        ends_smoothed = smooth_joint_trajectory(ends, cutoff_freq=0.4, order=3)
        ends_q = np.round(np.round(np.array(ends_smoothed / 0.1)).astype(int) * 0.1, 2)
        ends_label = np.zeros_like(ends_q)
        # x (side)
        # NOTE: we assume: left: -1, medium: 0, right: +1
        if 'left' in jTarget:
            ends_label[ends_q[:, 0] < -0.1, 0] = 1
            ends_label[ends_q[:, 0] > 0.3, 0] = -1
        else:
            ends_label[ends_q[:, 0] < -0.3, 0] = 1
            ends_label[ends_q[:, 0] > 0.1, 0] = -1
        # y (forward/backward)
        ends_label[ends_q[:, 1] < -0.2, 1] = 1
        ends_label[ends_q[:, 1] > 0.1, 1] = -1
        # z (height)
        # NOTE: we assume: low: -1, medium: 0, high: +1
        # 06/07 @ Heuristic: arm down -> -1/low, arm horizontal -> 0/medium, arm up -> 1/high
        ends_label[ends_q[:, 2] < -0.1, 2] = -1
        ends_label[ends_q[:, 2] > 0.1, 2] = 1
        # store
        event_label[jTarget] = ends_label
    return event_label


def calc_root_momentum(trajectory, fps=30.):
    """ Get every frame's root momentum, calc it's velocity

    :param trajectory: root trajectory in global: T, 3
    :param fps: get the real velocity, i.e., m/s
    :return: velocity label, [(horz_v_label, vert_v_label), ()]
    """
    velocity = np.diff(trajectory, axis=0)
    first_frame = 0 * velocity[..., [0], :]
    velocity = np.concatenate((first_frame, velocity), axis=-2) / (1 / fps)
    vel_event = {'horz': [], 'vert': []}
    for frame in velocity:
        horz_v = np.sqrt(frame[[0]] ** 2 + frame[[1]] ** 2)
        vert_v = np.abs(frame[2])
        horz_v_label = 0
        vert_v_label = 0
        # [fix] 04/23 @ Sitting motions may have ~0 horizontal velocity but were classified as level-1; adjust thresholds.
        # 06/06 @ Velocity distribution is uneven; hard to separate "slow" vs "still".
        if 0.1 < horz_v <= 0.5:
            horz_v_label = 1
        if 0.5 < horz_v <= 1.0:
            horz_v_label = 2
        if 1.0 < horz_v <= 2.0:
            horz_v_label = 3
        if 2.0 < horz_v:
            horz_v_label = 4
        #
        if 0.1 < vert_v <= 0.5:
            vert_v_label = 1
        if 0.5 < vert_v <= 1.0:
            vert_v_label = 2
        if 1.0 < vert_v <= 2.0:
            vert_v_label = 3
        if 2.0 < vert_v:
            vert_v_label = 4
        vel_event['horz'].append(horz_v_label)
        vel_event['vert'].append(vert_v_label)
    return vel_event


def calc_root_orientation(trajectory):
    """ Get every frame's root horizontal and vertical orientation

    :param trajectory: hip trajectory in global: T, J=2 (left hip, right hip), 3
    :return: velocity label
    """

    def quant_deg(cur_deg):
        cur_deg_quant = 0
        # Boundary condition (be careful).
        if -22.5 < cur_deg <= 22.5:
            cur_deg_quant = 0
        # Positive degrees
        elif 22.5 < cur_deg <= 67.5:
            cur_deg_quant = 1
        elif 67.5 < cur_deg <= 112.5:
            cur_deg_quant = 2
        elif 112.5 < cur_deg <= 157.5:
            cur_deg_quant = 3
        # Negative degrees
        elif -157.5 < cur_deg <= -112.5:
            cur_deg_quant = 5
        elif -112.5 < cur_deg <= -67.5:
            cur_deg_quant = 6
        elif -67.5 < cur_deg <= -22.5:
            cur_deg_quant = 7
        # Boundary condition (be careful).
        elif 157.5 < cur_deg or cur_deg <= -157.5:
            cur_deg_quant = 4
        return cur_deg_quant

    orient_q = {'horz': [], 'vert': []}
    vec_dst = np.array([1, 0])
    for hip_t in trajectory:
        # xy plane
        vec1 = hip_t[0] - hip_t[1]
        deg = np.rad2deg(np.math.atan2(np.linalg.det([vec1[:2], vec_dst]), np.dot(vec1[:2], vec_dst)))
        orient_q['horz'].append(quant_deg(deg))
        # yz plane
        vec1 = hip_t[0] - hip_t[1]
        deg = np.rad2deg(np.math.atan2(np.linalg.det([vec1[[1, 2]], vec_dst]), np.dot(vec1[[1, 2]], vec_dst)))
        orient_q['vert'].append(quant_deg(deg))
    return orient_q


def calc_endeffector_bend_status(skels, kinematic_tree, deg_quant=30.):
    """ Get every frame's bend labels
    """
    bend_q = {}
    for k in kinematic_tree:
        bend_q[k] = []
    for frame in skels:
        for jTarget in kinematic_tree:
            limb_a = frame[np.newaxis, kinematic_tree[jTarget][0]]
            limb_b = frame[np.newaxis, kinematic_tree[jTarget][1]]
            vec_a = limb_a[:, 0] - limb_a[:, 1]
            vec_b = limb_b[:, 0] - limb_b[:, 1]
            limb_deg = calc_vec_pairs_deg(vec_a, vec_b)[0]
            bend_q[jTarget].append(int(limb_deg // deg_quant))
    return bend_q


def process_lower_motion(skels, skels_canon, fps):
    """ Get lower event.
    :param skels: normal skeleton motion, T, >=22, 0
    :param skels_canon: canonicalized skeleton motion, T, >=22, 0
    :return: event_label, event_voxel, event_index
    """
    assert np.sum(skels[0, 0]) != 0.
    assert np.sum(skels_canon[0, 0]) == 0.

    # in cano space
    # fc_lr = detect_foot_contact(skels_canon)
    fc_lr = detect_foot_contact(skels)  # 06/06 @ fix better for golf motion
    lbn_lr = detect_frame_based_lbn_lo(skels_canon)

    # in global space (root)
    # get root's absolute momentum
    lbn_root_v = calc_root_momentum(skels[:, jNames['pelvis']], fps=fps)
    # get root horizontal & vertical orientation
    lbn_root_orient = calc_root_orientation(skels[:, [jNames['left_hip'], jNames['right_hip']]])

    return lbn_lr, lbn_root_v, lbn_root_orient, fc_lr


def process_upper_motion(skels):
    skels_neck = skels.copy()
    skels_neck -= skels_neck[:, [jNames['spine3']]]  # origin is spine3

    mv_lr = detect_wrist_movement(skels_neck, is_smoothed=False)
    lbn_lr = detect_frame_based_lbn_up(skels_neck)
    return lbn_lr, ~mv_lr


def process_upper_motion_local(skels, is_idle):
    """Compute relative arm movement (per-frame).
    :param skels: T, J, 3
    :param is_idle: NOTE: True means moving, False means not moving (kept for compatibility).
    """
    skels_neck = skels.copy()
    skels_neck -= skels_neck[:, [jNames['spine3']]]  # origin is spine3

    lbn_lst = {}
    # NOTE: in local, face -y, right hand -x, up +z
    for idx, jTarget in enumerate(['left_wrist', 'right_wrist']):
        lbn_lst[jTarget] = []
        event = find_event_bgn_end(is_idle[:, idx])
        ends = skels_neck[:, jNames[jTarget]]
        ends_q = np.round(np.round(np.array(ends / 0.1)).astype(int) * 0.1, 2)
        # for each event, get the 'end' related location
        for bgn, end in event:
            # event must be moving
            # [fix] 06/26 @ Previously we did not check movement.
            # Only when the event is moving (True) should we produce a non-zero relative label;
            # otherwise it stays at the neutral position [0, 0, 0].
            bgn_ends = ends_q[bgn]
            end_ends = ends_q[end] - bgn_ends
            related_lbn = [0, 0, 0]
            event_flg = np.all(is_idle[bgn:end + 1, idx])
            if event_flg:
                # x (side)
                # NOTE: we assume: left: -1, medium: 0, right: +1
                if end_ends[0] < 0:
                    related_lbn[0] = 1
                elif end_ends[0] > 0:
                    related_lbn[0] = -1
                # y (forward/backward)
                if end_ends[1] < 0:
                    related_lbn[1] = 1
                elif end_ends[1] > 0:
                    related_lbn[1] = -1
                # z (height)
                # NOTE: we assume: low: -1, medium: 0, high: +1
                if end_ends[2] < 0:
                    related_lbn[2] = -1
                elif end_ends[2] > 0:
                    related_lbn[2] = 1
            # store
            for _ in range(bgn, end + 1):
                lbn_lst[jTarget].append(related_lbn)
    return lbn_lst


def calc_head_status(skels):
    def quant_deg(cur_deg):
        if cur_deg < 0:
            cur_deg += 180
        if 0 < cur_deg <= 22.5:
            cur_deg_quant = 0
        elif 22.5 < cur_deg <= 45:
            cur_deg_quant = 1
        elif 45 < cur_deg <= 67.5:
            cur_deg_quant = 2
        elif 67.5 < cur_deg <= 90.:
            cur_deg_quant = 3
        elif 90. < cur_deg <= 112.5:
            cur_deg_quant = 4
        elif 112.5 < cur_deg <= 135.:
            cur_deg_quant = 5
        elif 135. < cur_deg <= 157.5:
            cur_deg_quant = 6
        elif 157.5 < cur_deg <= 180.:
            cur_deg_quant = 7
        else:
            assert False, f'wrong deg: {cur_deg}'
        return cur_deg_quant

    # 06/09 @ fix: this was likely wrong; the reference should be "neck" rather than "head".
    #       @ It doesn't affect the model output, but it affects text description.
    # 06/19 @ try to fix.
    skels_head = skels.copy()
    # skels_head -= skels_head[:, [jNames['neck']]]
    # Canonicalize so that right_collar/left_collar/neck are aligned with the XZ plane.
    skels_head = canonicalize_amass_skels(skels_head, 'neck')
    # draw_seq_3d_in_one_scene(skels_head[::50])
    # Use an alternative method to compute head-bone orientation.
    orient_q = {'horz': [], 'vert': []}
    vec_dst = np.array([1, 0])
    for kpt in skels_head[:, [jNames['head'], jNames['neck']]]:
        # xy plane
        # Horizontal: angle of the projection on the XY plane (left -> 0, right -> 7)
        vec1 = kpt[0] - kpt[1]
        deg = np.rad2deg(np.math.atan2(np.linalg.det([vec1[:2], vec_dst]), np.dot(vec1[:2], vec_dst)))
        orient_q['horz'].append(quant_deg(deg))
        # yz plane
        # Vertical: angle of the projection on the YZ plane.
        # Aligning with y- -> 0 (head forward), aligning with y+ -> 7 (head backward).
        vec1 = kpt[1] - kpt[0]
        deg = np.rad2deg(np.math.atan2(np.linalg.det([vec1[[1, 2]], vec_dst]), np.dot(vec1[[1, 2]], vec_dst)))
        orient_q['vert'].append(quant_deg(deg))
    return orient_q


def find_event_bgn_end(flg):
    """Return inclusive intervals [begin, end] for consecutive equal values.
    """
    event_list = []
    cur_item = flg[0]
    i = 0
    j = 0
    while i < len(flg) - 1 and j < len(flg) - 1:
        if flg[j + 1] == cur_item:
            j += 1
            continue
        else:
            event_list.append([i, j])
            cur_item = flg[j + 1]
            i = j + 1

    # append last
    if i < len(flg):
        # If input is a list of strings, skip validation.
        if isinstance(flg, np.ndarray):
            assert np.all(np.diff(flg[i:len(flg)]) == 0)
        event_list.append([i, len(flg) - 1])
    return event_list


def find_both_event_bgn_end(flg):
    """Return inclusive intervals [begin, end] for 2D flags (shape: T x 2).
    """
    assert flg.shape[1] == 2
    event_list = []
    i = 0
    j = 1
    while j < len(flg):
        if np.any(flg[i, :] != flg[j, :]):
            event_list.append([i, j - 1])
            i = j
            j += 1
        else:
            j += 1

    # append last
    if i < len(flg):
        assert np.all(np.diff(flg[i:len(flg)], axis=0) == 0)
        event_list.append([i, len(flg) - 1])
    return event_list


def lbn_summarize_lr(lbn_lst, is_idle, last_ratio=0.2):
    assert len(lbn_lst) == 2 and is_idle.shape[1] == 2
    new_lbn_lst = {}
    for idx, k in enumerate(lbn_lst):
        new_lbn_lst[k] = []
        event = find_event_bgn_end(is_idle[:, idx])
        cur_lbn = lbn_lst[k]
        for bgn, end in event:
            if bgn == end:
                new_lbn_lst[k].append(cur_lbn[bgn])
            else:
                event_len = end - bgn + 1
                last_idx = int(event_len * (1 - last_ratio))
                sel_lbn_range = cur_lbn[bgn + last_idx:end + 1]
                sel_lbn_string = []
                is_basic = len(np.array(sel_lbn_range[0]).shape) == 1
                for i in range(len(sel_lbn_range)):
                    if is_basic:
                        sel_lbn_string.append(f'{sel_lbn_range[i][0]},{sel_lbn_range[i][1]},{sel_lbn_range[i][2]}')
                    else:
                        sel_lbn_string.append(f'{sel_lbn_range[i]}')
                counter = Counter(sel_lbn_string)
                mode = counter.most_common(1)[0][0]
                if is_basic:
                    new_lbn = np.array([float(item) for item in mode.split(',')])
                else:
                    new_lbn = int(mode)
                for i in range(bgn, end + 1):
                    new_lbn_lst[k].append(new_lbn)
    return new_lbn_lst


def lbn_summarize_lr_v2(lbn_lst, is_idle, last_ratio=0.2):
    # 06/07 @ Special handling: both feet idle at the beginning/end forms a special event.
    # Compute LBN using the first 80% / last 80% of that event respectively.
    assert len(lbn_lst) == 2 and is_idle.shape[1] == 2
    new_lbn_lst = {}
    # 1) Original summarization logic
    for idx, k in enumerate(lbn_lst):
        new_lbn_lst[k] = []
        event = find_event_bgn_end(is_idle[:, idx])
        cur_lbn = lbn_lst[k]
        for bgn, end in event:
            if bgn == end:
                new_lbn_lst[k].append(cur_lbn[bgn])
            else:
                event_len = end - bgn + 1
                last_idx = int(event_len * (1 - last_ratio))
                sel_lbn_range = cur_lbn[bgn + last_idx:end + 1]
                sel_lbn_string = []
                is_basic = len(np.array(sel_lbn_range[0]).shape) == 1
                for i in range(len(sel_lbn_range)):
                    if is_basic:
                        sel_lbn_string.append(f'{sel_lbn_range[i][0]},{sel_lbn_range[i][1]},{sel_lbn_range[i][2]}')
                    else:
                        sel_lbn_string.append(f'{sel_lbn_range[i]}')
                counter = Counter(sel_lbn_string)
                mode = counter.most_common(1)[0][0]
                if is_basic:
                    new_lbn = np.array([float(item) for item in mode.split(',')])
                else:
                    new_lbn = int(mode)
                for i in range(bgn, end + 1):
                    new_lbn_lst[k].append(new_lbn)
    # 2) Add special begin/end events
    # Only update when bgn != end.
    both_idle = np.logical_and(is_idle[:, 0], is_idle[:, 1])
    both_event = find_event_bgn_end(both_idle)
    both_event_bgn, both_event_end = both_event[0], both_event[-1]
    for k in lbn_lst:
        cur_lbn = lbn_lst[k]
        # Locate begin event: compute LBN from the first 80%.
        if np.all(both_idle[both_event_bgn[0]:both_event_bgn[1] + 1]):
            bgn, end = both_event_bgn[0], both_event_bgn[1]
            if bgn != end:
                event_len = end - bgn + 1
                last_idx = int(event_len * (1 - last_ratio))
                sel_lbn_range = cur_lbn[bgn:bgn + last_idx + 1]
                sel_lbn_string = []
                is_basic = len(np.array(sel_lbn_range[0]).shape) == 1
                for i in range(len(sel_lbn_range)):
                    if is_basic:
                        sel_lbn_string.append(f'{sel_lbn_range[i][0]},{sel_lbn_range[i][1]},{sel_lbn_range[i][2]}')
                    else:
                        sel_lbn_string.append(f'{sel_lbn_range[i]}')
                counter = Counter(sel_lbn_string)
                mode = counter.most_common(1)[0][0]
                if is_basic:
                    new_lbn = np.array([float(item) for item in mode.split(',')])
                else:
                    new_lbn = int(mode)
                # update
                for i in range(bgn, end + 1):
                    new_lbn_lst[k][i] = new_lbn
        # Locate end event: compute LBN from the last 80%.
        if np.all(both_idle[both_event_end[0]:both_event_end[1] + 1]):
            bgn, end = both_event_end[0], both_event_end[1]
            if bgn != end:
                event_len = end - bgn + 1
                last_idx = int(event_len * (1 - last_ratio))
                sel_lbn_range = cur_lbn[bgn + last_idx:end + 1]
                sel_lbn_string = []
                is_basic = len(np.array(sel_lbn_range[0]).shape) == 1
                for i in range(len(sel_lbn_range)):
                    if is_basic:
                        sel_lbn_string.append(f'{sel_lbn_range[i][0]},{sel_lbn_range[i][1]},{sel_lbn_range[i][2]}')
                    else:
                        sel_lbn_string.append(f'{sel_lbn_range[i]}')
                counter = Counter(sel_lbn_string)
                mode = counter.most_common(1)[0][0]
                if is_basic:
                    new_lbn = np.array([float(item) for item in mode.split(',')])
                else:
                    new_lbn = int(mode)
                # update
                for i in range(bgn, end + 1):
                    new_lbn_lst[k][i] = new_lbn
    return new_lbn_lst


def lbn_summarize_single(lbn_lst, is_idle, last_ratio=0.2):
    assert len(np.array(lbn_lst).shape) == 1 and is_idle.shape[1] == 2
    # <- 06/07 @ fix. true + false = true. should use logical_and
    # 06/09 @ fix: if (True,False) and (False,True) appear, they may be merged into one event.
    new_lbn_lst = []
    event = find_both_event_bgn_end(is_idle)
    for bgn, end in event:
        if bgn == end:
            new_lbn_lst.append(lbn_lst[bgn])
        else:
            event_len = end - bgn + 1
            last_idx = int(event_len * (1 - last_ratio))
            sel_lbn_range = lbn_lst[bgn + last_idx:end + 1]
            sel_lbn_string = []
            for i in range(len(sel_lbn_range)):
                sel_lbn_string.append(f'{sel_lbn_range[i]}')
            counter = Counter(sel_lbn_string)
            mode = counter.most_common(1)[0][0]
            new_lbn = int(mode)
            for i in range(bgn, end + 1):
                new_lbn_lst.append(new_lbn)
    return new_lbn_lst


def fetch_lbn_and_event_all(skels, fps):
    """ Get lbn symbols and events.

    :param skels: T, J==22, 3
    :return: motion sequence-wise annotation
    """
    # canonicalize
    T = len(skels)
    skels_canon = canonicalize_amass_skels(skels, lock_bone='pelvis')
    # draw_seq_3d_in_one_scene(np.concatenate((skels[:, np.newaxis], skels_canon[:, np.newaxis]), axis=1)[::10])

    # get laban symbols and end-effector anchors
    lbn_foot_lr, lbn_root_v, lbn_root_orient, idle_foot_lr = process_lower_motion(skels, skels_canon, fps)
    lbn_wrist_lr, idle_wrist_lr = process_upper_motion(skels_canon)
    lbn_wrist_lr_local = process_upper_motion_local(skels_canon, idle_wrist_lr)
    lbn_head_orient = calc_head_status(skels_canon)
    lbn_bend = calc_endeffector_bend_status(skels_canon,
                                            {
                                                'left_knee': [[jNames['left_ankle'], jNames['left_knee']],
                                                              [jNames['left_knee'], jNames['left_hip']]],
                                                'right_knee': [[jNames['right_ankle'], jNames['right_knee']],
                                                               [jNames['right_knee'], jNames['right_hip']]],
                                                'left_hip': [[jNames['left_knee'], jNames['left_hip']],
                                                             [jNames['left_hip'], jNames['right_hip']]],
                                                'right_hip': [[jNames['right_knee'], jNames['right_hip']],
                                                              [jNames['right_hip'], jNames['left_hip']]],
                                                'left_elbow': [[jNames['left_wrist'], jNames['left_elbow']],
                                                               [jNames['left_elbow'], jNames['left_shoulder']]],
                                                'right_elbow': [[jNames['right_wrist'], jNames['right_elbow']],
                                                                [jNames['right_elbow'], jNames['right_shoulder']]],
                                                'left_shoulder': [[jNames['left_elbow'], jNames['left_shoulder']],
                                                                  [jNames['left_shoulder'], jNames['left_collar']]],
                                                'right_shoulder': [[jNames['right_elbow'], jNames['right_shoulder']],
                                                                   [jNames['right_shoulder'], jNames['right_collar']]],
                                                'spine': [[jNames['pelvis'], jNames['spine1']],
                                                          [jNames['spine1'], jNames['spine3']]],
                                            })

    # summarize
    new_lbn_foot_lr = lbn_summarize_lr_v2(lbn_foot_lr, idle_foot_lr)
    new_lbn_knee_lr = lbn_summarize_lr_v2({'left_knee': lbn_bend['left_knee'],
                                           'right_knee': lbn_bend['right_knee']}, idle_foot_lr)
    new_lbn_hip_lr = lbn_summarize_lr_v2({'left_hip': lbn_bend['left_hip'],
                                          'right_hip': lbn_bend['right_hip']}, idle_foot_lr)
    new_lbn_wrist_lr = lbn_summarize_lr(lbn_wrist_lr, idle_wrist_lr)
    new_lbn_elbow_lr = lbn_summarize_lr({'left_elbow': lbn_bend['left_elbow'],
                                         'right_elbow': lbn_bend['right_elbow']}, idle_wrist_lr)
    new_lbn_shoulder_lr = lbn_summarize_lr({'left_shoulder': lbn_bend['left_shoulder'],
                                            'right_shoulder': lbn_bend['right_shoulder']}, idle_wrist_lr)
    #
    new_lbn_root_v_horz = lbn_summarize_single(lbn_root_v['horz'], idle_foot_lr)
    new_lbn_root_v_vert = lbn_summarize_single(lbn_root_v['vert'], idle_foot_lr)
    new_lbn_root_orient_horz = lbn_summarize_single(lbn_root_orient['horz'], idle_foot_lr)
    new_lbn_root_orient_vert = lbn_summarize_single(lbn_root_orient['vert'], idle_foot_lr)

    # store
    return {'support_l': {'basic': new_lbn_foot_lr['left_foot'],
                          'knee_bend': new_lbn_knee_lr['left_knee'],
                          'hip_bend': new_lbn_hip_lr['left_hip'],
                          'is_idle': idle_foot_lr[:, 0]},
            'support_r': {'basic': new_lbn_foot_lr['right_foot'],
                          'knee_bend': new_lbn_knee_lr['right_knee'],
                          'hip_bend': new_lbn_hip_lr['right_hip'],
                          'is_idle': idle_foot_lr[:, 1]},
            'support_both': {'orient_horz': lbn_root_orient['horz'],
                             'orient_vert': lbn_root_orient['vert'],
                             'velocity_horz': lbn_root_v['horz'],
                             'velocity_vert': lbn_root_v['vert']},
            'upper_l': {'basic': new_lbn_wrist_lr['left_wrist'],
                        'elbow_bend': new_lbn_elbow_lr['left_elbow'],
                        'shoulder_bend': new_lbn_shoulder_lr['left_shoulder'],
                        'is_idle': idle_wrist_lr[:, 0]},
            'upper_r': {'basic': new_lbn_wrist_lr['right_wrist'],
                        'elbow_bend': new_lbn_elbow_lr['right_elbow'],
                        'shoulder_bend': new_lbn_shoulder_lr['right_shoulder'],
                        'is_idle': idle_wrist_lr[:, 1]},
            'torso': {'head_orient_horz': lbn_head_orient['horz'],
                      'head_orient_vert': lbn_head_orient['vert'],
                      'spine': lbn_bend['spine']},
            # 06/07 @ new code: relative wrist laban. to differentiate the waving gesture
            # 06/09 @ fix: bug note - previously fed the same left_wrist values by mistake.
            'upper_l_rel': {'basic': lbn_wrist_lr_local['left_wrist'], },
            'upper_r_rel': {'basic': lbn_wrist_lr_local['right_wrist'], },
            'length': T}
