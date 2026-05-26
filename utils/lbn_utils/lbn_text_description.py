import numpy as np

from utils.lbn_utils.mot2lbn import find_both_event_bgn_end, find_event_bgn_end

# Ordering note:
# support_words_dict_move, support_words_dict_hold
# arm_words_dict_move, arm_words_dict_hold, arm_words_dict_related

support_words_dict_move = {
    # mid level
    '0,0,0': 'steps to rest position',
    '0,1,0': 'steps forward',
    '0,-1,0': 'steps backward',
    '1,0,0': 'steps to right',
    '-1,0,0': 'steps to left',
    '1,1,0': 'steps forward diagonally to right',
    '-1,1,0': 'steps forward diagonally to left',
    '1,-1,0': 'steps backward diagonally to right',
    '-1,-1,0': 'steps backward diagonally to left',
    # high level
    '0,0,1': 'rises',
    '0,1,1': 'rises to forward',
    '0,-1,1': 'rises to backward',
    '1,0,1': 'rises to right',
    '-1,0,1': 'rises to left',
    '1,1,1': 'rises forward diagonally to right',
    '-1,1,1': 'rises forward diagonally to left',
    '1,-1,1': 'rises backward diagonally to right',
    '-1,-1,1': 'rises backward diagonally to left',
    # low level
    '0,0,-1': 'knee flex',
    '0,1,-1': 'knee flex forward',
    '0,-1,-1': 'knee flex backward',
    '1,0,-1': 'knee flex right',
    '-1,0,-1': 'knee flex left',
    '1,1,-1': 'knee flex forward diagonally to right',
    '-1,1,-1': 'knee flex forward diagonally to left',
    '1,-1,-1': 'knee flex backward diagonally to right',
    '-1,-1,-1': 'knee flex backward diagonally to left',
}

support_words_dict_hold = {
    # mid level
    '0,0,0': 'holds in rest position',
    '0,1,0': 'holds in forward position',
    '0,-1,0': 'holds in backward position',
    '1,0,0': 'holds in right position',
    '-1,0,0': 'holds in left position',
    '1,1,0': 'holds in forward diagonally to right position',
    '-1,1,0': 'holds in forward diagonally to left position',
    '1,-1,0': 'holds in backward diagonally to right position',
    '-1,-1,0': 'holds in backward diagonally to left position',
    # high level
    '0,0,1': 'holds in the raised position',
    '0,1,1': 'holds in the raised forward position',
    '0,-1,1': 'holds in the raised backward position',
    '1,0,1': 'holds in the raised right position',
    '-1,0,1': 'holds in the raised left position',
    '1,1,1': 'holds in the raised forward diagonally to right position',
    '-1,1,1': 'holds in the raised forward diagonally to left position',
    '1,-1,1': 'holds in the raised backward diagonally to right position',
    '-1,-1,1': 'holds in the raised backward diagonally to left position',
    # low level
    '0,0,-1': 'holds in knee-flexed position',
    '0,1,-1': 'holds in knee-flexed forward position',
    '0,-1,-1': 'holds in knee-flexed backward position',
    '1,0,-1': 'holds in knee-flexed right position',
    '-1,0,-1': 'holds in knee-flexed left position',
    '1,1,-1': 'holds in knee-flexed forward diagonally to right position',
    '-1,1,-1': 'holds in knee-flexed forward diagonally to left position',
    '1,-1,-1': 'holds in knee-flexed backward diagonally to right position',
    '-1,-1,-1': 'holds in knee-flexed backward diagonally to left position',
}

arm_words_dict_move = {
    # mid level
    '0,0,0': 'moves close to shoulder',
    '0,1,0': 'moves forward',
    '0,-1,0': 'moves backward',
    '1,0,0': 'moves to right',
    '-1,0,0': 'moves to left',
    '1,1,0': 'moves forward diagonally to right',
    '-1,1,0': 'moves forward diagonally to left',
    '1,-1,0': 'moves backward diagonally to right',
    '-1,-1,0': 'moves backward diagonally to left',
    # high level
    '0,0,1': 'rises up',
    '0,1,1': 'rises to up forward',
    '0,-1,1': 'rises to up backward',
    '1,0,1': 'rises to up right',
    '-1,0,1': 'rises to up left',
    '1,1,1': 'rises up forward diagonally to right',
    '-1,1,1': 'rises up forward diagonally to left',
    '1,-1,1': 'rises up backward diagonally to right',
    '-1,-1,1': 'rises up backward diagonally to left',
    # low level
    '0,0,-1': 'lowers down',
    '0,1,-1': 'lowers to down forward',
    '0,-1,-1': 'lowers to down backward',
    '1,0,-1': 'lowers to down right',
    '-1,0,-1': 'lowers to down left',
    '1,1,-1': 'lowers down forward diagonally to right',
    '-1,1,-1': 'lowers down forward diagonally to left',
    '1,-1,-1': 'lowers down backward diagonally to right',
    '-1,-1,-1': 'lowers down backward diagonally to left',
}

arm_words_dict_hold = {
    # mid level
    '0,0,0': 'holds close to shoulder position',
    '0,1,0': 'holds forward position',
    '0,-1,0': 'holds backward position',
    '1,0,0': 'holds right position',
    '-1,0,0': 'holds left position',
    '1,1,0': 'holds forward diagonally to right position',
    '-1,1,0': 'holds forward diagonally to left position',
    '1,-1,0': 'holds backward diagonally to right position',
    '-1,-1,0': 'holds backward diagonally to left position',
    # high level
    '0,0,1': 'holds up position',
    '0,1,1': 'holds up forward position',
    '0,-1,1': 'holds up backward position',
    '1,0,1': 'holds up right position',
    '-1,0,1': 'holds up left position',
    '1,1,1': 'holds up forward diagonally to right position',
    '-1,1,1': 'holds up forward diagonally to left position',
    '1,-1,1': 'holds up backward diagonally to right position',
    '-1,-1,1': 'holds up backward diagonally to left position',
    # low level
    '0,0,-1': 'holds low position',
    '0,1,-1': 'holds low forward position',
    '0,-1,-1': 'holds low backward position',
    '1,0,-1': 'holds low right position',
    '-1,0,-1': 'holds low left position',
    '1,1,-1': 'holds low forward diagonally to right position',
    '-1,1,-1': 'holds low forward diagonally to left position',
    '1,-1,-1': 'holds low backward diagonally to right position',
    '-1,-1,-1': 'holds low backward diagonally to left position',
}

arm_words_dict_related = {
    # mid level
    '0,0,0': 'moves relatively to previous position',
    '0,1,0': 'moves relatively forward',
    '0,-1,0': 'moves relatively backward',
    '1,0,0': 'moves to relatively right',
    '-1,0,0': 'moves to relatively left',
    '1,1,0': 'moves relatively forward diagonally to right',
    '-1,1,0': 'moves relatively forward diagonally to left',
    '1,-1,0': 'moves relatively backward diagonally to right',
    '-1,-1,0': 'moves relatively backward diagonally to left',
    # high level
    '0,0,1': 'moves relatively up',
    '0,1,1': 'moves relatively up forward',
    '0,-1,1': 'moves relatively up backward',
    '1,0,1': 'moves relatively up right',
    '-1,0,1': 'moves relatively up left',
    '1,1,1': 'moves relatively up forward diagonally to right',
    '-1,1,1': 'moves relatively up forward diagonally to left',
    '1,-1,1': 'moves relatively up backward diagonally to right',
    '-1,-1,1': 'moves relatively up backward diagonally to left',
    # low level
    '0,0,-1': 'moves relatively low',
    '0,1,-1': 'moves relatively low forward',
    '0,-1,-1': 'moves relatively low backward',
    '1,0,-1': 'moves relatively low right',
    '-1,0,-1': 'moves relatively low left',
    '1,1,-1': 'moves relatively low forward diagonally to right',
    '-1,1,-1': 'moves relatively low forward diagonally to left',
    '1,-1,-1': 'moves relatively low backward diagonally to right',
    '-1,-1,-1': 'moves relatively low backward diagonally to left',
}


def lbn_basic_words_support(lbn_basic, sec, part, is_idle):
    lbn_basic_str = f'{int(lbn_basic[0])},{int(lbn_basic[1])},{int(lbn_basic[2])}'
    if is_idle:
        sentence = f"{part} foot {support_words_dict_hold[lbn_basic_str]} in {sec} seconds"
    else:
        sentence = f"{part} foot {support_words_dict_move[lbn_basic_str]} in {sec} seconds"
    return sentence


def lbn_basic_words_arm(lbn_basic, lbn_related, sec, part, is_idle):
    lbn_basic_str = f'{int(lbn_basic[0])},{int(lbn_basic[1])},{int(lbn_basic[2])}'
    lbn_related_str = f'{int(lbn_related[0])},{int(lbn_related[1])},{int(lbn_related[2])}'
    if is_idle:
        sentence = f"{part} hand {arm_words_dict_hold[lbn_basic_str]} in {sec} seconds"
    else:
        sentence = f"{part} hand {arm_words_dict_move[lbn_basic_str]} in {sec} seconds"
    sentence_rel = f"{part} hand {arm_words_dict_related[lbn_related_str]} in {sec} seconds"
    return sentence, sentence_rel


def lbn_bend_words_spine(lbn_bend, sec, part):
    lbn_bend_str = f'{lbn_bend}'
    words_dict = {
        '0': 'stretches straight',
        '1': 'bends to 150 degrees',
        '2': 'bends to 120 degrees',
        '3': 'bends to 90 degrees',
        '4': 'bends to 60 degrees',
        '5': 'bends to 30 degrees',
    }
    sentence = f"{part} {words_dict[lbn_bend_str]} in {sec} seconds"
    return sentence


def lbn_orient_words_head(lbn_orient_horz, lbn_orient_vert, sec, part):
    words_dict_horz = {
        0: 'face forward',
        1: 'face forward diagonally to right',
        2: 'face right',
        3: 'face backward diagonally to right',
        4: 'face backward',
        5: 'face backward diagonally to left',
        6: 'face left',
        7: 'face forward diagonally to left'
    }
    words_dict_vert = {
        0: 'face straight up',
        1: 'face diagonally upward',
        2: 'face forward',
        3: 'face diagonally downward',
        4: 'face downward',
        5: 'face diagonally backward',
        6: 'face backward',
        7: 'face diagonally backward and upward'
    }
    if lbn_orient_horz == 0 and lbn_orient_vert == 2:
        sentence = f"{part} turns to {words_dict_horz[lbn_orient_horz]} in {sec} seconds"
    else:
        sentence = f"{part} turns to {words_dict_horz[lbn_orient_horz]} and turns to {words_dict_vert[lbn_orient_vert]} in {sec} seconds"
    return sentence


def readout_lbn_dict_with_text(lbn_dict, fps=20.):
    def process_support():
        # Left/right feet must be processed jointly (check both feet status per event).
        support_l = lbn_dict['support_l']
        support_r = lbn_dict['support_r']
        sup_basic_l = support_l['basic']
        sup_basic_r = support_r['basic']
        sup_basic_flg = np.concatenate((support_l['is_idle'][..., None], support_r['is_idle'][..., None]), axis=1)
        # 06/09 @ fix: if (True,False) and (False,True) appear, they may be merged into one event.
        sup_event = find_both_event_bgn_end(sup_basic_flg)
        sup_text_lst = []
        for bgn, end in sup_event:
            sec = (end + 1 - bgn) * 1 / fps
            cur_sup_l = sup_basic_l[bgn]
            cur_sup_r = sup_basic_r[bgn]
            cur_sup_l_flg = sup_basic_flg[bgn, 0]
            cur_sup_r_flg = sup_basic_flg[bgn, 1]
            cur_sup_l_txt = lbn_basic_words_support(cur_sup_l, sec, 'left', cur_sup_l_flg)
            cur_sup_r_txt = lbn_basic_words_support(cur_sup_r, sec, 'right', cur_sup_r_flg)
            if sec < 0.1:
                continue
            # Left foot idle, right foot active -> keep right-foot event only.
            if cur_sup_l_flg and ~cur_sup_r_flg:
                sup_text_lst.append(cur_sup_r_txt)
            # Left foot active, right foot idle -> keep left-foot event only.
            elif ~cur_sup_l_flg and cur_sup_r_flg:
                sup_text_lst.append(cur_sup_l_txt)
            # Both feet idle -> keep both states.
            elif cur_sup_l_flg and cur_sup_r_flg:
                sup_text_lst.append(f"{cur_sup_l_txt} while {cur_sup_r_txt}")
        return sup_text_lst

    def process_upper(part, fps=20.):
        # Left/right hands are processed separately.
        assert 'left' == part or 'right' == part, f'unknown arm part {part}'
        basic_key = 'upper_l' if part == 'left' else 'upper_r'
        upper = lbn_dict[basic_key]
        arm_basic = upper['basic']
        arm_related = lbn_dict[f'{basic_key}_rel']['basic']
        arm_idle = upper['is_idle']
        arm_event = find_event_bgn_end(arm_idle)
        arm_text_lst = []
        for eIdx, (bgn, end) in enumerate(arm_event):
            sec = (end + 1 - bgn) * 1 / fps
            cur_basic = arm_basic[bgn]
            cur_related = arm_related[bgn]
            cur_is_idle = arm_idle[bgn]
            # TODO: "is_idle" definition is opposite to the lower-body one (legacy behavior); keep as-is.
            cur_arm_txt, cur_arm_rel_txt = lbn_basic_words_arm(lbn_basic=cur_basic, lbn_related=cur_related,
                                                               sec=sec, part=part, is_idle=~cur_is_idle)
            if len(arm_text_lst) == 0:
                # Initialize
                arm_text_lst.append(cur_arm_txt)
            else:
                # If the global state equals the previous one, use relative movement to describe it.
                # Filter out events shorter than 0.1s.
                previous_basic = arm_basic[arm_event[eIdx - 1][0]]
                if np.all(cur_basic == previous_basic) and sec >= 0.1:
                    arm_text_lst.append(cur_arm_rel_txt)
                elif sec >= 0.1:
                    arm_text_lst.append(cur_arm_txt)
        return arm_text_lst

    def process_spine(fps=20.):
        # Summarize spine bending states.
        bend_stats = lbn_dict['torso']['spine']
        event_bgn = 0
        event_end = 1
        event_lst = []
        while event_end < len(bend_stats):
            if bend_stats[event_bgn] != bend_stats[event_end]:
                event_lst.append([event_bgn, event_end - 1])
                event_bgn = event_end
                event_end += 1
            else:
                event_end += 1
        # append last
        if event_bgn < len(bend_stats):
            assert np.all(np.diff(bend_stats[event_bgn:len(bend_stats)], axis=0) == 0)
            event_lst.append([event_bgn, len(bend_stats) - 1])
        # to text
        bend_text_lst = []
        for eIdx, (bgn, end) in enumerate(event_lst):
            sec = (end + 1 - bgn) * 1 / fps
            cur_bend_txt = lbn_bend_words_spine(bend_stats[bgn], sec=sec, part='spine')
            if len(bend_text_lst) == 0:
                # Initialize
                bend_text_lst.append(cur_bend_txt)
            elif sec >= 0.1:
                bend_text_lst.append(cur_bend_txt)
        return bend_text_lst

    def process_head(fps=20.):
        # Summarize head orientation.
        head_orient_horz = np.array(lbn_dict['torso']['head_orient_horz'])
        head_orient_vert = np.array(lbn_dict['torso']['head_orient_vert'])
        head_orient = np.concatenate((head_orient_horz[:, None], head_orient_vert[:, None]), axis=1)
        event_lst = find_both_event_bgn_end(head_orient)
        # to text
        orient_text_lst = []
        for eIdx, (bgn, end) in enumerate(event_lst):
            sec = (end + 1 - bgn) * 1 / fps
            cur_orient_txt = lbn_orient_words_head(lbn_orient_horz=head_orient_horz[bgn],
                                                   lbn_orient_vert=head_orient_vert[bgn],
                                                   sec=sec, part='head')
            if len(orient_text_lst) == 0:
                # Initialize
                orient_text_lst.append(cur_orient_txt)
            elif sec >= 0.1:
                orient_text_lst.append(cur_orient_txt)
        return orient_text_lst

    # support
    sup_text_lst = process_support()
    whole_sup_text = ', '.join(sup_text_lst)
    # upper
    upper_l_text_lst = process_upper('left')
    upper_r_text_lst = process_upper('right')
    whole_arm_l_text = ', '.join(upper_l_text_lst)
    whole_arm_r_text = ', '.join(upper_r_text_lst)
    # # spine bend
    # spine_text = process_spine()
    # # head
    # head_text = process_head()
    return whole_sup_text, whole_arm_l_text, whole_arm_r_text
