import copy
import re
import warnings

import numpy as np

from utils.lbn_utils.lbn_codebook import codebook_range
from utils.lbn_utils.lbn_text_description import arm_words_dict_hold, arm_words_dict_move, arm_words_dict_related
from utils.lbn_utils.lbn_text_description import support_words_dict_move, support_words_dict_hold
from utils.lbn_utils.mot2lbn import find_both_event_bgn_end, find_event_bgn_end

# Ordering / indexing note:
# support_words_dict_move, support_words_dict_hold
# arm_words_dict_move, arm_words_dict_hold, arm_words_dict_related
#
support_words_dict_move_digit = {}
idx = 1
for k, v in support_words_dict_move.items():
    support_words_dict_move_digit[k] = [v, idx]
    idx += 1
#
support_words_dict_hold_digit = {}
for k, v in support_words_dict_hold.items():
    support_words_dict_hold_digit[k] = [v, idx]
    idx += 1
#
arm_words_dict_move_digit = {}
idx = 1
for k, v in arm_words_dict_move.items():
    arm_words_dict_move_digit[k] = [v, idx]
    idx += 1
#
arm_words_dict_hold_digit = {}
for k, v in arm_words_dict_hold.items():
    arm_words_dict_hold_digit[k] = [v, idx]
    idx += 1
#
arm_words_dict_related_digit = {}
for k, v in arm_words_dict_related.items():
    arm_words_dict_related_digit[k] = [v, idx]
    idx += 1


def lbn_basic_words_support(lbn_basic, sec, part, is_idle):
    lbn_basic_str = f'{int(lbn_basic[0])},{int(lbn_basic[1])},{int(lbn_basic[2])}'
    if is_idle:
        sentence = f"({part}, {support_words_dict_hold_digit[lbn_basic_str][1]}, {sec})"
    else:
        sentence = f"({part}, {support_words_dict_move_digit[lbn_basic_str][1]}, {sec})"
    return sentence


def lbn_basic_words_arm(lbn_basic, lbn_related, sec, part, is_idle):
    lbn_basic_str = f'{int(lbn_basic[0])},{int(lbn_basic[1])},{int(lbn_basic[2])}'
    lbn_related_str = f'{int(lbn_related[0])},{int(lbn_related[1])},{int(lbn_related[2])}'
    if is_idle:
        sentence = f"({arm_words_dict_hold_digit[lbn_basic_str][1]}, {sec})"
    else:
        sentence = f"({arm_words_dict_move_digit[lbn_basic_str][1]}, {sec})"
    sentence_rel = f"({arm_words_dict_related_digit[lbn_related_str][1]}, {sec})"
    return sentence, sentence_rel


def readout_lbn_dict_with_digit(lbn_dict, fps=20.):
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
            # [fix] Keep short events; filtering them may drop many frames.
            # if sec < 0.1:
            #     continue
            # Left foot idle, right foot active -> keep right-foot event only.
            if cur_sup_l_flg and ~cur_sup_r_flg:
                sup_text_lst.append(cur_sup_r_txt)
            # Left foot active, right foot idle -> keep left-foot event only.
            elif ~cur_sup_l_flg and cur_sup_r_flg:
                sup_text_lst.append(cur_sup_l_txt)
            # Both feet idle -> keep both states.
            elif cur_sup_l_flg and cur_sup_r_flg:
                sup_text_lst.append(f"{cur_sup_l_txt} while {cur_sup_r_txt}")
            # [fix] 06/27 @ Both feet active.
            else:
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
            # TODO: "is_idle" definition is opposite to the lower-body one (legacy behavior):
            # - lower body: True = idle, False = moving
            # - upper body: True = moving, False = idle
            # Therefore we pass ~cur_is_idle here.
            cur_arm_txt, cur_arm_rel_txt = lbn_basic_words_arm(lbn_basic=cur_basic, lbn_related=cur_related,
                                                               sec=sec, part=part, is_idle=~cur_is_idle)
            if len(arm_text_lst) == 0:
                # Initialize
                arm_text_lst.append(cur_arm_txt)
            else:
                # If the global state equals the previous one, use relative movement to describe it.
                # (Originally intended to filter out events shorter than 0.1s; now kept without filtering.)
                previous_basic = arm_basic[arm_event[eIdx - 1][0]]
                # Rollback the duration filtering (keep all events).
                # if np.all(cur_basic == previous_basic) and sec >= 0.1:
                #     arm_text_lst.append(cur_arm_rel_txt)
                # elif sec >= 0.1:
                #     arm_text_lst.append(cur_arm_txt)
                if np.all(cur_basic == previous_basic):
                    arm_text_lst.append(cur_arm_rel_txt)
                else:
                    arm_text_lst.append(cur_arm_txt)
        return arm_text_lst

    # support
    sup_text_lst = process_support()
    whole_sup_text = ', '.join(sup_text_lst)
    # upper
    upper_l_text_lst = process_upper('left')
    upper_r_text_lst = process_upper('right')
    whole_arm_l_text = ', '.join(upper_l_text_lst)
    whole_arm_r_text = ', '.join(upper_r_text_lst)
    return whole_sup_text, whole_arm_l_text, whole_arm_r_text


# Convert digit-form text back to LBN numeric codes.
# NOTE (support): True = idle, False = moving.
support_word_to_digit = {}
for k, v in support_words_dict_move_digit.items():
    k = np.array([int(item) for item in k.split(',')])
    support_word_to_digit[v[1]] = [k, False]
for k, v in support_words_dict_hold_digit.items():
    k = np.array([int(item) for item in k.split(',')])
    support_word_to_digit[v[1]] = [k, True]
# NOTE (arm): True = moving, False = idle.
arm_word_to_digit = {}
for k, v in arm_words_dict_move_digit.items():
    k = np.array([int(item) for item in k.split(',')])
    arm_word_to_digit[v[1]] = [k, True]
for k, v in arm_words_dict_hold_digit.items():
    k = np.array([int(item) for item in k.split(',')])
    arm_word_to_digit[v[1]] = [k, False]
# relative movement
for k, v in arm_words_dict_related_digit.items():
    k = np.array([int(item) for item in k.split(',')])
    arm_word_to_digit[v[1]] = [k, 'rel']


def convert_back_lbn_dict_with_digit(text_dict, llm_compose=False):
    """
    :param text_dict: {'support': str, 'arm_left': str, 'arm_right': str}
    :param llm_compose: boolean, True when the input is produced by an LLM (may contain mistakes).
    :return:
    """

    def parse_event_support(event_str):
        output = re.findall(pattern, event_str)
        if len(output) != 3:
            raise Exception(f"ERROR <parse_event_support> {output}")
        else:
            part, part_id, sec = output
        part_id = int(part_id)
        sec = float(sec) * 20.
        return part, part_id, sec

    def parse_event_arm(event_str):
        output = re.findall(pattern, event_str)
        if len(output) != 2:
            raise Exception(f"ERROR <parse_event_arm> {output}")
        else:
            part_id, sec = output
        part_id = int(part_id)
        sec = float(sec) * 20.
        return part_id, sec

    def str_to_list_support(string):
        list = re.split(r"\),\s*\(", string)
        list_output = []
        event_accm = 0.
        for v in list:
            # Both-feet events are represented with "while".
            vv = re.split(r"\)\swhile\s\(", v)
            if len(vv) > 1:
                part1, part_id1, sec1 = parse_event_support(vv[0])
                part2, part_id2, sec2 = parse_event_support(vv[1])
                # assert sec1 == sec2
                sec = min(sec1, sec2)
                list_output.append([part1, part_id1, part2, part_id2, event_accm, event_accm + sec - 1])
            # Otherwise it is a single-foot event (the other foot is active).
            else:
                part, part_id, sec = parse_event_support(v)
                list_output.append([part, part_id, event_accm, event_accm + sec - 1])
            event_accm += sec
        return list_output, event_accm

    def str_to_list_arm(string):
        list = re.split(r"\),\s*\(", string)
        list_output = []
        event_accm = 0.
        for v in list:
            # If the global state equals the previous one, relative movement may be used for describing it.
            # (Originally intended to filter out events shorter than 0.1s; currently not enforced here.)
            part_id, sec = parse_event_arm(v)
            list_output.append([part_id, event_accm, event_accm + sec - 1])
            event_accm += sec
        return list_output, event_accm

    def get_is_idle_np(is_idle):
        """One-hot encoding: index-0 -> False, index-1 -> True.
        """
        is_idle_np = np.zeros(2)
        if is_idle:
            is_idle_np[1] = 1
        else:
            is_idle_np[0] = 1
        return is_idle_np

    pattern = r'[a-zA-Z_]+|\d+\.\d+(?:e[+-]?\d+)?|\d+e[+-]?\d+|\d+'
    support, arm_left, arm_right = text_dict['support'], text_dict['arm_left'], text_dict['arm_right']
    support, support_num = str_to_list_support(support)
    arm_left, arm_left_num = str_to_list_arm(arm_left)
    arm_right, arm_right_num = str_to_list_arm(arm_right)
    support_num, arm_left_num, arm_right_num = int(support_num), int(arm_left_num), int(arm_right_num)
    total_frame_num = max(support_num, arm_left_num, arm_right_num)
    # warn if too large
    ratio = np.array([support_num, arm_left_num, arm_right_num]) / total_frame_num
    if np.any(ratio) < 0.9:
        warnings.warn(ratio, UserWarning)

    lbn_code = np.zeros((total_frame_num, 158), dtype=int)
    # Support (feet)
    for event in support:
        # Single-foot event (mutually exclusive)
        if len(event) == 4:
            part, part_id, bgn, end = event
            bgn, end = int(bgn), int(end)
            lbn_symbol, is_idle = copy.deepcopy(support_word_to_digit[part_id])
            if 'left' == part:
                body_key_1, body_key_2 = 'support_l', 'support_r'
            else:
                body_key_1, body_key_2 = 'support_r', 'support_l'
            is_idle_np = get_is_idle_np(is_idle)
            # Assign to the primary part.
            cur_range = codebook_range[body_key_1]['is_idle']
            lbn_code[bgn:end + 1, cur_range[0]:cur_range[1]] = is_idle_np
            lbn_symbol = lbn_symbol + 1 + np.array([0, 3, 6])  # move [-1, 0, 1] to [0, 1, 2] and side, fwd, level
            lbn_symbol += codebook_range[body_key_1]['basic'][0]  # add stride
            lbn_code[bgn:end + 1, lbn_symbol] = 1
            # Assign to the opposite part.
            cur_range = codebook_range[body_key_2]['is_idle']
            lbn_code[bgn:end + 1, cur_range[0]:cur_range[1]] = np.abs(is_idle_np - 1)
        # Both-feet event
        elif len(event) == 6:
            part1, part_id1, part2, part_id2, bgn, end = event
            bgn, end = int(bgn), int(end)
            lbn_symbol1, is_idle1 = copy.deepcopy(support_word_to_digit[part_id1])
            lbn_symbol1 = lbn_symbol1 + 1 + np.array([0, 3, 6])  # move [-1, 0, 1] to [0, 1, 2] and side, fwd, level
            lbn_symbol1 += codebook_range['support_l']['basic'][0]  # add stride
            lbn_symbol2, is_idle2 = copy.deepcopy(support_word_to_digit[part_id2])
            lbn_symbol2 = lbn_symbol2 + 1 + np.array([0, 3, 6])  # move [-1, 0, 1] to [0, 1, 2] and side, fwd, level
            lbn_symbol2 += codebook_range['support_r']['basic'][0]  # add stride
            # assert 'left' == part1 and 'right' == part2  # LLM may output (left,right) in different order.
            # assert is_idle1 == is_idle2  # They can differ (e.g., both feet off-ground). [fix] 07/02 @ do not enforce.
            # store
            is_idle_np = get_is_idle_np(is_idle1)
            cur_range = codebook_range['support_l']['is_idle']
            lbn_code[bgn:end + 1, cur_range[0]:cur_range[1]] = is_idle_np
            cur_range = codebook_range['support_r']['is_idle']
            lbn_code[bgn:end + 1, cur_range[0]:cur_range[1]] = is_idle_np
            lbn_code[bgn:end + 1, lbn_symbol1] = 1
            lbn_code[bgn:end + 1, lbn_symbol2] = 1
    # Arms
    for body_name, body_event in zip(['l', 'r'], [arm_left, arm_right]):
        pre_lbn_symbol_basic = None
        for event in body_event:
            part_id, bgn, end = event
            bgn, end = int(bgn), int(end)
            lbn_symbol, is_idle_or_ref = copy.deepcopy(arm_word_to_digit[part_id])
            lbn_symbol = lbn_symbol + 1 + np.array([0, 3, 6])  # move [-1, 0, 1] to [0, 1, 2] and side, fwd, level
            # If the global state equals the previous one, fill using relative movement
            # (duration filtering was planned, but currently not enforced).
            if 'rel' == is_idle_or_ref:
                # Handle relative (rel) movement.
                # If the rel symbol is neutral, it indicates the basic arm state is idle.
                # NOTE: is_idle here follows the upper-body convention.
                is_idle = True  # moving
                if np.all(lbn_symbol == np.array([1, 4, 7])):
                    is_idle = False  # idle
                lbn_symbol += codebook_range[f'upper_{body_name}_rel']['basic'][0]  # add stride
                lbn_code[bgn:end + 1, lbn_symbol] = 1
                # Handle basic
                is_idle_np = get_is_idle_np(is_idle)
                cur_range = codebook_range[f'upper_{body_name}']['is_idle']
                lbn_code[bgn:end + 1, cur_range[0]:cur_range[1]] = is_idle_np
                # Repeat previous basic state.
                if pre_lbn_symbol_basic is not None:
                    lbn_code[bgn:end + 1, pre_lbn_symbol_basic] = 1
            else:
                # Handle basic
                is_idle_np_basic = get_is_idle_np(is_idle_or_ref)
                lbn_symbol += codebook_range[f'upper_{body_name}']['basic'][0]  # add stride
                lbn_code[bgn:end + 1, lbn_symbol] = 1
                cur_range = codebook_range[f'upper_{body_name}']['is_idle']
                lbn_code[bgn:end + 1, cur_range[0]:cur_range[1]] = is_idle_np_basic
                pre_lbn_symbol_basic = lbn_symbol.copy()
                # Handle rel
                if is_idle_or_ref:  # moving
                    # When moving, the rel symbol should be close to basic in principle.
                    # 06/27 @ Leave it empty for now.
                    continue
                    # lbn_symbol_rel = lbn_symbol.copy()
                    # lbn_symbol_rel = lbn_symbol_rel - codebook_range[f'upper_{body_name}']['basic'][0] + \
                    #                  codebook_range[f'upper_{body_name}_rel']['basic'][0]
                    # lbn_code[bgn:end + 1, lbn_symbol_rel] = 1
                else:  # idle
                    # If idle and no rel is given, rel defaults to neutral.
                    lbn_symbol_rel = np.array([1, 4, 7]) + codebook_range[f'upper_{body_name}_rel']['basic'][0]
                    lbn_code[bgn:end + 1, lbn_symbol_rel] = 1
    # Special handling for support: when one side is missing, fill it using the next available symbol.
    if not llm_compose:
        for i in range(len(lbn_code), 1, -1):
            cur_range = codebook_range['support_l']['basic']
            cur_supp_l = lbn_code[i - 1, cur_range[0]:cur_range[1]]
            next_supp_l = lbn_code[i - 2, cur_range[0]:cur_range[1]]
            if np.all(next_supp_l == 0):
                lbn_code[i - 2, cur_range[0]:cur_range[1]] = cur_supp_l
            #
            cur_range = codebook_range['support_r']['basic']
            cur_supp_r = lbn_code[i - 1, cur_range[0]:cur_range[1]]
            next_supp_r = lbn_code[i - 2, cur_range[0]:cur_range[1]]
            if np.all(next_supp_r == 0):
                lbn_code[i - 2, cur_range[0]:cur_range[1]] = cur_supp_r
    return lbn_code
