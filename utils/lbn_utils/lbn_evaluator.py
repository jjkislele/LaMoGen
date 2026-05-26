import pickle
import numpy as np
import torch

from utils.hml3d_utils import recover_from_ric
from utils.lbn_utils.lbn_codebook import convert_back_lbn_codebook
from utils.lbn_utils.mot2lbn import fetch_lbn_and_event_all
from utils.lbn_utils.mot2lbn import find_event_bgn_end

symbol_map = {-1: 'a', 0: 'b', 1: 'c'}


def find_lcseque(s1, s2):
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    s = []
    while m[p1][p2]:  # while not zero
        c = d[p1][p2]
        if c == 'ok':  # matched: take char and move to upper-left
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  # move left
            p2 -= 1
        if c == 'up':  # move up
            p1 -= 1
    s.reverse()
    match_rate = len(s) / max(len(s1), len(s2))
    return match_rate


def calc_semantic_alignment(pred_symbols, gt_symbols):
    def map_to_str(lbns_int):
        lbns_str = []
        for item in lbns_int:
            chars = ''
            for v in item:
                chars += symbol_map[v]
            lbns_str.append(chars)
        return lbns_str

    pred_symbols = np.array(pred_symbols)
    gt_symbols = np.array(gt_symbols)
    pred_str = [' '.join(map(str, item)) for item in pred_symbols]
    pred_event_lst = find_event_bgn_end(pred_str)
    gt_str = [' '.join(map(str, item)) for item in gt_symbols]
    gt_event_lst = find_event_bgn_end(gt_str)

    # semantic alignment
    pred_semantics = [pred_symbols[item[0]] for item in pred_event_lst]
    pred_semantics = map_to_str(pred_semantics)
    gt_semantics = [gt_symbols[item[0]] for item in gt_event_lst]
    gt_semantics = map_to_str(gt_semantics)
    align_ratio = find_lcseque(pred_semantics, gt_semantics)

    return align_ratio


def calc_time_alignment(pred_symbols, pred_flg, gt_symbols, gt_flg):
    def map_to_str(lbns_int, lbn_flgs):
        lbns_str = []
        for item, flg in zip(lbns_int, lbn_flgs):
            chars = ''
            for v in item:
                chars += symbol_map[v]
            chars += 'a' if flg else 'b'
            lbns_str.append(chars)
        return lbns_str

    pred_symbols = np.array(pred_symbols)
    gt_symbols = np.array(gt_symbols)
    pred_str = map_to_str(pred_symbols, pred_flg)
    gt_str = map_to_str(gt_symbols, gt_flg)
    align_ratio = find_lcseque(gt_str, pred_str)
    return align_ratio


def calc_harmony_alignment(pred_dict, gt_dict, pair_list, ratio=0.5):
    def map_to_str(ref_lbn, tar_lbn=None):
        ref_str = ''
        for v in ref_lbn:
            ref_str += symbol_map[v]
        if tar_lbn is not None:
            ref_str += '_'
            for v in tar_lbn:
                ref_str += symbol_map[v]
        return ref_str

    def get_active_event_pairs(ref_lbns, ref_flgs, tar_lbns, tar_flgs):
        ref_event = find_event_bgn_end(ref_flgs)
        ref_str = []
        for ref_bgn, ref_end in ref_event:
            if ref_flgs[ref_bgn]:
                continue
            # active event: is_idle = False
            tar_cur_flg = tar_flgs[ref_bgn:ref_end + 1]
            tar_act_count = np.sum(tar_cur_flg == False)
            ref_act_count = ref_end - ref_bgn + 1
            ref_cur_lbn = ref_lbns[ref_bgn]
            if tar_act_count / ref_act_count > ratio:
                tar_cur_lbn = tar_lbns[np.where(tar_cur_flg == False)[0][0]]
                ref_str.append(map_to_str(ref_cur_lbn, tar_cur_lbn))
            else:
                ref_str.append(map_to_str(ref_cur_lbn, None))
        return ref_str

    pair_score = {}
    for ref_pair, tar_pair in pair_list:
        # prediction
        pred_str = get_active_event_pairs(
            ref_lbns=pred_dict[ref_pair]['basic'], ref_flgs=pred_dict[ref_pair]['is_idle'],
            tar_lbns=pred_dict[tar_pair]['basic'], tar_flgs=pred_dict[tar_pair]['is_idle'])
        # gt
        gt_str = get_active_event_pairs(
            ref_lbns=gt_dict[ref_pair]['basic'], ref_flgs=gt_dict[ref_pair]['is_idle'],
            tar_lbns=gt_dict[tar_pair]['basic'], tar_flgs=gt_dict[tar_pair]['is_idle'])
        # find lcs
        if len(pred_str) == 0:
            pred_str = 'empty'
        if len(gt_str) == 0:
            gt_str = 'empty'
        align_ratio = find_lcseque(gt_str, pred_str)
        pair_score[f'{ref_pair} v.s. {tar_pair}'] = align_ratio
    return pair_score


class Evaluator:
    def __init__(self, pkl_feat, pkl_lbn, **kwargs):
        with open(pkl_feat, 'rb') as f:
            pkl_data = pickle.load(f)
        self.names = pkl_data['ids']
        self.kpts = pkl_data['kpts']
        del pkl_data
        with open(pkl_lbn, 'rb') as f:
            self.lbns = pickle.load(f)
        self.mean = np.load(kwargs.get('mean_path', None))
        self.std = np.load(kwargs.get('std_path', None))
        #
        self.smt_supp_l = []
        self.smt_supp_r = []
        self.smt_arm_l = []
        self.smt_arm_r = []
        #
        self.tm_supp_l = []
        self.tm_supp_r = []
        self.tm_arm_l = []
        self.tm_arm_r = []
        #
        self.hm_basic = {'upper_l v.s. upper_r': [],
                         'upper_l v.s. support_l': [],
                         'upper_l v.s. support_r': [],
                         'upper_r v.s. support_l': [],
                         'upper_r v.s. support_r': [],
                         'support_l v.s. support_r': []}

    def debug(self, pred_kpt, sel_idx):
        from utils.skel_painter import draw_seq_3d_in_one_scene
        from utils.lbn_utils.lbn_codebook import convert_to_lbn_codebook
        gt_kpt = self.kpts[sel_idx]
        gt_kpt[:, :, [0, 1, 2]] = gt_kpt[:, :, [0, 2, 1]]
        gt_kpt[:, :, 1] = -gt_kpt[:, :, 1]
        min_len = min(len(gt_kpt), len(pred_kpt))
        draw_seq_3d_in_one_scene(np.concatenate([gt_kpt[:min_len, None, ...], pred_kpt[:min_len, None, ...]],
                                                axis=1)[::20])
        gt_dict_ = fetch_lbn_and_event_all(gt_kpt, 20.)
        gt_lbn_ = convert_to_lbn_codebook(gt_dict_)
        gt_lbn = self.lbns[sel_idx]
        is_equal_lbn_codebook = np.all(gt_lbn_ == gt_lbn)
        assert is_equal_lbn_codebook
        return

    def detect_lbn_symbols(self, feats):
        # recover feat263 to position
        feats = feats * self.std + self.mean
        kpts = recover_from_ric(torch.from_numpy(feats).float(), 22).numpy()
        # to correct xyz coordinate
        kpts[:, :, [0, 1, 2]] = kpts[:, :, [0, 2, 1]]
        kpts[:, :, 1] = -kpts[:, :, 1]
        # to lbn
        lbn_dict = fetch_lbn_and_event_all(kpts, fps=20.)
        return lbn_dict

    def __call__(self, pred_feat, pred_id, **kwargs):
        """
        :param pred_feat: T, 263
        :param pred_id: string. HumanML3D's motion id
        """
        pred_lbn_dict = kwargs.get('pred_lbn_dict', None)
        if pred_lbn_dict is None:
            pred_lbn_dict = self.detect_lbn_symbols(pred_feat)
        # retrieve gt
        sel_idx = kwargs.get('sel_idx', None)
        if sel_idx is None:
            try:
                sel_idx = self.names.index(pred_id)
            except Exception as e:
                print(e)
                return
        gt_lbn = self.lbns[sel_idx]
        gt_lbn_dict = convert_back_lbn_codebook(gt_lbn)
        # # debug
        # self.debug(pred_kpt, sel_idx)

        # semantic alignment
        basic_supp_l = calc_semantic_alignment(pred_lbn_dict['support_l']['basic'], gt_lbn_dict['support_l']['basic'])
        basic_supp_r = calc_semantic_alignment(pred_lbn_dict['support_r']['basic'], gt_lbn_dict['support_r']['basic'])
        basic_arm_l = calc_semantic_alignment(pred_lbn_dict['upper_l']['basic'], gt_lbn_dict['upper_l']['basic'])
        basic_arm_r = calc_semantic_alignment(pred_lbn_dict['upper_r']['basic'], gt_lbn_dict['upper_r']['basic'])
        self.smt_supp_l.append(basic_supp_l)
        self.smt_supp_r.append(basic_supp_r)
        self.smt_arm_l.append(basic_arm_l)
        self.smt_arm_r.append(basic_arm_r)
        # time alignment
        basic_supp_l = calc_time_alignment(pred_lbn_dict['support_l']['basic'], pred_lbn_dict['support_l']['is_idle'],
                                           gt_lbn_dict['support_l']['basic'], gt_lbn_dict['support_l']['is_idle'])
        basic_supp_r = calc_time_alignment(pred_lbn_dict['support_r']['basic'], pred_lbn_dict['support_r']['is_idle'],
                                           gt_lbn_dict['support_r']['basic'], gt_lbn_dict['support_r']['is_idle'])
        basic_arm_l = calc_time_alignment(pred_lbn_dict['upper_l']['basic'], pred_lbn_dict['upper_l']['is_idle'],
                                          gt_lbn_dict['upper_l']['basic'], gt_lbn_dict['upper_l']['is_idle'])
        basic_arm_r = calc_time_alignment(pred_lbn_dict['upper_r']['basic'], pred_lbn_dict['upper_r']['is_idle'],
                                          gt_lbn_dict['upper_r']['basic'], gt_lbn_dict['upper_r']['is_idle'])
        self.tm_supp_l.append(basic_supp_l)
        self.tm_supp_r.append(basic_supp_r)
        self.tm_arm_l.append(basic_arm_l)
        self.tm_arm_r.append(basic_arm_r)
        # harmony alignment
        harmony_basic = calc_harmony_alignment(pred_lbn_dict, gt_lbn_dict, [['upper_l', 'upper_r'],
                                                                            ['upper_l', 'support_l'],
                                                                            ['upper_l', 'support_r'],
                                                                            ['upper_r', 'support_l'],
                                                                            ['upper_r', 'support_r'],
                                                                            ['support_l', 'support_r']])
        for k, v in harmony_basic.items():
            self.hm_basic[k].append(v)
        return

    def readout(self):
        stat_lines = []
        stat_lines.append(f"===\n{len(self.smt_supp_l)}\n===")
        stat_lines.append(f'smt_supp_l: {np.average(self.smt_supp_l)}')
        stat_lines.append(f'smt_supp_r: {np.average(self.smt_supp_l)}')
        stat_lines.append(f'smt_arm_l: {np.average(self.smt_arm_l)}')
        stat_lines.append(f'smt_arm_r: {np.average(self.smt_arm_r)}')
        stat_lines.append(f'tm_supp_l: {np.average(self.tm_supp_l)}')
        stat_lines.append(f'tm_supp_r: {np.average(self.tm_supp_l)}')
        stat_lines.append(f'tm_arm_l: {np.average(self.tm_arm_l)}')
        stat_lines.append(f'tm_arm_r: {np.average(self.tm_arm_r)}')
        for k, v in self.hm_basic.items():
            stat_lines.append(f'{k}: {np.average(v)}')
        for lines in stat_lines:
            print(lines)
        return stat_lines

    def fetch_statistics(self):
        stat = {
            'smt_supp_l': self.smt_supp_l,
            'smt_supp_r': self.smt_supp_l,
            'smt_arm_l': self.smt_arm_l,
            'smt_arm_r': self.smt_arm_r,
            'tm_supp_l': self.tm_supp_l,
            'tm_supp_r': self.tm_supp_l,
            'tm_arm_l': self.tm_arm_l,
            'tm_arm_r': self.tm_arm_r,
        }
        return stat

    def clean(self):
        self.smt_supp_l = []
        self.smt_supp_r = []
        self.smt_arm_l = []
        self.smt_arm_r = []


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="LabanLite evaluator (utilities subset).")
    parser.add_argument("--pkl_feat", type=str, required=True, help="Path to features pkl (e.g., HumanML3D *.pkl)")
    parser.add_argument("--pkl_lbn", type=str, required=True, help="Path to LabanLite codebook pkl (*_lbns_158.pkl)")
    parser.add_argument("--mean_path", type=str, required=True, help="Path to mean.npy")
    parser.add_argument("--std_path", type=str, required=True, help="Path to std.npy")
    parser.add_argument("--pred_feat", type=str, required=True, help="Path to predicted feature file (.npy)")
    parser.add_argument("--pred_id", type=str, required=True, help="Motion id (string)")
    args = parser.parse_args()

    my_eval = Evaluator(
        pkl_feat=args.pkl_feat,
        pkl_lbn=args.pkl_lbn,
        mean_path=args.mean_path,
        std_path=args.std_path,
    )
    pred_feat = np.load(args.pred_feat)
    my_eval(pred_feat, args.pred_id)
    my_eval.readout()
    my_eval.clean()
