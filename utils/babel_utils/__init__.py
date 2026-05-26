from typing import List, Tuple
import json
import numpy as np

from .spell import fix_spell

LBN_EXCLUDED_ACTIONS = (['t-pose', 'a-pose', 'a pose', 't pose', 'tpose', 'apose', 'transition', 'stair'] +
                        ['t-pose', 'a-pose', 'a pose', 't pose', 'tpose', 'apose', 'transition', 'stair',
                         'in place', 'lie', 'lay'])
LBN_INCLUDED_ACTIONS = ['walk', 'run', 'step'] + ['walk', 'step'] + \
                       ['jump', 'hop', 'skip', 'leap'] + \
                       ['run', 'sprint', 'jog']


def read_json(p):
    with open(p, 'r') as fp:
        json_contents = json.load(fp)
    return json_contents


def segments_sorted(segs_fr: List[List], acts: List) -> Tuple[List[List], List]:
    assert len(segs_fr) == len(acts)
    if len(segs_fr) == 1: return segs_fr, acts
    L = [(segs_fr[i], i) for i in range(len(segs_fr))]
    L.sort()
    sorted_segs_fr, permutation = zip(*L)
    sort_acts = [acts[i] for i in permutation]

    return list(sorted_segs_fr), sort_acts


def extract_frame_labels(babel_labels, fps, seqlen, local_include=None, local_exclude=None, include_all_flg=False):
    if local_include is None:
        local_include = LBN_INCLUDED_ACTIONS
    if local_exclude is None:
        local_exclude = LBN_EXCLUDED_ACTIONS
    seg_ids = []
    seg_acts = []
    is_valid = True
    if babel_labels['frame_ann'] is None:
        # Whole sequence has the same label
        action_label = babel_labels['seq_ann']['labels'][0]['proc_label']
        seg_ids.append((0, seqlen))
        seg_acts.append(fix_spell(action_label))
    else:
        # Get segments
        for seg_an in babel_labels['frame_ann']['labels']:
            action_label = fix_spell(seg_an['proc_label'])
            st_f = int(seg_an['start_t'] * fps)
            end_f = int(seg_an['end_t'] * fps)
            if st_f > end_f:
                st_f, end_f = end_f, st_f
            if end_f > seqlen:
                end_f = seqlen
            seg_ids.append((st_f, end_f))
            seg_acts.append(action_label)
        # Process segments
        assert len(seg_ids) == len(seg_acts)
        seg_ids, seg_acts = segments_sorted(seg_ids, seg_acts)
    # filter a/t pose for pair calculation
    idx_to_keep = []
    # # debug
    # seg_acts = ['walk forward', 'transition', 'sit', 'transition', 'walk back']
    # seg_acts = ['lie in prone position', 'lay']
    for i, a in enumerate(seg_acts):
        # remove
        remove_flg = np.any([act in a for act in local_exclude])
        if remove_flg:
            continue
        # keep
        keep_flg = np.any([act in a for act in local_include])
        if keep_flg or include_all_flg:
            idx_to_keep.append(i)
    seg_acts_for_pairs = [a for i, a in enumerate(seg_acts) if i in idx_to_keep]
    seg_ids_for_pairs = [s for i, s in enumerate(seg_ids) if i in idx_to_keep]
    assert len(seg_acts_for_pairs) == len(seg_ids_for_pairs)
    if len(seg_acts_for_pairs) == 0:
        print(f"jumped {babel_labels['feat_p']} due to filtering {seg_acts}")
        return None, None, False
    else:
        print(f"-> keep actions: {seg_acts_for_pairs} from {seg_acts}")
    return seg_acts_for_pairs, seg_ids_for_pairs, is_valid
