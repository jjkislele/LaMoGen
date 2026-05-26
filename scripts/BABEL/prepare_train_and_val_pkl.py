from tqdm.auto import tqdm
import copy
import pickle
from pathlib import Path
import joblib

from utils.smpl_utils.aug import norm_amass_motion, flip_amass_motion
from utils.babel_utils import read_json, extract_frame_labels
from scripts.HumanML3D.prepare_train_and_val_pkl import process_file
from scripts.BABEL.smpl_model import SMPLHandler


def babel_data_loader(dType='val_tiny'):
    mocap_path = Path(YOUR_PATH_BABEL_root) / f'{dType}.pth.tar'
    anno_path = Path(YOUR_PATH_BABEL_root) / f"{dType.split('_')[0]}.json"
    mocap_data = joblib.load(mocap_path)
    anno_data = read_json(anno_path)
    return mocap_data, anno_data


def prepare_babel_npz_norm_and_flip(dst_pkl_path, dType):
    mocap_data, anno_data = babel_data_loader(dType)
    ###################################
    # loop all mocap samples
    mocap_data_norm = []
    mocap_data_flip = []
    for i, sample in enumerate(mocap_data):
        babel_id = sample['babel_id']
        print(f"-> [#{i}] {babel_id} processing...")
        # NOTE: no hand
        cur_len = len(sample['poses'])
        cur_fps = copy.deepcopy(sample['fps'])
        poses = (copy.deepcopy(sample['poses'])).reshape(cur_len, -1, 3)[:, :22]
        trans = copy.deepcopy(sample['trans'])
        # process
        poses_norm, trans_norm = norm_amass_motion(poses, trans)
        poses_flip, trans_flip = flip_amass_motion(poses_norm, trans_norm, is_norm=True)
        # store
        mocap_data_norm.append({'poses': poses_norm, 'trans': trans_norm, 'fps': cur_fps,
                                'fname': sample['fname'], 'babel_id': sample['babel_id']})
        mocap_data_flip.append({'poses': poses_flip, 'trans': trans_flip, 'fps': cur_fps,
                                'fname': sample['fname'], 'babel_id': sample['babel_id']})

    # store in pkl
    dst_pkl_path = Path(dst_pkl_path)
    with open(dst_pkl_path / f"{dType}.n.pkl", 'wb') as handle:
        pickle.dump(mocap_data_norm, handle, protocol=4)
    with open(dst_pkl_path / f"{dType}.f.pkl", 'wb') as handle:
        pickle.dump(mocap_data_flip, handle, protocol=4)
    return


def process_with_pickle(pkl_path, dtype):
    src_pkls = [f'{pkl_path}/{dtype}.f.pkl',
                f'{pkl_path}/{dtype}.n.pkl', ]
    out_pkl_path = f'{LOCO_root}/{dtype}.pkl'
    anno_json = read_json(f'{LOCO_root}/{dtype}.json')

    pkl_data = []
    for pkl_path in src_pkls:
        pkl_path = Path(pkl_path)
        with open(pkl_path, 'rb') as f:
            pkl_data += pickle.load(f)

    seg_pkl = []
    for i, sample in enumerate(tqdm(pkl_data)):
        # # debug
        # sample = pkl_data[1886]
        babel_id = sample['babel_id']
        print(f"-> [#{i}] {babel_id} processing...")
        cur_fps = sample['fps']
        cur_len = len(sample['poses'])
        babel_label = anno_json[babel_id]
        seg_acts, seg_ids, is_valid = extract_frame_labels(
            babel_label, cur_fps, cur_len,
        )
        if not is_valid:
            continue
        for seg_id, seg_act in zip(seg_ids, seg_acts):
            cur_poses = sample['poses'][seg_id[0]:min(seg_id[1] + 1, cur_len)]
            cur_trans = sample['trans'][seg_id[0]:min(seg_id[1] + 1, cur_len)]
            cur_seg_len = len(cur_trans)
            # jump if not enough frame
            if cur_seg_len < 40 or cur_seg_len > 200:
                print(f"Jump due to length: {seg_act} - {cur_seg_len}")
                continue
            #
            cur_seg = {
                'poses': cur_poses,
                'trans': cur_trans,
                'fps': cur_fps,
                'fname': sample['fname'],
                'babel_id': sample['babel_id'],
                'babel_seg': seg_id,
                'babel_sname': seg_act,
            }
            seg_pkl.append(cur_seg)
        # break
    ###################################
    sel_ids = []
    sel_feats = []
    sel_kpts = []
    sel_txts = []
    for seg_data in tqdm(seg_pkl):
        poses, trans, babel_id, babel_label = \
            seg_data['poses'], seg_data['trans'], seg_data['babel_id'], seg_data['babel_sname']
        # trans: T, 3
        # poses: T, 22, 3
        cur_poses = poses.reshape(-1, 66)
        kpts = smpl_handler.amass_to_pose(poses=cur_poses[:, :66], trans=trans)
        kpts = kpts[:, :22]
        # flip
        kpts[:, [2, 1, 5, 4, 8, 7, 11, 10, 14, 13, 17, 16, 19, 18, 21, 20]] = \
            kpts[:, [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21]]
        data, ground_positions, _, _ = process_file(kpts, 0.002)  # T, 263
        # store
        sel_ids.append(babel_id)
        sel_feats.append(data)
        sel_kpts.append(ground_positions)
        sel_txts.append(babel_label)
    with open(out_pkl_path, 'wb') as handle:
        pickle.dump({'ids': sel_ids, 'feats': sel_feats, 'kpts': sel_kpts, 'txts': sel_txts}, handle, protocol=4)


# BABEL-TEACH data root (download by yourself)
YOUR_PATH_BABEL_root = 'workspace/babel_teach'
# HumanML3D's SMPL model root (same to HumanML3D's instructions)
SMPL_root = 'workspace/HumanML3D/body_models'
# Laban Benchmark's root (output root)
LOCO_root = '../../assets/LOCO'

smpl_handler = SMPLHandler(SMPL_root)
# step1. Augment BABEL actions
prepare_babel_npz_norm_and_flip(YOUR_PATH_BABEL_root, 'val')
prepare_babel_npz_norm_and_flip(YOUR_PATH_BABEL_root, 'train')

# step2. Convert motion into HumanML3D format
process_with_pickle(YOUR_PATH_BABEL_root, 'val')
process_with_pickle(YOUR_PATH_BABEL_root, 'train')
print("Done!")
