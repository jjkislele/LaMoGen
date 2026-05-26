from pathlib import Path
import numpy as np
import pickle
from tqdm.auto import tqdm

from common.quaternion import *
from original_files.paramUtil import *
from common.skeleton import Skeleton

###################
# Configs
###################
# Filter sequences by length following the paper's setting.
# Also save text tokens.
# IMPORTANT: follow the train/test/val split order.

YOUR_PATH_joint_vecs = 'workspace/HumanML3D/new_joint_vecs'
YOUR_PATH_joints = 'workspace/HumanML3D/new_joints'
YOUR_PATH_texts = 'workspace/HumanML3D/texts'
YOUR_PATH_data_split_root = 'workspace/HumanML3D/HumanML3D'
HML3D_root = '../../assets/HML3D'

feats_lst = {item.stem: item for item in Path(YOUR_PATH_joint_vecs).rglob("*.npy")}
kpts_lst = {item.stem: item for item in Path(YOUR_PATH_joints).rglob("*.npy")}
texts_lst = {item.stem: item for item in Path(YOUR_PATH_texts).rglob("*.txt")}

n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain
# Get offsets of target skeleton
example_data = np.load(f'{YOUR_PATH_joints}/000021.npy')
example_data = example_data.reshape(len(example_data), -1, 3)
example_data = torch.from_numpy(example_data)
tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
# (joints_num, 3)
tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 22


def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()
    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt
    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def process_file(positions):
    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)
    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz
    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    positions_b = positions.copy()
    positions = qrot_np(root_quat_init, positions)
    '''New ground truth positions'''
    global_positions = positions.copy()
    return global_positions


def store_in_pkl(txt_path, dst_pkl_path):
    with open(txt_path, 'r') as file:
        idx_lst = file.readlines()
    idx_lst = [item.strip('\n') for item in idx_lst]

    sel_ids = []
    sel_feats = []
    sel_kpts = []
    sel_txts = []
    sel_tkns = []

    for idx in tqdm(idx_lst):
        # # debug
        # if 'M002611' not in idx:
        #     continue
        feat_path = feats_lst[idx]
        kpt_path = kpts_lst[idx]
        text_path = texts_lst[idx]
        feats = np.load(feat_path, allow_pickle=True)  # 263-dim feature (HumanML3D representation)
        kpts = np.load(kpt_path, allow_pickle=True)  # keypoint locations recovered from the 263-dim feature
        txts = []
        tkns = []
        T = len(feats)
        if T < 40 or T >= 200:
            continue

        # norm kpts
        norm_kpts = process_file(kpts.copy())

        with open(text_path, 'r') as f:
            for line in f.readlines():
                line_split = line.strip().split('#')
                caption = line_split[0]
                tokens = line_split[1].split(' ')
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                f_tag, to_tag = f_tag * 20, to_tag * 20

                if f_tag == 0.0 and to_tag == 0.0:
                    txts.append([caption, f_tag, to_tag])
                    tkns.append(tokens)
                else:
                    new_T = to_tag - f_tag
                    if new_T < 40 or T >= 200:
                        continue
                    else:
                        txts.append([caption, f_tag, to_tag])
                        tkns.append(tokens)

        sel_feats.append(feats)
        sel_kpts.append(norm_kpts)
        sel_txts.append(txts)
        sel_tkns.append(tkns)
        sel_ids.append(idx)

    with open(dst_pkl_path, 'wb') as handle:
        pickle.dump({'ids': sel_ids, 'feats': sel_feats, 'kpts': sel_kpts, 'txts': sel_txts, 'tkns': sel_tkns}, handle,
                    protocol=4)


store_in_pkl(f'{YOUR_PATH_data_split_root}/train.txt',
             f'{HML3D_root}/train.pkl')
store_in_pkl(f'{YOUR_PATH_data_split_root}/test.txt',
             f'{HML3D_root}/test.pkl')
store_in_pkl(f'{YOUR_PATH_data_split_root}/val.txt',
             f'{HML3D_root}/val.pkl')
print("Done!")
