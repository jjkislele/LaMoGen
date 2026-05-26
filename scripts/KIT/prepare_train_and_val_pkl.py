from pathlib import Path
import numpy as np
import pickle
from tqdm.auto import tqdm

YOUR_PATH_joint_vecs = 'workspace/KIT-ML/new_joint_vecs'
YOUR_PATH_joints = 'workspace/KIT-ML/new_joints'
YOUR_PATH_texts = 'workspace/KIT-ML/texts'
YOUR_PATH_data_split_root = 'workspace/KIT-ML'
KIT_root = '../../assets/KIT'

feats_lst = {item.stem: item for item in Path(YOUR_PATH_joint_vecs).rglob("*.npy")}
kpts_lst = {item.stem: item for item in Path(YOUR_PATH_joints).rglob("*.npy")}
texts_lst = {item.stem: item for item in Path(YOUR_PATH_texts).rglob("*.txt")}


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
        try:
            # # debug
            # if 'M002611' not in idx:
            #     continue
            feat_path = feats_lst[idx]
            kpt_path = kpts_lst[idx]
            text_path = texts_lst[idx]
            feats = np.load(feat_path, allow_pickle=True)
            kpts = np.load(kpt_path, allow_pickle=True)
            txts = []
            tkns = []
            T = len(feats)
            if T < 40 or T >= 200:
                continue

            # normalization
            # NOTE: kpts 与 norm_kpts 近似
            # norm_kpts = process_file(kpts)

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
        except Exception as e:
            print(f"Error: {e}")

        sel_feats.append(feats)
        sel_kpts.append(kpts)
        sel_txts.append(txts)
        sel_tkns.append(tkns)
        sel_ids.append(idx)

    with open(dst_pkl_path, 'wb') as handle:
        pickle.dump({'ids': sel_ids, 'feats': sel_feats, 'kpts': sel_kpts, 'txts': sel_txts, 'tkns': sel_tkns}, handle,
                    protocol=4)


store_in_pkl(f'{YOUR_PATH_data_split_root}/train.txt',
             f'{KIT_root}/train.pkl')
store_in_pkl(f'{YOUR_PATH_data_split_root}/test.txt',
             f'{KIT_root}/test.pkl')
