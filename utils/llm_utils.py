from tqdm.auto import tqdm
import pickle
from pathlib import Path
import re

from utils.lbn_utils.lbn_digit_description import convert_back_lbn_dict_with_digit


def parse_llm_batch_lbn(batch_root, llm_compose=False):
    """ convert CDs back to LabanLite codes
    """
    batch_root = Path(batch_root)
    pkl_path_lst = [str(item) for item in batch_root.rglob('*.pkl')]
    out_path = str(batch_root) + f'_lbn_codes.pkl'
    pkl_data = {}
    for file_path in pkl_path_lst:
        with open(file_path, 'rb') as f:
            cur_pkl = pickle.load(f)
            for k, v in cur_pkl.items():
                new_v = v['laban'][0] if isinstance(v['laban'], (list, tuple)) else None
                if k not in pkl_data:
                    pkl_data[k] = new_v
                else:
                    if '_' in k:
                        i = int(k.split('_')[1])
                        i += 1
                        new_k = f"{k}_{i}"
                        while new_k in pkl_data:
                            i += 1
                            new_k = f"{k}_{i}"
                    else:
                        i = 1
                        new_k = f"{k}_{i}"
                        while new_k in pkl_data:
                            i += 1
                            new_k = f"{k}_{i}"
                    pkl_data[new_k] = new_v
            f.close()
    #
    lbn_results = []
    names = []
    for k, v in pkl_data.items():
        lbn_results.append(v)
        names.append(k)
    ########################
    label_pattern = r"(\[?\s*(support|left hand|right hand)\s*\]?)\s*[:-]?\s*"
    pattern = label_pattern + r"((?:.|\n)*?)(?=(\[?\s*(support|left hand|right hand)\s*\]?\s*[:-]?|$))"
    lbn_codes = {}
    for lIdx, line in enumerate(tqdm(lbn_results)):
        try:
            ########################
            matches = re.findall(pattern, line, re.IGNORECASE)
            lbn_data = {}
            for full_label, label, tuples, *_ in matches:
                if 'support' in label.lower():
                    lbn_data['support'] = tuples
                elif 'left' in label.lower():
                    lbn_data['arm_left'] = tuples
                elif 'right' in label.lower():
                    lbn_data['arm_right'] = tuples
            lbn_code = convert_back_lbn_dict_with_digit(lbn_data, llm_compose)
            lbn_codes[names[lIdx]] = lbn_code
        except Exception as e:
            print(f"[Parse error]: [{lIdx}]: {e}")

    with open(out_path, 'wb') as handle:
        pickle.dump(lbn_codes, handle, protocol=4)
    print(f"saved to {out_path}")
