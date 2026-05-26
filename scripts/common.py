import pickle
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path

from utils.clip_handler import CLIPWrapper
from utils.lbn_utils.lbn_digit_description import readout_lbn_dict_with_digit, convert_back_lbn_dict_with_digit
from utils.lbn_utils.lbn_codebook import convert_back_lbn_codebook

DIR_ROOT = Path.cwd()
print("Current working root:", DIR_ROOT)


def prepare_gen_data(pkl_path, dst_pkl_path):
    """ For generator: append [EOS]
    """
    MAX_FRAME_NUM = 200
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    event_seq_all = []
    for sIdx, lbn_code in enumerate(tqdm(pkl_data)):
        lbn_length = len(lbn_code)
        new_lbn_length = min(lbn_length, MAX_FRAME_NUM - 1)
        # The first N frames are LBN codes. We add one extra dimension for [EOS],
        # and append one extra frame to represent the [EOS] token.
        new_lbn_code = np.zeros((new_lbn_length + 1, 158 + 1), dtype=int)
        new_lbn_code[:new_lbn_length, :158] = lbn_code[:new_lbn_length]
        new_lbn_code[-1, 158] = 1  # [EOS] active
        event_seq_all.append(new_lbn_code)

    with open(dst_pkl_path, 'wb') as handle:
        pickle.dump(event_seq_all, handle, protocol=4)
    return


def to_text_cd(pkl_path, lbn_path, dst_pkl_path):
    """ Convert LabanLite code to conceptual descriptions
    """
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    ids_all = pkl_data['ids']
    texts_all = pkl_data['txts']
    del pkl_data
    with open(lbn_path, 'rb') as f:
        lbn_codes = pickle.load(f)
    total_num = len(ids_all)

    lbn_texts = {}
    for i in tqdm(range(total_num)):
        ids = ids_all[i]
        texts = texts_all[i]
        time_major_dict = {}
        for caption, bgn, end in texts:
            time_gap = f"{int(bgn)}-{int(end)}"
            if time_gap not in time_major_dict:
                time_major_dict[time_gap] = {'caption': [], 'laban_text': None}
            time_major_dict[time_gap]['caption'].append(caption)
            if time_major_dict[time_gap]['laban_text'] is None:
                if bgn == end and bgn == 0.:
                    lbn_code = lbn_codes[i]
                    lbn_dict = convert_back_lbn_codebook(lbn_code)
                    support_text, arm_left_text, arm_right_text = readout_lbn_dict_with_digit(lbn_dict)
                time_major_dict[time_gap]['laban_text'] = {
                    'support': support_text,
                    'arm_left': arm_left_text,
                    'arm_right': arm_right_text,
                }
        # store
        lbn_texts[ids] = time_major_dict

    with open(dst_pkl_path, 'wb') as handle:
        pickle.dump(lbn_texts, handle, protocol=4)
    return


def enc_captions(pkl_path, dst_pkl_path):
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
        ids, txts = pkl_data['ids'], pkl_data['txts']
        ids_sorted, txts_sorted = zip(*(sorted(zip(ids, txts))))
        del pkl_data
    clip_model = CLIPWrapper()
    out_data = {}
    for k, v in tqdm(zip(ids_sorted, txts_sorted)):
        out_data[k] = clip_model.emb_text_np(v[0][0])
    with open(dst_pkl_path, 'wb') as handle:
        pickle.dump(out_data, handle, protocol=4)
    return


def retrieval_topk(query_data_path, query_lbn_path, ref_data_path, ref_lbn_path, dst_path, topk=5):
    from sklearn.metrics.pairwise import cosine_similarity

    with open(query_data_path, 'rb') as f:
        clip_embs = pickle.load(f)
        query_embs = []
        query_names = []
        for k, v in clip_embs.items():
            query_embs.append(v)
            query_names.append(k)
        query_embs = np.vstack(query_embs)
        del clip_embs
        f.close()
    with open(query_lbn_path, 'rb') as f:
        query_lbns = pickle.load(f)
        f.close()
    with open(ref_data_path, 'rb') as f:
        clip_embs = pickle.load(f)
        ref_embs = []
        ref_names = []
        for k, v in clip_embs.items():
            ref_embs.append(v)
            ref_names.append(k)
        ref_embs = np.vstack(ref_embs)
        del clip_embs
        f.close()
    with open(ref_lbn_path, 'rb') as f:
        ref_lbns = pickle.load(f)
        f.close()

    def calc_similarities():
        """Calculate top-k retrieval results based on cosine similarities between query and reference embeddings.
        """
        retrieval_results = {}
        similarities = cosine_similarity(query_embs, ref_embs)
        query_length = len(query_embs)
        for query_idx in range(query_length):
            # Select the top 20 most similar samples by similarity score (descending order)
            most_similar_index = similarities[query_idx].argsort()[-20:][::-1]
            most_similar_names = [ref_names[item] for item in most_similar_index]
            # Check if the query is valid (exists in query_lbns)
            query_name = query_names[query_idx]
            if query_name not in query_lbns:
                print(f"filtered query {query_name}")
                continue
            query_caption = next(iter(query_lbns[query_name].values()))['caption'][0]
            most_similar_laban = []
            for item in most_similar_names:
                if item not in ref_lbns:
                    print(f"filtered reference {item}")
                    continue
                item_dict = next(iter(ref_lbns[item].values()))['laban_text']
                most_similar_caption = next(iter(ref_lbns[item].values()))['caption'][0]
                item_dict['caption'] = most_similar_caption
                most_similar_laban.append(item_dict)
                if len(most_similar_laban) >= topk:
                    break
            assert len(most_similar_laban) == topk
            retrieval_results[query_name] = {'laban': most_similar_laban, 'caption': query_caption}
            # print(f"-> [{query_idx}] {query_name}, cap: {query_caption}")
        return retrieval_results

    retrieval_results = calc_similarities()
    with open(dst_path, 'wb') as handle:
        pickle.dump(retrieval_results, handle, protocol=4)
    return


def back_lbnlite_codes(pkl_path, dst_pkl_path):
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
        lbn_data = {}
        for k, v in pkl_data.items():
            for kk, vv in v.items():
                lbn_data[k] = vv['laban_text']
                continue
        del pkl_data
        f.close()

    lbn_dict = {}
    for k, v in tqdm(lbn_data.items()):
        lbn_dict[k] = convert_back_lbn_dict_with_digit(v)
    with open(dst_pkl_path, 'wb') as handle:
        pickle.dump(lbn_dict, handle, protocol=4)
    return


def prepare_pipeline_llm(name='HML3D'):
    data_root = DIR_ROOT / f'../assets/{name}'
    lbn_root = DIR_ROOT / f'../assets/{name}_lbn'

    # step4: Prepare caption embeddings for similarity calculation
    if not (lbn_root / f'train_text_embs.pkl').exists():
        enc_captions(data_root / 'train.pkl', lbn_root / f'train_text_embs.pkl')
    else:
        print(f"Step4: resume with {lbn_root / f'train_text_embs.pkl'}")
    if not (lbn_root / f'test_text_embs.pkl').exists():
        enc_captions(data_root / 'test.pkl', lbn_root / f'test_text_embs.pkl')
    else:
        print(f"Step4: resume with {lbn_root / f'test_text_embs.pkl'}")

    # step5: Prepare top-k CDs for LLM RAG prompting
    topk = 5
    # train vs train
    if not (lbn_root / f'train_llm_top_{topk}.pkl').exists():
        retrieval_topk(query_data_path=lbn_root / f'train_text_embs.pkl',
                       query_lbn_path=lbn_root / f'train_lbns_cd.pkl',
                       ref_data_path=lbn_root / f'train_text_embs.pkl',
                       ref_lbn_path=lbn_root / f'train_lbns_cd.pkl',
                       dst_path=lbn_root / f'train_llm_top_{topk}.pkl', topk=topk)
    else:
        print(f"Step5: resume with {lbn_root / f'train_llm_top_{topk}.pkl'}")
    # test vs train
    if not (lbn_root / f'test_llm_top_{topk}.pkl').exists():
        retrieval_topk(query_data_path=lbn_root / f'test_text_embs.pkl',
                       query_lbn_path=lbn_root / f'test_lbns_cd.pkl',
                       ref_data_path=lbn_root / f'train_text_embs.pkl',
                       ref_lbn_path=lbn_root / f'train_lbns_cd.pkl',
                       dst_path=lbn_root / f'test_llm_top_{topk}.pkl', topk=topk)
    else:
        print(f"Step5: resume with {lbn_root / f'test_llm_top_{topk}.pkl'}")

    # step6.1: Convert CDs back to LabanLite codes
    if not (lbn_root / f'train_llm_codes.pkl').exists():
        back_lbnlite_codes(lbn_root / 'train_lbns_cd.pkl', lbn_root / f'train_llm_codes.pkl')
    else:
        print(f"Step6.1: resume with {lbn_root / f'train_llm_codes.pkl'}")
    if not (lbn_root / f'test_llm_codes.pkl').exists():
        back_lbnlite_codes(lbn_root / 'test_lbns_cd.pkl', lbn_root / f'test_llm_codes.pkl')
    else:
        print(f"Step6.1: resume with {lbn_root / f'test_llm_codes.pkl'}")
    return
