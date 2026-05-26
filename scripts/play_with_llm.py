from pathlib import Path
import time
import argparse
from tqdm.auto import tqdm
import numpy as np
import pickle
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

from utils.llm_utils import parse_llm_batch_lbn

DIR_ROOT = Path.cwd()
print("Current working root:", DIR_ROOT)


def get_llm_output_compose(cap, lbns, seconds=None, client=None, model_name=None, ip_only_flg=False):
    def give_a_post():
        time.sleep(0.1)
        if seconds is not None:
            dev_prompt = "There are five digit collections describing movements, where each line consists of: [number] [Caption] - a general description of the motion sequence. [Support] - detailed descriptions of the movements of the supporting body parts, specifically the left and right feet, using series of triplets. [Left hand] - detailed descriptions of the movements of the left hand, using series of tuples. [Right hand] - detailed descriptions of the movements of the right hand, using series of tuples. In the detailed descriptions, we specify the movement details for each body part and their duration in seconds. For the support movements, the details must be selected from these 54 categories: [1: steps to rest position, 2: steps forward, 3: steps backward, 4: steps to right, 5: steps to left, 6: steps forward diagonally to right, 7: steps forward diagonally to left, 8: steps backward diagonally to right, 9: steps backward diagonally to left, 10: rises, 11: rises to forward, 12: rises to backward, 13: rises to right, 14: rises to left, 15: rises forward diagonally to right, 16: rises forward diagonally to left, 17: rises backward diagonally to right, 18: rises backward diagonally to left, 19: knee flex, 20: knee flex forward, 21: knee flex backward, 22: knee flex right, 23: knee flex left, 24: knee flex forward diagonally to right, 25: knee flex forward diagonally to left, 26: knee flex backward diagonally to right, 27: knee flex backward diagonally to left, 28: holds in rest position, 29: holds in forward position, 30: holds in backward position, 31: holds in right position, 32: holds in left position, 33: holds in forward diagonally to right position, 34: holds in forward diagonally to left position, 35: holds in backward diagonally to right position, 36: holds in backward diagonally to left position, 37: holds in the raised position, 38: holds in the raised forward position, 39: holds in the raised backward position, 40: holds in the raised right position, 41: holds in the raised left position, 42: holds in the raised forward diagonally to right position, 43: holds in the raised forward diagonally to left position, 44: holds in the raised backward diagonally to right position, 45: holds in the raised backward diagonally to left position, 46: holds in knee-flexed position, 47: holds in knee-flexed forward position, 48: holds in knee-flexed backward position, 49: holds in knee-flexed right position, 50: holds in knee-flexed left position, 51: holds in knee-flexed forward diagonally to right position, 52: holds in knee-flexed forward diagonally to left position, 53: holds in knee-flexed backward diagonally to right position, 54: holds in knee-flexed backward diagonally to left position]. For the hand movements, the details must be selected from these 81 categories: [1: moves close to shoulder, 2: moves forward, 3: moves backward, 4: moves to right, 5: moves to left, 6: moves forward diagonally to right, 7: moves forward diagonally to left, 8: moves backward diagonally to right, 9: moves backward diagonally to left, 10: rises up, 11: rises to up forward, 12: rises to up backward, 13: rises to up right, 14: rises to up left, 15: rises up forward diagonally to right, 16: rises up forward diagonally to left, 17: rises up backward diagonally to right, 18: rises up backward diagonally to left, 19: lowers down, 20: lowers to down forward, 21: lowers to down backward, 22: lowers to down right, 23: lowers to down left, 24: lowers down forward diagonally to right, 25: lowers down forward diagonally to left, 26: lowers down backward diagonally to right, 27: lowers down backward diagonally to left, 28: holds close to shoulder position, 29: holds forward position, 30: holds backward position, 31: holds right position, 32: holds left position, 33: holds forward diagonally to right position, 34: holds forward diagonally to left position, 35: holds backward diagonally to right position, 36: holds backward diagonally to left position, 37: holds up position, 38: holds up forward position, 39: holds up backward position, 40: holds up right position, 41: holds up left position, 42: holds up forward diagonally to right position, 43: holds up forward diagonally to left position, 44: holds up backward diagonally to right position, 45: holds up backward diagonally to left position, 46: holds low position, 47: holds low forward position, 48: holds low backward position, 49: holds low right position, 50: holds low left position, 51: holds low forward diagonally to right position, 52: holds low forward diagonally to left position, 53: holds low backward diagonally to right position, 54: holds low backward diagonally to left position, 55: moves relatively to previous position, 56: moves relatively forward, 57: moves relatively backward, 58: moves to relatively right, 59: moves to relatively left, 60: moves relatively forward diagonally to right, 61: moves relatively forward diagonally to left, 62: moves relatively backward diagonally to right, 63: moves relatively backward diagonally to left, 64: moves relatively up, 65: moves relatively up forward, 66: moves relatively up backward, 67: moves relatively up right, 68: moves relatively up left, 69: moves relatively up forward diagonally to right, 70: moves relatively up forward diagonally to left, 71: moves relatively up backward diagonally to right, 72: moves relatively up backward diagonally to left, 73: moves relatively low, 74: moves relatively low forward, 75: moves relatively low backward, 76: moves relatively low right, 77: moves relatively low left, 78: moves relatively low forward diagonally to right, 79: moves relatively low forward diagonally to left, 80: moves relatively low backward diagonally to right, 81: moves relatively low backward diagonally to left]. For example, for the [Support] line, the triplet list would be like: (left, 1, 0.25), (right, 2, 0.25), (left, 1, 0.25) while (right, 2, 0.25). This means that the first movement is \"left foot steps to rest position in 0.25 seconds\". The second movement is \"right foot steps forward in 0.25 seconds\". The third movement is \"left foot steps to rest position in 0.25 seconds while right foot steps forward in 0.25 seconds. For the [Left hand] line, the tuple list would be like: (1, 0.5), (2, 0.2). This means that the first movement is \"left hand moves close to shoulder in 0.5 seconds\" and the second movement is \"left hand moves forward in 0.2 seconds\". For the [Right hand] line, the structure and definition is similar to [Left hand] lines. Below is the main body of the digit collection describing the movements.\nYou should strictly imitate the following content and create only one digit collection of '{}' movement which lasts in {} seconds. Reply without explaination.\n".format(
                cap, seconds)
        else:
            dev_prompt = "There are five digit collections describing movements, where each line consists of: [number] [Caption] - a general description of the motion sequence. [Support] - detailed descriptions of the movements of the supporting body parts, specifically the left and right feet, using series of triplets. [Left hand] - detailed descriptions of the movements of the left hand, using series of tuples. [Right hand] - detailed descriptions of the movements of the right hand, using series of tuples. In the detailed descriptions, we specify the movement details for each body part and their duration in seconds. For the support movements, the details must be selected from these 54 categories: [1: steps to rest position, 2: steps forward, 3: steps backward, 4: steps to right, 5: steps to left, 6: steps forward diagonally to right, 7: steps forward diagonally to left, 8: steps backward diagonally to right, 9: steps backward diagonally to left, 10: rises, 11: rises to forward, 12: rises to backward, 13: rises to right, 14: rises to left, 15: rises forward diagonally to right, 16: rises forward diagonally to left, 17: rises backward diagonally to right, 18: rises backward diagonally to left, 19: knee flex, 20: knee flex forward, 21: knee flex backward, 22: knee flex right, 23: knee flex left, 24: knee flex forward diagonally to right, 25: knee flex forward diagonally to left, 26: knee flex backward diagonally to right, 27: knee flex backward diagonally to left, 28: holds in rest position, 29: holds in forward position, 30: holds in backward position, 31: holds in right position, 32: holds in left position, 33: holds in forward diagonally to right position, 34: holds in forward diagonally to left position, 35: holds in backward diagonally to right position, 36: holds in backward diagonally to left position, 37: holds in the raised position, 38: holds in the raised forward position, 39: holds in the raised backward position, 40: holds in the raised right position, 41: holds in the raised left position, 42: holds in the raised forward diagonally to right position, 43: holds in the raised forward diagonally to left position, 44: holds in the raised backward diagonally to right position, 45: holds in the raised backward diagonally to left position, 46: holds in knee-flexed position, 47: holds in knee-flexed forward position, 48: holds in knee-flexed backward position, 49: holds in knee-flexed right position, 50: holds in knee-flexed left position, 51: holds in knee-flexed forward diagonally to right position, 52: holds in knee-flexed forward diagonally to left position, 53: holds in knee-flexed backward diagonally to right position, 54: holds in knee-flexed backward diagonally to left position]. For the hand movements, the details must be selected from these 81 categories: [1: moves close to shoulder, 2: moves forward, 3: moves backward, 4: moves to right, 5: moves to left, 6: moves forward diagonally to right, 7: moves forward diagonally to left, 8: moves backward diagonally to right, 9: moves backward diagonally to left, 10: rises up, 11: rises to up forward, 12: rises to up backward, 13: rises to up right, 14: rises to up left, 15: rises up forward diagonally to right, 16: rises up forward diagonally to left, 17: rises up backward diagonally to right, 18: rises up backward diagonally to left, 19: lowers down, 20: lowers to down forward, 21: lowers to down backward, 22: lowers to down right, 23: lowers to down left, 24: lowers down forward diagonally to right, 25: lowers down forward diagonally to left, 26: lowers down backward diagonally to right, 27: lowers down backward diagonally to left, 28: holds close to shoulder position, 29: holds forward position, 30: holds backward position, 31: holds right position, 32: holds left position, 33: holds forward diagonally to right position, 34: holds forward diagonally to left position, 35: holds backward diagonally to right position, 36: holds backward diagonally to left position, 37: holds up position, 38: holds up forward position, 39: holds up backward position, 40: holds up right position, 41: holds up left position, 42: holds up forward diagonally to right position, 43: holds up forward diagonally to left position, 44: holds up backward diagonally to right position, 45: holds up backward diagonally to left position, 46: holds low position, 47: holds low forward position, 48: holds low backward position, 49: holds low right position, 50: holds low left position, 51: holds low forward diagonally to right position, 52: holds low forward diagonally to left position, 53: holds low backward diagonally to right position, 54: holds low backward diagonally to left position, 55: moves relatively to previous position, 56: moves relatively forward, 57: moves relatively backward, 58: moves to relatively right, 59: moves to relatively left, 60: moves relatively forward diagonally to right, 61: moves relatively forward diagonally to left, 62: moves relatively backward diagonally to right, 63: moves relatively backward diagonally to left, 64: moves relatively up, 65: moves relatively up forward, 66: moves relatively up backward, 67: moves relatively up right, 68: moves relatively up left, 69: moves relatively up forward diagonally to right, 70: moves relatively up forward diagonally to left, 71: moves relatively up backward diagonally to right, 72: moves relatively up backward diagonally to left, 73: moves relatively low, 74: moves relatively low forward, 75: moves relatively low backward, 76: moves relatively low right, 77: moves relatively low left, 78: moves relatively low forward diagonally to right, 79: moves relatively low forward diagonally to left, 80: moves relatively low backward diagonally to right, 81: moves relatively low backward diagonally to left]. For example, for the [Support] line, the triplet list would be like: (left, 1, 0.25), (right, 2, 0.25), (left, 1, 0.25) while (right, 2, 0.25). This means that the first movement is \"left foot steps to rest position in 0.25 seconds\". The second movement is \"right foot steps forward in 0.25 seconds\". The third movement is \"left foot steps to rest position in 0.25 seconds while right foot steps forward in 0.25 seconds. For the [Left hand] line, the tuple list would be like: (1, 0.5), (2, 0.2). This means that the first movement is \"left hand moves close to shoulder in 0.5 seconds\" and the second movement is \"left hand moves forward in 0.2 seconds\". For the [Right hand] line, the structure and definition is similar to [Left hand] lines. Below is the main body of the digit collection describing the movements.\nYou should strictly imitate the following content and create only one digit collection of '{}'. Reply without explaination.\n".format(
                cap)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": dev_prompt
                },
                {
                    "role": "user",
                    "content": f"{lbns}",
                } if ip_only_flg else {},
            ],
            extra_body={"enable_thinking": False},
        )
        response_data = completion.choices[0].message.content
        return response_data

    try:
        res = give_a_post()
        status = True
    except Exception as e:
        print("failed", e)
        status = False
    finally:
        return res, status


def read_data(data_root, lbn_root, model_name):
    with open(lbn_root / 'test_llm_top_5.pkl', 'rb') as f:
        pkl_data = pickle.load(f)
    with open(data_root / 'test.pkl', 'rb') as f:
        ref_data = pickle.load(f)
        # NOTE: reorder is needed
        lens = []
        for k in pkl_data.keys():
            lens.append(len(ref_data['feats'][ref_data['ids'].index(k)]))
    result_dir = lbn_root / 'LLM' / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    total_length = len(pkl_data)
    return result_dir, pkl_data, lens, total_length


def parse_batch_lbn(result_dir, pkl_data, lens, total_length, batch_idx, rep, client=None, model_name=None,
                    ip_only_flg=False):
    result_path = str(result_dir / f'r{rep}_b{batch_idx}.pkl')
    #
    txts_batch_idx = np.array([i for i in range(total_length)])
    bgn_idx = batch_idx * 1000
    end_idx = min((batch_idx + 1) * 1000, total_length)
    txts_batch_idx = txts_batch_idx[bgn_idx:end_idx]
    print(f"from {txts_batch_idx[0]} to {txts_batch_idx[-1]}")
    names = list(pkl_data.keys())

    ##############################
    # Resume
    if Path(result_path).exists():
        with open(Path(result_path), 'rb') as f:
            llm_results = pickle.load(f)
        new_txts_batch_idx = []
        for sIdx in txts_batch_idx:
            cur_name = names[sIdx]
            if cur_name in llm_results:
                continue
            else:
                new_txts_batch_idx.append(sIdx)
        # rewrite to-do idx
        if len(new_txts_batch_idx) != 0:
            txts_batch_idx = np.array(new_txts_batch_idx)
            print(f"\tLoaded existing: {result_path}\n\tJump to {txts_batch_idx[0]}")
        else:
            txts_batch_idx = []
            print(f"\tLoaded existing: {result_path} with completed data\n")
    else:
        llm_results = {}
    ##############################
    for sIdx in tqdm(txts_batch_idx):
        cur_name = names[sIdx]
        cur_lbn_examples = pkl_data[cur_name]['laban']
        cur_caption = pkl_data[cur_name]['caption']
        lbn_example_txt = ''
        for idx, lbn in enumerate(cur_lbn_examples):
            # header
            lbn_example_txt += f"{idx + 1} [Caption] - {lbn['caption']}\n"
            # support
            lbn_example_txt += f"[Support] - {lbn['support']}\n"
            # arm left
            lbn_example_txt += f"[Left hand] - {lbn['arm_left']}\n"
            # arm right
            lbn_example_txt += f"[Right hand] - {lbn['arm_right']}\n"
        out_str = get_llm_output_compose(cap=cur_caption, lbns=lbn_example_txt, seconds=lens[sIdx] / 20., client=client,
                                         model_name=model_name, ip_only_flg=ip_only_flg)
        llm_results[cur_name] = {'laban': out_str, 'caption': cur_caption}
        if sIdx % 10 == 0:
            with open(result_path, 'wb') as handle:
                print(f"temp save in {sIdx}")
                pickle.dump(llm_results, handle, protocol=4)
    with open(result_path, 'wb') as handle:
        pickle.dump(llm_results, handle, protocol=4)
    print(f"[{rep}] saved to {result_path}")


def main():
    parser = argparse.ArgumentParser(description='LLM-based caption generation for motion datasets')
    parser.add_argument('--api_key', type=str, default='xxxxxxxxxxxxxxxx', help='API key for LLM service')
    parser.add_argument('--base_url', type=str, default='https://dashscope.aliyuncs.com/compatible-mode/v1',
                        help='Base URL for LLM API')
    parser.add_argument('--model_name', type=str, default='qwen3-8b', help='Model name to use')
    parser.add_argument('--choice', type=str, default='HML3D', choices=['HML3D', 'KIT', 'LOCO'],
                        help='Dataset choice: HML3D (HumanML3D), KIT (KIT-ML), LOCO (LabanBench)')
    parser.add_argument('--ip_only_flg', action='store_true', default=False, help='Use IP-only flag for LLM requests')
    parser.add_argument('--num_thread', type=int, default=1, help='Number of threads for parallel processing')
    args = parser.parse_args()

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )
    #############################
    data_root = DIR_ROOT / f'../assets/{args.choice}'
    lbn_root = DIR_ROOT / f'../assets/{args.choice}_lbn'
    #############################
    # 1. LLM compose in batch for faster inference
    result_dir, pkl_data, lens, total_length = read_data(data_root, lbn_root, args.model_name)
    args_list = [(i, 0) for i in range(int(np.ceil(total_length / 1000)))]
    with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
        futures = [
            executor.submit(parse_batch_lbn, result_dir, pkl_data, lens, total_length, a, b, client, args.model_name,
                            args.ip_only_flg) for a, b in args_list]
        for future in futures:
            future.result()
    # 2. parse and combine batches into one LabanLite code pkl file
    parse_llm_batch_lbn(lbn_root / 'LLM' / args.model_name, True)
    print("Done!")


if __name__ == '__main__':
    main()
