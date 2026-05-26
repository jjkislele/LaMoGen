import pickle
from tqdm.auto import tqdm

from utils.lbn_utils.mot2lbn import fetch_lbn_and_event_all
from utils.lbn_utils.lbn_codebook import convert_to_lbn_codebook
from utils.skel_painter import draw_seq_3d_in_one_scene
from scripts.common import DIR_ROOT, prepare_gen_data, prepare_pipeline_llm, to_text_cd


def prepare_lbn_codebook(pkl_path, dst_pkl_path):
    """ Read HumanML3D's 3D keypoint sequences, convert to LabanLite code sequences
    """
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    kpt_lst = pkl_data['kpts']
    del pkl_data

    lbns = []
    for kpt in tqdm(kpt_lst):
        ###############################
        # note: be careful of the axis
        # draw_seq_3d_in_one_scene(kpt[::20])
        kpt[:, :, [0, 1, 2]] = kpt[:, :, [0, 2, 1]]
        kpt[:, :, 1] = -kpt[:, :, 1]
        # draw_seq_3d_in_one_scene(kpt[::20])
        ###############################

        lbn_dict = fetch_lbn_and_event_all(kpt, fps=20.)
        lbns.append(convert_to_lbn_codebook(lbn_dict))
    with open(dst_pkl_path, 'wb') as handle:
        pickle.dump(lbns, handle, protocol=4)
    return


def prepare_pipeline(dtype='test', name='HML3D'):
    data_root = DIR_ROOT / f'../assets/{name}'
    lbn_root = DIR_ROOT / f'../assets/{name}_lbn'

    assert (data_root / f'{dtype}.pkl').exists()

    # step1: Prepare LabanLite annotations: convert 3D keypoints to LabanLite codebook indices.
    # Requires assets/{dataset_name}/{train,test,val}.pkl to exist.
    if not (lbn_root / f'{dtype}_lbns_158.pkl').exists():
        prepare_lbn_codebook(data_root / f'{dtype}.pkl', lbn_root / f'{dtype}_lbns_158.pkl')
    else:
        print(f"Step1: resume with {lbn_root / f'{dtype}_lbns_158.pkl'}")

    # step2: Prepare training/testing data for a code generator:
    # input is LabanLite codebook vectors (d=158), plus an extra token for [EOS].
    if not (lbn_root / f'{dtype}_lbns_158_eos.pkl').exists():
        prepare_gen_data(lbn_root / f'{dtype}_lbns_158.pkl', lbn_root / f'{dtype}_lbns_158_eos.pkl')
    else:
        print(f"Step2: resume with {lbn_root / f'{dtype}_lbns_158_eos.pkl'}")

    # step3: Prepare Conceptual Description Database
    if not (lbn_root / f'{dtype}_lbns_cd.pkl').exists():
        to_text_cd(data_root / f'{dtype}.pkl',
                   lbn_root / f'{dtype}_lbns_158.pkl',
                   lbn_root / f'{dtype}_lbns_cd.pkl')
    else:
        print(f"Step3: resume with {lbn_root / f'{dtype}_lbns_cd.pkl'}")


if __name__ == '__main__':
    # steps 1-3
    prepare_pipeline('train', 'LOCO')
    prepare_pipeline('test', 'LOCO')
    # steps 4-6.1
    prepare_pipeline_llm('LOCO')
    print("Done!")
