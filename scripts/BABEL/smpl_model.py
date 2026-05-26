import numpy as np
import torch

from scripts.HumanML3D.human_body_prior.body_model.body_model import BodyModel


def c2c(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()


class SMPLHandler:
    def __init__(self, smpl_root):
        #######################################
        # config
        comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        male_bm_path = f'{smpl_root}/smplh/male/model.npz'
        male_dmpl_path = f'{smpl_root}/dmpls/male/model.npz'
        num_betas = 10  # number of body parameters
        num_dmpls = 8  # number of DMPL parameters
        self.bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls,
                            dmpl_fname=male_dmpl_path).to(comp_device)
        self.trans_matrix = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0],
                                      [0.0, 1.0, 0.0]])
        self.ex_fps = 20
        self.comp_device = comp_device

    def amass_to_pose(self, poses, trans):
        """ gender: male, src fps: 30
        @param poses: T, 66
        @param trans: T, 3
        """
        T = len(poses)
        fps = 30
        down_sample = int(fps / self.ex_fps)
        bdata_poses = np.concatenate((poses[::down_sample, ...], np.zeros((T, 156 - 66))), axis=-1)  # T, 156
        bdata_trans = trans[::down_sample, ...]  # T, 3

        body_parms = {
            'root_orient': torch.Tensor(bdata_poses[:, :3]).to(self.comp_device),
            'pose_body': torch.Tensor(bdata_poses[:, 3:66]).to(self.comp_device),
            'pose_hand': torch.Tensor(bdata_poses[:, 66:]).to(self.comp_device),
            'trans': torch.Tensor(bdata_trans).to(self.comp_device),
            'betas': torch.Tensor(np.repeat(np.zeros((1, 10)), repeats=T, axis=0)).to(self.comp_device),
        }
        with torch.no_grad():
            body = self.bm(**body_parms)
        pose_seq_np = body.Jtr.detach().cpu().numpy()
        pose_seq_np_n = np.dot(pose_seq_np, self.trans_matrix)
        return pose_seq_np_n
