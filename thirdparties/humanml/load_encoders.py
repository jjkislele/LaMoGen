from .modules import MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo


#################################################
def get_movement_enc(ckpt, tgt_dim=135):
    movement_enc = MovementConvEncoder(tgt_dim, 512, 512)
    movement_enc.load_state_dict(ckpt)
    movement_enc = movement_enc.eval()
    return movement_enc


def get_text_enc(ckpt):
    text_enc = TextEncoderBiGRUCo(word_size=300, pos_size=15, hidden_size=512, output_size=512, device='cuda:0')
    text_enc.load_state_dict(ckpt)
    text_enc = text_enc.eval()
    return text_enc


def get_motion_enc(ckpt):
    motion_enc = MotionEncoderBiGRUCo(input_size=512, hidden_size=1024, output_size=512, device='cuda:0')
    motion_enc.load_state_dict(ckpt)
    motion_enc = motion_enc.eval()
    return motion_enc


def motion_wo_foot_contact(motion, foot_contact_entries):
    if foot_contact_entries == 0:
        return motion
    else:
        return motion[..., :-foot_contact_entries]
