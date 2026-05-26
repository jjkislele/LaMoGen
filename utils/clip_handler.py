import clip
import torch


def load_and_freeze_clip(clip_version):
    clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False)
    clip.model.convert_weights(clip_model)

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return clip_model


def emb_text(clip_model, raw_text):
    texts = clip.tokenize(raw_text, truncate=True).to('cuda')
    emb = clip_model.encode_text(texts).float().detach().cpu().numpy()
    return emb


class CLIPWrapper:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = load_and_freeze_clip('ViT-B/32').to(self.device)

    def emb_text_np(self, raw_text):
        """
        :param raw_text: string
        :return: text embedding, np.array [1, 512]
        """
        texts = clip.tokenize(raw_text, truncate=True).to(self.device)
        emb = self.model.encode_text(texts).float().detach().cpu().numpy()
        return emb
