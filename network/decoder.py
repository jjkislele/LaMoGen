import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1,T,E)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # X: (B,T,E)
        x = x + self.pe[:, :x.shape[1], :x.shape[2]]
        return self.dropout(x)


class LbnDecoderAttn(nn.Module):
    def __init__(self,
                 nb_joints=22,
                 nb_code=512,
                 pose_dim=263,
                 code_dim=512,
                 **kwargs):
        super().__init__()
        self.code_dim = code_dim  # latent dim
        self.num_code = nb_code
        self.nb_joints = nb_joints
        # decoder
        decoder_layers = nn.TransformerEncoderLayer(d_model=self.code_dim, nhead=8, dim_feedforward=1024,
                                                    dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layers, num_layers=8)
        self.sequence_pos_encoder = PositionalEncoding(self.code_dim, 0.1)
        self.linear_out = nn.Linear(self.code_dim, pose_dim)
        #
        self.codebook = nn.Embedding(self.num_code, self.code_dim)
        # Initialized random/uniformly
        self.codebook.weight.data.uniform_(-1.0 / self.num_code, 1.0 / self.code_dim)

    def forward(self, code_indices, masks, **kwargs):
        B, T, _ = code_indices.shape
        device = self.codebook.weight.device
        codes_flattened = code_indices.contiguous().view(-1, self.num_code).to(device)
        masks = masks.to(device)

        # Retrieve latent representation as linear combination of codes
        z = torch.matmul(codes_flattened, self.codebook.weight).view((-1, self.code_dim))
        z = z.view(B, T, -1).permute(0, 1, 2).contiguous()

        # decoder
        z = self.sequence_pos_encoder(z)
        x_decoder = self.decoder(src=z, src_key_padding_mask=~masks)
        x_out = self.linear_out(x_decoder)

        return x_out
