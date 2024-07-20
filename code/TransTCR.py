import random

random.seed(1234)


import os
import torch
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import Trans_Module
from Trans_Module import Encoder, Decoder


class TransTCR(nn.Module):
    def __init__(self, use_cuda = True, tgt_len=81,d_model=64):
        super(TransTCR, self).__init__()
        self.use_cuda = use_cuda
        device = torch.device("cuda" if use_cuda else "cpu")
        self.pep_encoder = Encoder().to(device)
        self.hla_encoder = Encoder().to(device)
        self.tcr_encoder = Encoder().to(device)
        # self.hlapep_encoder= Decoder2().to(device)

        self.decoder = Decoder().to(device)
        self.tgt_len = tgt_len
        self.projection = nn.Sequential(
            nn.Linear(tgt_len * d_model, 256),
            nn.ReLU(True),

            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),

            # output layer
            nn.Linear(64, 2)
        ).to(device)

    def forward(self, tcr_inputs, pep_inputs, hla_inputs):
        '''
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        '''
        tcr_enc_outputs, tcr_enc_self_attns = self.tcr_encoder(tcr_inputs)
        pep_enc_outputs, pep_enc_self_attns = self.pep_encoder(pep_inputs)
        hla_enc_outputs, hla_enc_self_attns = self.hla_encoder(hla_inputs)

        enc1_outputs = torch.cat((pep_enc_outputs, hla_enc_outputs), 1)  # concat pep & hla embedding

        enc_outputs = torch.cat((enc1_outputs, tcr_enc_outputs), 1)
        dec_outputs, dec_self_attns = self.decoder(enc_outputs)
        dec_outputs = dec_outputs.view(dec_outputs.shape[0], -1)  # Flatten [batch_size, tgt_len * d_model]
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]

        return dec_logits.view(-1, dec_logits.size(-1)), \
               tcr_enc_self_attns, pep_enc_self_attns, hla_enc_self_attns, dec_self_attns



