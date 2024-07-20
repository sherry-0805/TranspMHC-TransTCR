import numpy as np
import pandas as pd
import time
import datetime
from datetime import datetime
import random
import warnings

warnings.filterwarnings("ignore")


from tqdm import tqdm, trange
import sys
import os
import torch
import torch.nn as nn

import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

from TransTCR import TransTCR
import Test_DataProcessing
from Test_DataProcessing import pMHCI_TCR_train_data_with_loader,pMHCI_TCR_test_data_with_loader

from Performance import performances
from sklearn.metrics import precision_recall_curve



vocab = np.load('vocab_dict.npy', allow_pickle=True).item()
vocab_size = len(vocab)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
print(use_cuda)


def parse_args():
    parser = argparse.ArgumentParser(description='The params in training procedure')
    parser.add_argument('-d', '--data', help='The filename of the .csv file contains testing TCR-pMHC-I pairs',
                        dest='data',
                        default='../data/pmhc_tcr_test.csv',
                        type=str
                        )
    parser.add_argument('-m', '--model', help='Trained model',
                        dest='model',
                        default='../model/TransTCR/model_head8_fold0_epoch7.pkl',
                        type=str
                        )
    args = parser.parse_args()
    return args

def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


def eval_step(model, loader, use_cuda=True, name='test'):
    device = torch.device("cuda" if use_cuda else "cpu")

    # torch.manual_seed(19961231)
    # torch.cuda.manual_seed(19961231)
    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list = [], []
        tcr_inputs, pep_inputs, hla_inputs = [], [], []
        for val_tcr_inputs, val_pep_inputs, val_hla_inputs in tqdm(loader):
            val_tcr_inputs, val_pep_inputs, val_hla_inputs = val_tcr_inputs.to(device), val_pep_inputs.to(
                device), val_hla_inputs.to(device)
            val_outputs = model(val_tcr_inputs, val_pep_inputs, val_hla_inputs)
            correct_output = val_outputs[0]
            y_prob_val = nn.Softmax(dim=1)(correct_output)[:, 1].cpu().detach().numpy()
            y_prob_val_list.extend(y_prob_val)

        y_pred_val_list = transfer(y_prob_val_list, 0.5)

        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

    return ys_val, loss_val_list


# Transformer Parameters
d_model = 64  # Embedding Size
d_ff = 512  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 1  # number of Encoder of Decoder Layer
batch_size = 1024
threshold = 0.5
pep_max_len = 20  # peptide; enc_input max sequence length
hla_max_len = 34  # hla; dec_input(=dec_output) max sequence length
tcr_max_len = 27
tgt_len = pep_max_len + hla_max_len + tcr_max_len


args = parse_args()
test_data = pd.read_csv(args.data)
neg_tcr_set = pd.read_csv('../data/pmhc_tcr_train_neg.csv')


model = TransTCR(use_cuda = use_cuda, tgt_len=tgt_len,d_model=d_model)
model_path = args.model  # Update this path
if use_cuda:
    state_dict = torch.load(model_path)
else:
    state_dict = torch.load(model_path, map_location=device)


model.load_state_dict(state_dict, strict=False)
model = model.to(device)

test_tcr_inputs, test_pep_inputs, test_hla_inputs, test_loader = pMHCI_TCR_test_data_with_loader(
    test_data, vocab, batch_size=1024,pep_max_len=20,hla_max_len=34,tcr_max_len=27, shuffle=False)
ys_test_val, loss_test_val_list = eval_step(model,test_loader, use_cuda,'test')

df = {'tcr': test_data['tcr'],
      'peptide': test_data['peptide'],
      'hla': test_data['hla'],
      'HLA_sequence': test_data['HLA_sequence'],
       'prob': ys_test_val[2]}
df = pd.DataFrame(df)
df['percentage'] = 0
df = df.reset_index(drop=True)


for index, (pep, hla, prob) in enumerate(zip(df['peptide'], df['HLA_sequence'], df['prob'])):
    selected_rows = neg_tcr_set.sample(n=1000)
    tcr_list = selected_rows['tcr'].tolist()
    pep_list = [pep] * len(selected_rows)
    hla_list = [hla] * len(selected_rows)
    df_create = pd.DataFrame({'tcr': tcr_list,
                        'peptide': pep_list,
                        'HLA_sequence': hla_list})
    test_tcr_inputs, test_pep_inputs, test_hla_inputs, test_loader = pMHCI_TCR_test_data_with_loader(
        df_create, vocab, batch_size=1024, pep_max_len=20, hla_max_len=34, tcr_max_len=27, shuffle=False)
    ys_test_val, loss_test_val_list = eval_step(model, test_loader, use_cuda, 'test')
    sorted_ys_test_val = sorted(ys_test_val[2], reverse=True)
    for i, v in enumerate(sorted_ys_test_val):
        if prob >= v:
            position = i + 1
            break
        else:
            position = i + 1
    percentage = position/1000
    percentage = percentage*100
    df.at[index, 'percentage'] = percentage



current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

res_df = {'tcr': df['tcr'],
         'peptide': df['peptide'],
         'hla': df['hla'],
         'percentage': df['percentage']}

res_df = pd.DataFrame(res_df)
df_path = os.path.join(os.path.dirname(__file__), '..', 'result','TransTCR', f'{current_time}',
                                       f'tcr_pmhc_result.csv')
os.makedirs(os.path.dirname(df_path), exist_ok=True)
res_df.to_csv(df_path)
print(f'DataFrame saved to {df_path}')





