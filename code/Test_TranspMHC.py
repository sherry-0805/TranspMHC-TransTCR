import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import os
from TranspMHC import TranspMHC  # Assuming this is the model class
from Test_DataProcessing import pMHCI_data_with_loader
import argparse
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
from Performance import performances
from tqdm import tqdm
from datetime import datetime

vocab = np.load('vocab_dict.npy', allow_pickle=True).item()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
print(use_cuda)

def parse_args():
    parser = argparse.ArgumentParser(description='The params in training procedure')
    parser.add_argument('-d', '--data', help='The filename of the .csv file contains training pMHC-I pairs',
                        dest='data',
                        default='../data/pmhc_test.csv',
                        type=str
                        )
    parser.add_argument('-m', '--model', help='Trained model',
                        dest='model',
                        default='../model/TranspMHC/model_head5_fold4_epoch8.pkl',
                        type=str
                        )
    args = parser.parse_args()
    return args


def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


def eval_step(model, loader, use_cuda=True,  name='test'):
    device = torch.device("cuda" if use_cuda else "cpu")

    # torch.manual_seed(19961231)
    # torch.cuda.manual_seed(19961231)
    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list = [], []
        pep_inputs, hla_inputs = [], []
        for val_pep_inputs, val_hla_inputs in tqdm(loader):
            val_pep_inputs, val_hla_inputs = val_pep_inputs.to(device), val_hla_inputs.to(device)
            val_outputs = model(val_pep_inputs, val_hla_inputs)
            correct_output = val_outputs[0]
            y_prob_val = nn.Softmax(dim=1)(correct_output)[:, 1].cpu().detach().numpy()
            y_prob_val_list.extend(y_prob_val)

        y_pred_val_list = transfer(y_prob_val_list, 0.5)
        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

    return ys_val, loss_val_list


args = parse_args()

test_data = pd.read_csv(args.data)
test_pep_inputs, test_hla_inputs, test_loader = pMHCI_data_with_loader(
    test_data, vocab, batch_size=1024, pep_max_len=25, hla_max_len=34, shuffle=False)

pep_max_len = 25  # peptide; enc_input max sequence length
hla_max_len = 34  # hla; dec_input(=dec_output) max sequence length
tgt_len = pep_max_len + hla_max_len
d_model = 64
d_ff = 512
d_k = d_v = 64
n_layers = 1
# Load the pretrained model
model = TranspMHC(use_cuda=use_cuda, tgt_len=tgt_len, d_model=d_model)
model_path = args.model  # Update this path
if use_cuda:
    state_dict = torch.load(model_path)
else:
    state_dict = torch.load(model_path, map_location=device)

model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()  # Set the model to evaluation mode

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
ys_test_val, loss_test_val_list = eval_step(model, test_loader,use_cuda=use_cuda,name='test')
df = {'peptide': test_data['peptide'],
      'hla': test_data['hla'],
      'prob': ys_test_val[2]}  # Update this line according to the actual data you have
df = pd.DataFrame(df)
df_path = os.path.join(os.path.dirname(__file__), '..', 'result','TranspMHC', f'{current_time}',
                                       f'pmhc_result.csv')
os.makedirs(os.path.dirname(df_path), exist_ok=True)
df.to_csv(df_path)
print(f'DataFrame saved to {df_path}')


