import numpy as np
import pandas as pd
import time
import datetime
from datetime import datetime
import random

random.seed(1234)

import warnings

warnings.filterwarnings("ignore")


from tqdm import tqdm, trange
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import StratifiedKFold
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from torch.utils.data import DataLoader, TensorDataset
import TranspMHC
from TranspMHC import TranspMHC
import DataProcessing
from DataProcessing import pMHCI_dataset
from DataProcessing import pMHCI_data_with_loader
import Performance
from Performance import performances
from sklearn.metrics import precision_recall_curve



vocab = np.load('vocab_dict.npy', allow_pickle=True).item()
vocab_size = len(vocab)
# Transformer Parameters
d_model = 64  # Embedding Size
d_ff = 512  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 1  # number of Encoder of Decoder Layer

batch_size = 1024
epochs = 18

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

threshold = 0.5
pep_max_len = 25  # peptide; enc_input max sequence length
hla_max_len = 34  # hla; dec_input(=dec_output) max sequence length
tgt_len = pep_max_len + hla_max_len

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
result_path = os.path.join(os.path.dirname(__file__), '..', 'result','TranspMHC', f'{current_time}.txt')
os.makedirs(os.path.dirname(result_path), exist_ok=True)
sys.stdout = open(result_path, "w")


def parse_args():
    parser = argparse.ArgumentParser(description='The params in training procedure')
    parser.add_argument('-d', '--data', help='The filename of the .csv file contains test pMHC-I pairs',
                        dest='data',
                        default='../data/pmhc_test.csv',
                        type=str
                        )
    parser.add_argument('-train', '--pmhc_train', help='The filename of the .csv file contains training pMHC-I pairs',
                        dest='pmhc_train',
                        default='../data/pmhc_train.csv',
                        type=str
                        )
    args = parser.parse_args()
    return args


def f_mean(l):
    if len(l) == 0:
        return 0
    return sum(l) / len(l)


def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


def train_step(model, optimizer, train_loader, fold, epoch, epochs, use_cuda=True):
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    time_train_ep = 0
    model.train()
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list, dec_attns_train_list = [], []
    for train_pep_inputs, train_hla_inputs, train_labels in tqdm(train_loader):
        train_pep_inputs, train_hla_inputs, train_labels = train_pep_inputs.to(device), train_hla_inputs.to(device), train_labels.to(device)

        t1 = time.time()

        optimizer.zero_grad()
        train_outputs, _, _, train_dec_self_attns = model(train_pep_inputs,train_hla_inputs)

        criterion = nn.CrossEntropyLoss()
        train_loss = criterion(train_outputs, train_labels)
        time_train_ep += time.time() - t1

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        y_true_train = train_labels.cpu().numpy()
        y_prob_train = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()

        y_true_train_list.extend(y_true_train)
        y_prob_train_list.extend(y_prob_train)
        loss_train_list.append(train_loss)

    y_pred_train_list = transfer(y_prob_train_list, threshold)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)

    print('Fold-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(fold, epoch, epochs,
                                                                                              f_mean(loss_train_list),
                                                                                              time_train_ep))
    return loss_train_list


def eval_step(model, loader, use_cuda=True,  name='test'):
    model.eval()
    device = torch.device("cuda" if use_cuda else "cpu")

    # torch.manual_seed(19961231)
    # torch.cuda.manual_seed(19961231)
    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list = [], []
        pep_inputs, hla_inputs = [], []
        for val_pep_inputs, val_hla_inputs, val_labels in tqdm(loader):
            val_pep_inputs, val_hla_inputs, val_labels = val_pep_inputs.to(
                device), val_hla_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_pep_inputs, val_hla_inputs)
            correct_output = val_outputs[0]
            y_prob_val = nn.Softmax(dim=1)(correct_output)[:, 1].cpu().detach().numpy()
            y_true_val = val_labels.cpu().numpy()
            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)

        if name=='test':
            precision, recall, thresholds = precision_recall_curve(y_true_val_list, y_prob_val_list)
            best_index = np.argmax(precision * recall)
            best_threshold = thresholds[best_index]
            print('best_threshold = ', best_threshold)
            y_pred_val_list = transfer(y_prob_val_list, best_threshold)

        else:
            y_pred_val_list = transfer(y_prob_val_list, 0.5)

        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

        metrics_val = performances(y_true_val_list, y_pred_val_list,y_prob_val_list, print_=True)
    return ys_val, loss_val_list, metrics_val


args = parse_args()
vocab = np.load('vocab_dict.npy', allow_pickle = True).item()
train_data = pd.read_csv(args.pmhc_train)
test_data = pd.read_csv(args.data)

train_pep_inputs, train_hla_inputs, train_labels, train_loader = pMHCI_data_with_loader(
    train_data, vocab, batch_size=1024,pep_max_len=25,hla_max_len=34,shuffle=False)

test_pep_inputs, test_hla_inputs, test_labels, test_loader = pMHCI_data_with_loader(
    test_data, vocab, batch_size=1024,pep_max_len=25,hla_max_len=34,shuffle=False)

k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True)

for n_heads in range(1,9):
    train_fold_metrics_list, val_fold_metrics_list = [], []
    ys_train_fold_dict, ys_val_fold_dict = {}, {}
    loss_test_list=[]
    for fold, (train_index, val_index) in enumerate(skf.split(train_pep_inputs,train_labels)):
        train_pep, train_hla= train_pep_inputs[train_index], train_hla_inputs[train_index]
        val_pep, val_hla=train_pep_inputs[val_index], train_hla_inputs[val_index]
        train_label=train_labels[train_index]
        val_label=train_labels[val_index]
        train_loader = DataLoader(pMHCI_dataset(train_pep, train_hla, train_label), batch_size=1024, shuffle=True)
        val_loader = DataLoader(pMHCI_dataset(val_pep, val_hla,val_label), batch_size=1024, shuffle=True)

        """ Load model """
        model = TranspMHC(use_cuda=use_cuda, tgt_len=tgt_len, d_model=d_model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        metric_best, ep_best = 0, 0
        time_train = 0
        print('-----Train-----')

        for epoch in range(1,epochs + 1):
            loss_train_list = train_step(model, optimizer, train_loader, fold, epoch, epochs, use_cuda=use_cuda)
            print('-----Validation results---------')
            ys_train_val, loss_train_val_list, metrics_train_val = eval_step(model, val_loader,use_cuda=use_cuda, name='val')
            metrics_train_ep_avg = sum(metrics_train_val[:2]) / 2
            print(metrics_train_ep_avg)
            print('-----Validation completed---------')

            print('------Test results-----')
            ys_test_val, loss_test_val_list, metrics_test_val = eval_step(model, test_loader,use_cuda=use_cuda,name='test')
            metrics_test_ep_avg = sum(metrics_test_val[:2]) / 2
            print(metrics_test_ep_avg)
            print('------Test completed-----')

            metrics_ep_avg = metrics_test_ep_avg
            if metrics_ep_avg > metric_best:
                metric_best, ep_best = metrics_ep_avg, epoch
                dir_saver = os.path.join(os.path.dirname(__file__), '..', 'model','TranspMHC', f'{current_time}')
                path_saver = os.path.join(dir_saver, 'model_head{}_fold{}_epoch{}.pkl'.format(n_heads, fold, ep_best))
                print('dir_saver: ', dir_saver)
                print('path_saver: ', path_saver)
                if not os.path.exists(dir_saver):
                    os.makedirs(dir_saver)
                print('****Saving model:Fold{} Best epoch = {} | 5metrics_Best_avg = {:.4f}'.format(fold,ep_best, metric_best))
                print('*****Path saver: ', path_saver)
                torch.save(model.eval().state_dict(), path_saver)

                # save the prediction results
                df = {'peptide': test_data['peptide'],
                      'hla': test_data['hla'],
                      'label': ys_test_val[0],
                      'prob': ys_test_val[2]}  # Update this line according to the actual data you have
                df = pd.DataFrame(df)
                df_path = os.path.join(os.path.dirname(__file__), '..', 'result','TranspMHC', f'{current_time}',
                                       f'result_head{n_heads}_fold{fold}_epoch{epoch}.csv')
                os.makedirs(os.path.dirname(df_path), exist_ok=True)
                df.to_csv(df_path)
                print(f'DataFrame saved to {df_path}')



sys.stdout.close()
















