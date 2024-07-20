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
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import KFold
import argparse

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1234)

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from torch.utils.data import DataLoader, TensorDataset
import TransTCR
from TransTCR import TransTCR
import DataProcessing
from DataProcessing import pMHCI_TCR_train_data_with_loader,pMHCI_TCR_test_data_with_loader
from DataProcessing import pMHCI_TCR_train_dataset, pMHCI_TCR_test_dataset
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
epochs = 15

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

threshold = 0.5
pep_max_len = 20  # peptide; enc_input max sequence length
hla_max_len = 34  # hla; dec_input(=dec_output) max sequence length
tcr_max_len = 27
tgt_len = pep_max_len + hla_max_len + tcr_max_len

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
result_path = os.path.join(os.path.dirname(__file__), '..', 'result','TransTCR', f'{current_time}.txt')
os.makedirs(os.path.dirname(result_path), exist_ok=True)
sys.stdout = open(result_path, "w")

def parse_args():
    parser = argparse.ArgumentParser(description='The params in training procedure')
    parser.add_argument('-d', '--data', help='The filename of the .csv file contains testing TCR-pMHC-I pairs',
                        dest='data',
                        default='../data/pmhc_tcr_test.csv',
                        type=str
                        )
    parser.add_argument('-pos', '--pos_tcr_pmhc', help='The filename of the .csv file contains binding TCR-pMHC-I pairs for training',
                        dest='pos_tcr_pmhc',
                        default='../data/pmhc_tcr_train_pos.csv',
                        type=str
                        )
    parser.add_argument('-neg', '--neg_tcr_pmhc', help='The filename of the .csv file contains non-binding TCR-pMHC-I pairs for training',
                        dest='neg_tcr_pmhc',
                        default='../data/pmhc_tcr_train_neg.csv',
                        type=str
                        )
    args = parser.parse_args()
    return args


def differential_loss(y_pos, y_neg,margin=0.8):
    relu = nn.ReLU()
    diff = relu(margin + y_neg - y_pos).mean()
    regularization_term = 0.03 * (y_neg ** 2 + y_pos ** 2).mean()
    diff_loss = diff + regularization_term
    return diff_loss


def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])



def f_mean(l):
    if len(l) == 0:
        return 0
    return sum(l) / len(l)

def train_step(model, optimizer, train_loader, fold, epoch, epochs, use_cuda=True):
    device = torch.device("cuda" if use_cuda else "cpu")

    time_train_ep = 0
    model.train()
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list, dec_attns_train_list = [], []
    for train_postcr_inputs, train_pep_inputs, train_hla_inputs, train_negtcr_inputs, train_labels in tqdm(
            train_loader):
        train_postcr_inputs, train_pep_inputs, train_hla_inputs, train_negtcr_inputs, train_labels = train_postcr_inputs.to(
            device), train_pep_inputs.to(device), train_hla_inputs.to(device), train_negtcr_inputs.to(
            device), train_labels.to(device)

        t1 = time.time()

        optimizer.zero_grad()
        train_pos_outputs, _, _, _, train_pos_dec_self_attns = model(train_postcr_inputs, train_pep_inputs,
                                                                 train_hla_inputs)
        train_neg_outputs, _, _, _, train_neg_dec_self_attns = model(train_negtcr_inputs, train_pep_inputs,
                                                                 train_hla_inputs)
        y_pos_tcr = train_pos_outputs[:, 1]
        y_neg_tcr = train_neg_outputs[:, 1]
        train_loss = differential_loss( y_pos_tcr, y_neg_tcr)

        time_train_ep += time.time() - t1

        train_loss.backward()
        optimizer.step()

        y_true_train = train_labels.cpu().numpy()
        y_prob_train = nn.Softmax(dim=1)(train_pos_outputs)[:, 1].cpu().detach().numpy()

        y_true_train_list.extend(y_true_train)
        y_prob_train_list.extend(y_prob_train)
        loss_train_list.append(train_loss)

    y_pred_train_list = transfer(y_prob_train_list, threshold)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)

    print('Fold-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(fold, epoch, epochs,
                                                                                              f_mean(loss_train_list),
                                                                                              time_train_ep))
    return loss_train_list



def eval_step(model, n_heads,fold,loader, epoch, use_cuda=True, output=False, name='test'):
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)

    # torch.manual_seed(19961231)
    # torch.cuda.manual_seed(19961231)
    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list = [], []
        tcr_inputs, pep_inputs, hla_inputs = [], [], []
        for val_tcr_inputs, val_pep_inputs, val_hla_inputs, val_labels in tqdm(loader):
            val_tcr_inputs, val_pep_inputs, val_hla_inputs, val_labels = val_tcr_inputs.to(device), val_pep_inputs.to(
                device), val_hla_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_tcr_inputs, val_pep_inputs, val_hla_inputs)
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
train_posdata = pd.read_csv(args.pos_tcr_pmhc)
train_negdata = pd.read_csv(args.neg_tcr_pmhc)
test_data = pd.read_csv(args.data)

train_postcr_inputs, train_pep_inputs, train_hla_inputs, train_negtcr_inputs, train_labels, train_loader = pMHCI_TCR_train_data_with_loader(
    train_posdata, train_negdata, vocab, batch_size=1024, pep_max_len=20, hla_max_len=34,tcr_max_len=27,shuffle=True)
test_tcr_inputs, test_pep_inputs, test_hla_inputs, test_labels, test_loader = pMHCI_TCR_test_data_with_loader(
    test_data, vocab, batch_size=1024,pep_max_len=20,hla_max_len=34,tcr_max_len=27, shuffle=False)


k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True)

for n_heads in range(1,9):
    print('Heads',n_heads)
    train_fold_metrics_list, val_fold_metrics_list = [], []
    ys_train_fold_dict, ys_val_fold_dict = {}, {}
    loss_test_list=[]
    for fold, (train_index, val_index) in enumerate(kf.split(train_postcr_inputs)):
        """ 5 CV process, generating the train and val data loader  """
        train_postcr, train_pep, train_hla, train_negtcr, train_label = (
            train_postcr_inputs[train_index],
            train_pep_inputs[train_index],
            train_hla_inputs[train_index],
            train_negtcr_inputs[train_index],
            train_labels[train_index]
        )
        train_loader = Data.DataLoader(pMHCI_TCR_train_dataset(train_postcr, train_pep, train_hla, train_negtcr, train_label), batch_size=1024, shuffle=True, num_workers=0)

        val_postcr, val_pep, val_hla, val_negtcr, val_label = (
            train_postcr_inputs[val_index],
            train_pep_inputs[val_index],
            train_hla_inputs[val_index],
            train_negtcr_inputs[val_index],
            train_labels[val_index]
        )
        concatenated = torch.cat((val_postcr, val_pep, val_hla), dim=1)
        unique_rows = torch.unique(concatenated, dim=0)
        val_postcr_unique, val_pep_unique, val_hla_unique = torch.split(
            unique_rows,
            [val_postcr.size(1), val_pep.size(1), val_hla.size(1)],
            dim=1
        )

        val_tcr_inputs = torch.cat((val_postcr_unique, val_negtcr), dim=0)
        val_pep_inputs = torch.cat((val_pep_unique, val_pep), dim=0)
        val_hla_inputs = torch.cat((val_hla_unique, val_hla), dim=0)
        # Create labels for the validation set
        pos_label = torch.LongTensor([1] * val_postcr_unique.size(0))
        neg_label = torch.LongTensor([0] * val_negtcr.size(0))
        val_label_inputs = torch.cat((pos_label, neg_label), dim=0)
        val_loader  = Data.DataLoader(pMHCI_TCR_test_dataset(val_tcr_inputs, val_pep_inputs, val_hla_inputs, val_label_inputs), batch_size, shuffle=True,num_workers=0)

        """ Load model """
        model = TransTCR(use_cuda = use_cuda, tgt_len=tgt_len,d_model=d_model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        metric_best, ep_best = 0, 0
        time_train = 0
        print('-----Train-----')

        for epoch in range(1,epochs + 1):
            loss_train_list = train_step(model, optimizer, train_loader, fold, epoch, epochs, use_cuda)
            print('-----Validation results---------')
            ys_train_val, loss_train_val_list, metrics_train_val = eval_step(model, n_heads, fold, val_loader, epoch,
                                                                             use_cuda, False, 'val')
            metrics_train_ep_avg = sum(metrics_train_val[:2]) / 2
            print(metrics_train_ep_avg)
            print('-----Validation completed---------')

            print('------Test results-----')
            ys_test_val, loss_test_val_list, metrics_test_val = eval_step(model, n_heads, fold,test_loader, epoch, use_cuda,
                                                                          True, 'test')
            metrics_test_ep_avg = sum(metrics_test_val[:2]) / 2
            print(metrics_test_ep_avg)
            print('------Test completed-----')

            metrics_ep_avg = metrics_test_ep_avg
            if metrics_ep_avg > metric_best:
                metric_best, ep_best = metrics_ep_avg, epoch
                dir_saver = os.path.join(os.path.dirname(__file__), '..', 'model','TransTCR', f'{current_time}')
                path_saver = os.path.join(dir_saver, 'model_head{}_fold{}_epoch{}.pkl'.format(n_heads, fold, ep_best))
                print('dir_saver: ', dir_saver)
                print('path_saver: ', path_saver)
                if not os.path.exists(dir_saver):
                    os.makedirs(dir_saver)
                print('****Saving model:Fold{} Best epoch = {} | 5metrics_Best_avg = {:.4f}'.format(fold,ep_best, metric_best))
                print('*****Path saver: ', path_saver)
                """
                torch.save(model.eval().state_dict(), path_saver)
                """
                model.eval()
                with open(path_saver, 'wb') as f:
                    torch.save(model.state_dict(), f)
                state_dict = torch.load(path_saver)
                model.load_state_dict(state_dict)
                ys_test_val, loss_test_val_list, metrics_test_val = eval_step(model, n_heads,fold,test_loader, epoch, use_cuda,
                                                                          True, 'test')

                # save the prediction results
                df = {'tcr': test_data['tcr'],
                      'peptide': test_data['peptide'],
                      'hla': test_data['hla'],
                      'label': ys_test_val[0],
                      'prob': ys_test_val[2]}  # Update this line according to the actual data you have
                df = pd.DataFrame(df)
                df_path = os.path.join(os.path.dirname(__file__), '..', 'result','TransTCR', f'{current_time}',
                                       f'result_head{n_heads}_fold{fold}_epoch{epoch}.csv')
                os.makedirs(os.path.dirname(df_path), exist_ok=True)
                df.to_csv(df_path)
                print(f'DataFrame saved to {df_path}')



sys.stdout.close()
