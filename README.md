# TranspMHC-TransTCR

This is the source code of "GLIMS: A two-stage gradual learning method for cancer genes prediction using multi-omics data and co-splicing network". This project presents a comprehensive approach for cancer gene prediction by integrating multiple omics data types and protein-protein interaction (PPI) networks. GLIMS adopts a two-stage progressive learning strategy: (1) a hierarchical graph neural network framework known as HIM-GCN is employed in a semi-supervised manner to predict candidate cancer genes by integrating multi-omics data and PPI networks; (2) the initial predictions are further refined by incorporating co-splicing network information using an unsupervised approach, which serves to prioritize the identified cancer genes.

# Dependencies
python(version=3.7.9) 



# Guided Tutorial

The direct prediction can be performed using our trained TranspMHC and TransTCR models with the scripts ```Test_TranspMHC.py``` and ```Test_TransTCR.py```. 
Here, we illustrate how to use these trained models for pMHC-I and TCR-pMHC-I binding predictions.

```
python ./Test_TranspMHC.py -d ../data/pmhc_test.csv
```
```-d```: default = '../data/pmhc_test.csv', help = 'The filename of the .csv file contains testing pMHC-I pairs'
```
python ./Test_TransTCR.py -d ../data/pmhc_test.csv -d ../data/pmhc_tcr_test.csv
```
```-d```: default = '../data/pmhc_tcr_test.csv',  help = 'The filename of the .csv file contains testing TCR-pMHC-I pairs'

By replacing the training datasets, users can also train TranspMHC and TransTCR on their own data using the scripts ```Train_TranspMHC.py``` and ```Train_TransTCR.py```. 
Here, we illustrate how to use these scripts to train new models for pMHC-I and TCR-pMHC-I binding predictions.

```
python ./Train_TranspMHC.py -train ../data/pmhc_train.csv  -d ../data/pmhc_test.csv
```
```-train```: default = '../data/pmhc_train.csv', help = 'The filename of the .csv file contains training pMHC-I pairs'

```-d```: default = '../data/pmhc_test.csv', help = 'The filename of the .csv file contains testing pMHC-I pairs'
```
python ./Train_TransTCR.py  -pos ../data/pmhc_tcr_train_pos.csv -neg ../data/pmhc_tcr_train_neg.csv -d ../data/pmhc_tcr_test.csv
```
```-pos```: default='../data/pmhc_tcr_train_pos.csv', help = 'The filename of the .csv file contains binding TCR-pMHC-I pairs for training'

```-neg```: default='../data/pmhc_tcr_train_neg.csv', help = 'The filename of the .csv file contains non-binding TCR-pMHC-I pairs for training'

```-d```: default = '../data/pmhc_tcr_test.csv',  help = 'The filename of the .csv file contains testing TCR-pMHC-I pairs'

# Input file example


# Output file example
