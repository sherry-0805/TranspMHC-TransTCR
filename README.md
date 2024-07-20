# TranspMHC-TransTCR

This source code is from “Attention-aware differential learning for predicting peptide-MHC class I binding and T cell receptor recognition.” The project offers a comprehensive approach for neoantigen presentation by MHC-I and recognition by TCR. The proposed framework consists of two models: (1) TranspMHC, a transformer-based model for pMHC-I binding prediction, and (2) TransTCR, a transformer-based model using differential learning to predict TCR-pMHC-I recognition.


# Dependencies
python(version=3.7.9); torch(version=2.0.0); numpy(version=1.23.4); pandas(version=1.5.0); scikit-learn(version=1.1.2); tqdm(version=4.66.1)  


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

