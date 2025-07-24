import numpy as np; import pandas as pd; 
from sklearn.preprocessing import StandardScaler; from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold; from scipy.stats import pearsonr; 
from bayes_opt import BayesianOptimization, UtilityFunction
import pyreadr, sys, datetime, time, os, glob, random; from patsy import dmatrix
import matplotlib.pyplot as plt; from statsmodels.graphics.api import abline_plot
import torch as tch

print("Mensaje", flush=True)
GPU = True if tch.cuda.is_available() else False
Num_Threads = 1
os.environ["OMP_NUM_THREADS"] = str(Num_Threads)
os.environ["MKL_NUM_THREADS"] = str(Num_Threads)
tch.set_num_threads(Num_Threads)
tch.set_num_interop_threads(Num_Threads)

Dir = os.getcwd()
Dir_Progs = os.path.normpath(Dir)
sys.path.append(Dir_Progs)

from Funs_Auxs import *
Match_f
from MLP_v8_0 import *

Predictor = 'GID'
Dir_DataSet = os.path.normpath(os.path.join(Dir, 'dat-PP'))
Folds_df = pd.read_csv(os.path.join(Dir_DataSet, '10Folds_df.csv'))

DataSets = ['Wheat_' + str(j + 1) + '_AL' for j in range(6)]

for dn in range(1, 6 + 1):
    DataNumber = dn - 1
    Dir_dat = os.path.join(Dir_DataSet, DataSets[DataNumber] + '.RData')
    OD = pyreadr.read_r(Dir_dat)
    print(OD.keys())

    Pheno = OD['Pheno'].rename(columns={'Line': 'GID'})
    Traits = ["GY"]

    for t in range(1, 2):
        print('DataSet', DataSets[DataNumber])
        y = Pheno[Traits[t - 1]]
        y = y.to_numpy(copy=True).astype(float).reshape(-1, 1)

        XM = OD['Markers']
        XM = XM.apply(np.round)
        Scaler_Markers = StandardScaler()
        XM = Scaler_Markers.fit_transform(XM)
        G = np.matmul(XM, XM.T) / XM.shape[1]
        G = pd.DataFrame(G)
        G5 = XR2_f(G)
        Order = OD['Markers'].index
        Pheno['GID'] = pd.Categorical(Pheno['GID'], categories=Order, ordered=True)
        ZL = pd.get_dummies(Pheno['GID'], columns=['GID'])
        XL = np.matmul(ZL, G5[Match_f(ZL.columns, OD['Markers'].index), :])
        XL = XL.to_numpy()
        ZL = None
        GIDs = Pheno['GID'].unique()
        Fold_df = Folds_df[Folds_df['DataSet'] == DataSets[DataNumber]]
        K = len(Fold_df['Fold'].unique())
        x_ls_all = [XL]
        Params = {}
        Hypers_Fixed = {'Epochs': 128, 'Batch_size': len(y), 'Seed': 42, 'Iters': 250,
                        'IK': 10, 'Use': 10}
        Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Hypers_Fixed['Device'] = Device
        Dir_Outs0 = os.path.join(Dir, 'Outs-Mod-v8_0-Shared-Iters-' + str(Hypers_Fixed['Iters']), DataSets[dn - 1])
        os.makedirs(Dir_Outs0, exist_ok=True)

        for k in range(1, K + 1):
            Fold = np.repeat(100, len(y))
            Fold[Fold_df['Pos_tst'][Fold_df['Fold'] == k].to_numpy() - 1] = k
            x_ls_tr, y_tr = [x_ls_all[m][Fold != k, :] for m in range(len(x_ls_all))], y[Fold != k]
            Scaler_y = StandardScaler()
            y_tr = Scaler_y.fit_transform(y_tr)
            dat_tr = {'x_ls_tr': x_ls_tr, 'y_tr': y_tr}
            Hypers_Fixed['IFold'] = SIFold(dat_tr['y_tr'], K=Hypers_Fixed['IK'], nq=10, random_state=Hypers_Fixed['Seed'])
            Time = time.time()

            def f_O(**kwargs):
                nHL1, Units1 = kwargs['nHL1'], kwargs['Units1']
                ll, llr, lwd = kwargs['ll'], kwargs['llr'], kwargs['lwd']
                Ind = np.all(2 ** nHL1 <= Units1)
                if Ind:
                    nHLs, Units = [nHL1], [Units1]
                    nHL_ls, Units = [int(nHLs[j]) for j in range(len(nHLs))], [int(Units[j]) for j in range(len(nHLs))]
                    Units_ls = [[int(Units[m] / (2 ** j)) for j in range(nHL_ls[m])] for m in range(len(x_ls_all))]
                    Hypers_R = {'UnitsB1_lsls': Units_ls,
                                'l': np.exp(ll), 'lr': np.exp(llr), 'wd': np.exp(lwd)}
                    Hypers_p = {**Hypers_Fixed, **Hypers_R}
                    Val_metric = IKFCV_f(dat_tr, Hypers_p, IK=Hypers_Fixed['IK'],
                                         Use=Hypers_Fixed['Use'], PlotTraining=False)
                    MSEP_Val = np.mean(Val_metric['Val_metric'])
                    if np.isnan(MSEP_Val) or np.isinf(MSEP_Val) or MSEP_Val > np.var(y_tr) * 1000:
                        MSEP_Val = np.var(y_tr) * 1000
                    return -MSEP_Val
                else:
                    return -np.var(y_tr) * 1000

            Bounds = {
                'nHL1': [1, 4],
                'Units1': [64, 512],
                'll': [np.log(np.e ** -10), np.log(np.e ** -1)],
                'llr': [np.log(np.e ** -10), np.log(np.e ** 0)],
                'lwd': [np.log(1 - 0.95), np.log(1 - 0.05)]
            }

            BO = BayesianOptimization(
                f=f_O,
                pbounds=Bounds,
                random_state=50,
                verbose=1
            )

            utility = UtilityFunction(kind="ei", xi=0.1, kappa=2.576)

            for _ in range(5):
                next_point = BO.suggest(utility)
                target = f_O(**next_point)
                BO.register(params=next_point, target=target)

            for _ in range(Hypers_Fixed['Iters']):
                next_point = BO.suggest(utility)
                target = f_O(**next_point)
                BO.register(params=next_point, target=target)

            print(BO.max)
