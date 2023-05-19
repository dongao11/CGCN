import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import time
import utils2
import models
import spectral
# np.random.seed(1337)



parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 160)
parser.add_argument("-c","--src_input_dim",type = int, default = 128)
parser.add_argument("-d","--tar_input_dim",type = int, default = 200) # PaviaU=103；salinas=204
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 16)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 20000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-m","--test_class_num",type=int, default=16)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')

args = parser.parse_args(args=[])


FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit


TEST_CLASS_NUM = args.test_class_num
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class

utils2.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()

pwd = os.getcwd()
f_pwd = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")

test_data = f_pwd+'/few-shot-gnn-master/datasets/IP/indian_pines_corrected.mat'
test_label = f_pwd+'/few-shot-gnn-master/datasets/IP/indian_pines_gt.mat'

Data_Band_Scaler, GroundTruth = utils2.load_data(test_data, test_label)


def get_train_test_loader(Data_Band_Scaler, GroundTruth,partition):
    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils2.flip(Data_Band_Scaler)
    groundtruth = utils2.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]


    [Row, Column] = np.nonzero(G)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    val_train = {}
    da_train={}
    m = int(np.max(G))



    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]#将多数组维度拉成一维数组
        classlen=len(indices)
        np.random.shuffle(indices)
        nb_val = (classlen//10)*8
        nb_val2=(classlen//10)*9
        da_train[i] = []
        train[i]=[]
        test[i]=[]
        val_train[i]=[]
        train[i] = indices[:nb_val]
        test[i] = indices[nb_val:nb_val2]
        val_train[i] = indices[nb_val2:]
        if i==0 or i==6 or i==8 or i==12 or i==14 or i==15:
            for j in range(50):
                train[i] += indices[:nb_val]
                test[i] += indices[nb_val:nb_val2]
                val_train[i] += indices[nb_val2:]


    train_indices = []
    test_indices = []
    val_train_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        val_train_indices += val_train[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))
    print('the number of test_indices:', len(test_indices))
    print('the number of val_indices:', len(val_train_indices))
    print('the number of train_indices after data argumentation:', len(da_train_indices))


    nTrain = len(train_indices)
    nTest = len(test_indices)
    val_nTrain = len(val_train_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = utils2.radiation_noise(data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,#[Row, Column] = np.nonzero(G)
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :])
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)







    train_dataset,train_labels = utils2.matcifar(imdb, train=True, d=3, medicinal=0).trainandtest()


    test_dataset,test_labels = utils2.matcifar(imdb, train=False, d=3, medicinal=0).trainandtest()

    del imdb

    imdbgraph = {}
    imdbgraph['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest + val_nTrain],
                                 dtype=np.float32)  # (9,9,100,n)
    imdbgraph['Labels'] = np.zeros([nTrain + nTest + val_nTrain], dtype=np.int64)
    imdbgraph['set'] = np.zeros([nTrain + nTest + val_nTrain], dtype=np.int64)
    RandPermgraph = train_indices + test_indices + val_train_indices
    RandPermgraph = np.array(RandPermgraph)
    for iSample in range(nTrain + nTest + val_nTrain):
        imdbgraph['data'][:, :, :, iSample] = data[Row[RandPermgraph[iSample]] - HalfWidth:  Row[RandPermgraph[
            iSample]] + HalfWidth + 1,
                                              Column[RandPermgraph[iSample]] - HalfWidth: Column[RandPermgraph[
                                                  iSample]] + HalfWidth + 1, :]
        imdbgraph['Labels'][iSample] = G[Row[RandPermgraph[iSample]], Column[RandPermgraph[iSample]]].astype(np.int64)
    imdbgraph['Labels'] = imdbgraph['Labels'] - 1  # 1-16 0-15
    imdbgraph['set'] = np.hstack((3 * np.ones([nTrain]), 3 * np.ones([nTest]), 3 * np.ones([val_nTrain]))).astype(
        np.int64)
    test_datasetgraph = utils2.matcifar(imdbgraph, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_datasetgraph, batch_size=3, shuffle=False, num_workers=0)

    del test_datasetgraph
    del imdbgraph



    imdb_val_train = {}
    imdb_val_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, val_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_val_train['Labels'] = np.zeros([val_nTrain], dtype=np.int64)
    imdb_val_train['set'] = np.zeros([val_nTrain], dtype=np.int64)

    da_RandPerm = np.array(val_train_indices)
    for iSample in range(val_nTrain):  #
        imdb_val_train['data'][:, :, :, iSample] = utils2.radiation_noise(data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_val_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_val_train_labels = imdb_val_train['Labels'] - 1  # 1-16 0-15
    imdb_val_train_data = imdb_val_train['data'].transpose((3, 2, 0, 1))
    imdb_val_train['set'] = np.ones([val_nTrain]).astype(np.int64)



    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],dtype=np.float32)  #
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)


    da_da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils2.radiation_noise(
            data[Row[da_da_RandPerm[iSample]] - HalfWidth:  Row[da_da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_da_RandPerm[iSample]] - HalfWidth: Column[da_da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_da_RandPerm[iSample]], Column[da_da_RandPerm[iSample]]].astype(np.int64)


    imdb_da_train_labels = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train_data = imdb_da_train['data'].transpose((3, 2, 0, 1))
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)


    train_datasets={}
    test_datasets={}
    imdb_val_trains={}

    imdb_da_trains={}

    for iSample in range(nTrain):
        if train_labels[iSample] not in train_datasets:
            train_datasets[train_labels[iSample]] = []
        train_datasets[train_labels[iSample]].append(train_dataset[iSample,:,:,:])

    for iSample in range(nTest):
        if test_labels[iSample] not in test_datasets:
            test_datasets[test_labels[iSample]] = []
        test_datasets[test_labels[iSample]].append(test_dataset[iSample,:,:,:])


    for iSample in range(val_nTrain):
        if imdb_val_train_labels[iSample] not in imdb_val_trains:
            imdb_val_trains[imdb_val_train_labels[iSample]] = []
        imdb_val_trains[imdb_val_train_labels[iSample]].append(imdb_val_train_data[iSample,:,:,:])


    for iSample in range(da_nTrain):
        if imdb_da_train_labels[iSample] not in imdb_da_trains:
            imdb_da_trains[imdb_da_train_labels[iSample]] = []
        imdb_da_trains[imdb_da_train_labels[iSample]].append(imdb_da_train_data[iSample,:,:,:])


    if partition=="train":
        return train_datasets,train_labels,test_loader,G,RandPermgraph,Row,Column
    elif partition=="test":
        return test_datasets,test_labels,test_loader,G,RandPermgraph,Row,Column
    else:
        return imdb_val_trains,imdb_val_train_labels,test_loader,G,RandPermgraph,Row,Column











