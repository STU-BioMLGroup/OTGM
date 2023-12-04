import os, sys
import numpy as np
import scipy.io as sio
import util
import torch

from sklearn.utils import shuffle


def load_data(config):
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []
    mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
    if data_name in ['Scene_15']:
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['HandWritten']:
        X = mat['X'][0]
        for view in [0, 2]:
            x = X[view]
            x = util.normalize(x).astype('float32')
            X_list.append(x)
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)

    elif data_name in ['Caltech101-7', 'Caltech101-20']:
        X = mat['X'][0]
        for view in [3, 4]:
            x = X[view]
            x = util.normalize(x).astype('float32')
            X_list.append(x)
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)

    elif data_name in ['BDGP']:
        x1 = mat['X1']
        x2 = mat['X2']
        X_list.append(util.normalize(x1).astype('float32'))  # (1449,2048)
        X_list.append(util.normalize(x2).astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)

    elif data_name in ['Reuters_dim10']:
        X = mat['x_train']
        y = np.squeeze(mat['y_train']).astype('int')
        idx = np.argsort(y)
        y = y[idx]
        for view in [0, 1]:
            x = X[view][idx]
            x = util.normalize(x).astype('float32')
            X_list.append(x)
        Y_list.append(y)

    return X_list, Y_list,


def get_aligned(num_sample, aligned_ratio, random_state=2):
    """
    根据样本数量和对齐率进行分配
        inputs:
            num_sample: 样本数量
            aligned_ratio: 对齐率
        returns:
            flag: 长度为n的分配列表，其中 True: 表示该索引位置两个视图的数据是对齐的，False: 表示该索引位置两个视图的数据不是对齐的
    """
    num_aligned = int(num_sample * aligned_ratio)
    # 打乱索引
    index = np.linspace(0, num_sample - 1, num_sample, dtype=int)
    index = shuffle(index, random_state=random_state)
    # 分配, 取打乱后的前一部分索引位置作为对齐数据
    flag = index < 0
    flag[index[:num_aligned, ]] = True
    return flag


def shuffle_data(x1_train, x2_train, flag, device, random_state=2):
    """
        inputs:
            x1_train: 视图1的数据
            x2_train: 视图2的数据
            flag: 对齐和非对齐的分配，长度为n的分配列表，其中 True: 表示该索引位置两个视图的数据是对齐的，False: 表示该索引位置两个视图的数据不是对齐的
        returns:
            x1: 视图1的数据
            x2: 视图2的数据
            P_index: 打乱的索引
            index_mis_aligned: 非对齐部分的索引
            P_gt: 真实的转换矩阵
    """
    num_sample = x1_train.shape[0]
    P_index = np.linspace(0, num_sample - 1, num_sample, dtype=int)
    index_mis_aligned = shuffle(P_index[~flag], random_state=random_state)
    P_index[~flag] = index_mis_aligned
    P_gt = np.eye(num_sample).astype('float32')
    P_gt = P_gt[:, P_index]
    P_gt = torch.from_numpy(P_gt).to(device)
    x1 = x1_train
    x2 = x2_train[P_index]
    return x1, x2, P_index, index_mis_aligned, P_gt

def get_mis_aligned(x1_train, x2_train, flag, device):
    num_sample = x1_train.shape[0]
    # 对齐
    P_index = np.linspace(0, num_sample - 1, num_sample, dtype=int)
    index_mis_aligned = shuffle(P_index[~flag], random_state=2)
    P_index[~flag] = index_mis_aligned
    P_gt = np.eye(num_sample).astype('float32')
    P_gt = P_gt[:, P_index]
    P_gt = torch.from_numpy(P_gt).to(device)
    x1_eval = x1_train
    x2_eval = x2_train[P_index]
    return x1_eval, x2_eval, P_index, index_mis_aligned, P_gt

