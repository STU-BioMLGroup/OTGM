import logging
import random

import numpy as np
import math
from sklearn.utils import shuffle
import torch
import scipy.io as sio
import seaborn as sns
import matplotlib.pyplot as plt

def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def next_batch_aligned(x1_train, x2_train, flag, batch_size, device='cuda'):
    num_sample = x1_train.shape[0]
    index = np.linspace(0, num_sample - 1, num_sample, dtype=int)
    index_aligned, index_mis_aligned = shuffle(index[flag]), shuffle(index[~flag])
    num_aligned, num_mis_aligned = len(index_aligned), len(index_mis_aligned)
    # 对齐和未对齐的数据
    x1_aligned, x2_aligned = x1_train[index_aligned], x2_train[index_aligned]
    x1_mis_aligned, x2_mis_aligned = x1_train[index_mis_aligned], x2_train[index_mis_aligned]
    P_index = shuffle(np.linspace(0, num_mis_aligned - 1, num_mis_aligned, dtype=int))
    x2_mis_aligned = x2_mis_aligned[P_index]
    # 循环次数
    total = math.ceil(num_aligned / batch_size)
    # 未对齐的每次取的个数
    batch_size_mis_aligned = math.ceil(num_mis_aligned / total)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(num_aligned, end_idx)
        start_idx1 = i * batch_size_mis_aligned
        end_idx1 = (i + 1) * batch_size_mis_aligned
        end_idx1 = min(num_mis_aligned, end_idx1)
        # 批量对齐数据
        batch_x1_aligned = x1_aligned[start_idx: end_idx, ...]
        batch_x2_aligned = x2_aligned[start_idx: end_idx, ...]
        batch_x1_mis_aligned = x1_mis_aligned[start_idx1: end_idx1, ...]
        batch_x2_mis_aligned = x2_mis_aligned[start_idx1: end_idx1, ...]
        yield (batch_x1_aligned, batch_x2_aligned, batch_x1_mis_aligned, batch_x2_mis_aligned, (i + 1))

def cal_std(logger, accumulated_metrics):
    """Return the average and its std"""
    logger.info('ACC:'+ str(accumulated_metrics['acc']))
    logger.info('NMI:'+ str(accumulated_metrics['nmi']))
    logger.info('ARI:'+ str(accumulated_metrics['ari']))
    output = """ ACC {:.4f} NMI {:.4f} ARI {:.4f}""".format(
        np.mean(accumulated_metrics['acc']),
        np.mean(accumulated_metrics['nmi']),
        np.mean(accumulated_metrics['ari']))
    logger.info(output)

    return

def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def to_tensor(x, device='cuda'):
    return torch.from_numpy(x.astype(np.float32)).to(device)

def to_numpy(x):
    return x.cpu().numpy()