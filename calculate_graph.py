import numpy as np
import torch
from util import to_numpy, to_tensor
from sklearn.preprocessing import normalize

# 计算度矩阵
def calculate_degree_matrix(similarity_matrix):
    degree_matrix = torch.diag(torch.sum(similarity_matrix, dim=1))
    return degree_matrix

def knn(matrix, k=10, largest=True):
    # 取出每一行前k个最大值的索引
    _, indices = torch.topk(matrix, k=k, dim=1, largest=largest, sorted=True)
    # 将其他元素置零
    mask = torch.zeros_like(matrix)
    mask.scatter_(1, indices, 1)
    matrix_knn = matrix * mask
    # 返回保留前k个元素后的矩阵
    return matrix_knn

# 计算拉普拉斯矩阵
def calculate_laplacian(similarity_matrix, k=10):
    similarity_matrix = (similarity_matrix + similarity_matrix.t()) * 0.5
    if k > 0:
        similarity_matrix = knn(similarity_matrix, k=k)
    similarity_matrix = (similarity_matrix + similarity_matrix.t()) * 0.5
    # 2. 计算度矩阵
    degree_matrix = calculate_degree_matrix(similarity_matrix)

    # 3.计算拉普拉斯矩阵
    laplacian_matrix = degree_matrix - similarity_matrix

    return laplacian_matrix

def calculate_cosine_similarity(x1, x2):
    x1_ = normalize(x1, axis=1)
    x2_ = normalize(x2, axis=1)
    similarity = np.matmul(x1_, x2_.T)
    return similarity

def calculate_graphs(fea1, fea2, device='cuda'):
    fea1 = to_numpy(fea1)
    fea2 = to_numpy(fea2)
    coef1 = calculate_cosine_similarity(fea1, fea1)
    coef2 = calculate_cosine_similarity(fea2, fea2)
    return to_tensor(coef1, device), to_tensor(coef2, device)