import numpy as np
import scipy as sp

def RND(d, input_weights=None):
    patch_means = d.mean(axis=(1))[:,None]
    data = d - patch_means
    C = np.cov(data.T)   
    U, S, V = np.linalg.svd(C)
    S = 1/np.sqrt(S + 1e-6)
    S = np.diag(S)
    rnd = np.random.standard_normal(C.shape)
    if input_weights is None:
        weights = np.matmul(np.linalg.inv(sp.linalg.sqrtm(np.matmul(rnd, rnd.T))), rnd)
        weights = np.matmul(np.matmul(weights, S), U.T)
    else:
        weights = input_weights
    coeffs = np.matmul(data, weights)
    return weights, coeffs, patch_means

# def SYM(data):
# #     data_mean = d.mean(axis=(1))[:,None]
# #     data = d - data_mean
#     C = np.cov(data.T)
#     C += np.identity(C.shape[0]) * 1e-3
#     Wsym = np.linalg.inv(sp.linalg.sqrtm(C))
#     u, s, v = np.linalg.svd(Wsym)
#     Wsym = Wsym / np.prod(s ** (1 / len(s)))
#     coeffs = np.matmul(data, Wsym)
#     recon = np.matmul(coeffs, Wsym.T)
#     return Wsym, coeffs, recon

def SYM(d, input_weights=None):
    patch_means = d.mean(axis=(1))[:,None]
    data = d - patch_means
    C = np.cov(data.T)   
    U, S, V = np.linalg.svd(C)
    S = 1/np.sqrt(S + 1e-6)
    S = np.diag(S)
    if input_weights is None:
        weights = np.matmul(U, np.matmul(S, U.T))
    else:
        weights = input_weights
    coeffs = np.matmul(data, weights)
    return weights, coeffs, patch_means

def HAAR(d):
    patch_means = d.mean(axis=(1))[:,None]
    data = d - patch_means    
    n = np.sqrt(data.shape[1])
    level = int(np.log2(n/2)) + 1
    H = np.array([1])
#   NC = 1/np.sqrt(2) option for normalization
    NC = 1
    LP = [1, 1]
    HP = [1, -1]
    for i in range(level):
        H = NC * np.vstack((np.kron(H,LP), np.kron(np.identity(len(H)),HP)))
    weights = np.kron(H, H)
#     weights = H.T
#     coeffs = np.matmul(data, np.linalg.inv(weights))  
    coeffs = np.matmul(data, weights)
    return weights, coeffs, patch_means

def PCA(data, method='orthogonal', input_weights=None):
#     patch_means = d.mean(axis=(1))[:,None]
#     data = d - patch_means
    C = np.cov(data.T)   
    U, S, V = np.linalg.svd(C)
    S = 1/np.sqrt(S + 1e-6)
    S = np.diag(S)
    weights = U
    if method == 'orthogonal':
        coeffs = np.matmul(data, weights) 
    elif method == 'whitening':
#         whitening_matrix = np.matmul(np.matmul(U, S), U.T)
        if input_weights is None:
            weights = np.matmul(S, U.T)
        else:
            weights = input_weights
        coeffs = np.matmul(data, weights)
    return weights, coeffs

def recon(weights, coeffs, patch_means):
    return np.matmul(coeffs, np.linalg.inv(weights)) + patch_means