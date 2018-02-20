import numpy as np
from collections import OrderedDict
import pandas as pd
import os   
import scipy.stats
import scipy.cluster.vq as vq
import utils.data_processing as dp
import pickle
import sys
rg_install_dir = '/home/sanborn/packages/rg-toolbox/'
sys.path.append(rg_install_dir)
from rg_toolbox.core.invert_rg import invert_rg

def discretize_coeffs(coeffs, num_bins, disc_type='uniform'):
    discretized_coeffs = np.zeros(coeffs.shape)
    entropy = []
    for idx, row in enumerate(coeffs.T):
        if disc_type == 'uniform':
            hist, b_edges = np.histogram(row, num_bins)
            b = [np.mean([b_edges[idx], b_edges[idx+1]]) for idx in range(len(b_edges)-1)] + [b_edges[-1]] 
            b_idxs = np.digitize(row, b_edges) - 1
            disc = [b[idx] for idx in b_idxs]
            discretized_coeffs[:, idx] = disc
        elif disc_type == 'vq':
            codebook, distortion = vq.kmeans(row, num_bins)
            code, dist = vq.vq(row, codebook)
            disc = [codebook[i] for i in code]
            discretized_coeffs[:, idx] = disc
            hist, b_edges = np.histogram(discretized_coeffs, num_bins)
        else:
            raise ValueError("Undefined disc_type")  
        distribution = hist/np.sum(hist)
        entropy.append(scipy.stats.entropy(distribution))           
    H = np.mean(entropy)
    return discretized_coeffs, H

def rate_dist(params):
    rate_distortion = np.array(["model", "n_neurons", "overcompleteness", "cost", "lambda", "p_active", "mse", "mse_sd", "log_mse", "entropy", "transmission_rate", "nbins"], dtype="object")
    with np.load(params["input_dir"]) as d:
        D = d['arr_0'].item()['train']
        data = D.images
    for m in params["mod_names"]:
        for n in params["n_neurons"]:
            for c in params["costs"]:
                for idx, l in enumerate(params["lams"]):
                    name = m+'_'+str(n)+'_'+c+'_'+l+'_'+params['version'] if m == 'lca' else m        
                    if m == "rg":
                        logs = pickle.load(open(params["out_dir"], 'rb'))
                        coeffs = logs['coded_patches'].T #num_components x num_samples 
                        radial_scalings = logs['radial_scalings']
                        p_whitening = logs['p_whitening']
                    else:
                        with np.load(params["out_dir"]+name+'_coeffs.npz') as d:
                            coeffs = d['arr_0']
                        with np.load(params["out_dir"]+name+'_weights.npz') as d:
                            weights = d['arr_0']
                    active = np.mean(np.count_nonzero(coeffs, axis=1))/coeffs.shape[1]
                    over = coeffs.shape[1] / data.shape[1]
                    for b in params["n_bins"]:
                        discretized_coeffs, H = discretize_coeffs(coeffs, b, params["disc_type"])
                        if m == "rg":
                            reconstructions = invert_rg(discretized_coeffs.T, radial_scalings, p_whitening).T
                        else: 
                            reconstructions = np.matmul(discretized_coeffs, weights.T)
                        mse_per_img = np.mean(((data - reconstructions) ** 2), axis=1)
                        error = np.mean(mse_per_img)
                        error_sd = np.std(mse_per_img)
                        rate_distortion = np.vstack((rate_distortion, np.array([name, coeffs.shape[1], over, c, l, active, error, error_sd,  np.log(error), H, H * over, b])))
                        df = pd.DataFrame(rate_distortion[1:], columns=rate_distortion[0])
                        if params["print"]:
                            print(df) 
                        df.to_pickle(params["save_name"])  
    return df
