import numpy as np
from collections import OrderedDict
import pandas as pd
import os   
import scipy.stats
from sklearn import cluster
import scipy.cluster.vq as vq
import utils.data_processing as dp
import pickle
import sys
import math
import matplotlib
import matplotlib.pyplot as plt
rg_install_dir = '/home/sanborn/packages/rg-toolbox/'
sys.path.append(rg_install_dir)
from rg_toolbox.core.invert_rg import invert_rg
from skimage.measure import compare_ssim as ssim
import copy

def plot_coeff_hists(hists, bin_edges, params, img_idx, bins):
    plt.rcParams["figure.figsize"] = [100,100]
    fig, axes = plt.subplots(nrows=int(np.sqrt(params["n_neurons"])), ncols=int(np.sqrt(params["n_neurons"])))
    for i, ax in enumerate(axes.flat):
        ax.fill_between(bin_edges[i], np.concatenate(([0],hists[i])), step="pre")
        ax.xaxis.set_ticks(bin_edges[i])
        ax.xaxis.set_tick_params(width=2, length=15)
        labels = [""] * len(bin_edges[i])
        labels[0] = bin_edges[i][0] + bins/2
        labels[-1] = bin_edges[i][-1] - bins/2
        labels[int(len(labels)/2)] = bin_edges[i][int(len(labels)/2)] - bins/2
        ax.set_xticklabels(labels)
    fig.tight_layout()
    plt.savefig(params["plot_dir"]+"img_"+str(img_idx)+"_bw_"+str(bins)+".pdf")
    plt.close()
    
def discretize_coeffs(coeffs, bins, params, bf_length_factors=None, plot=False, crop_idxs=None, img_idx="unspecified"):
    coeffs = dp.reshape_data(coeffs, flatten=True)[0]
    discretized_coeffs = np.zeros(coeffs.shape)
    entropy = []
    nbins = []
    hists = []
    edges = []
    distributions = []
    for idx, row in enumerate(coeffs.T): 
        if bf_length_factors is not None:
            bin_width = bins * bf_length_factors[idx]
        else:
            bin_width = bins
        if params["set_fixed_range"]:
            num_neg_bins = math.ceil(abs(params["range_min"]) / bin_width)
            num_pos_bins = math.ceil(abs(params["range_max"]) / bin_width)
            positive_bins = [bin_width/2]
            negative_bins = [-bin_width/2]
            for n in range(num_neg_bins):
                negative_bins.append(negative_bins[-1]-bin_width)
            for n in range(num_pos_bins):
                positive_bins.append(positive_bins[-1]+bin_width)
            bin_edges = np.array(negative_bins[::-1] + positive_bins)
            bin_edges[-1] += 1e-6
            nbins.append(len(bin_edges)-1)
        else:
            coeff_range = np.max(row) - np.min(row)
            nbins = math.ceil(coeff_range / bin_width)
            if nbins == 0:
                bin_edges = np.linspace(-bin_width/2, bin_width/2, 2, dtype=np.float64)
            else:
                bin_edges = np.linspace(np.min(row), np.max(row), nbins+1, dtype=np.float64)
                bin_edges[-1] += 1e-6
        bin_centers = bin_edges[1:] - bin_width/2
        if params['disc_type'] == 'uniform':
            bin_centers = np.append(bin_centers, bin_centers[-1] + bin_width)
            bin_centers = np.insert(bin_centers, [0], bin_centers[0] - bin_width)
            bin_idxs = np.digitize(row, bin_edges)
            disc = [bin_centers[i] for i in bin_idxs]
            discretized_coeffs[:, idx] = disc
            if crop_idxs is not None:
                hist_batch = [x for i, x in enumerate(row) if i not in crop_idxs]
            else: 
                hist_batch = row
            hist = np.histogram(hist_batch, bin_edges, range=(params["range_min"], params["range_max"]))[0]
#         elif params['disc_type'] == 'lloyd':
#             bin_centers = bin_centers.reshape((-1, 1))
#             X = row.reshape((-1, 1))
#             k_means = cluster.KMeans(n_clusters=len(bin_centers), init=bin_centers)
#             k_means.fit(X)
#             codebook = k_means.cluster_centers_.squeeze()
#             code = k_means.labels_
#             disc = np.array([codebook[c] for i, c in enumerate(code)])
#             disc.shape = row.shape
#             discretized_coeffs[:, idx] = disc
#             codebook = sorted(list(codebook))
#             bin_edges = [(codebook[i] + codebook[i+1])/2 for i in range(len(codebook)-1)]
#             bin_edges.insert(0, np.min(row))
#             bin_edges.append(np.max(row))
#             hist, bin_edges = np.histogram(row, bin_edges)
        elif params['disc_type'] == 'lloyd_max':
            print("img: "+str(img_idx))
            print("bin_width: "+str(bins))
            print("coeff: "+str(idx))
            disc, bin_edges, bin_centers = lloyd_max(row, bin_centers)
            discretized_coeffs[:, idx] = disc
            hist, bin_edges = np.histogram(row, bin_edges)
        else:
            raise ValueError("Undefined disc_type")  
        distribution = hist/np.sum(hist)
        entropy.append(scipy.stats.entropy(distribution))        
    if plot:
        plot_coeff_hists(hists, edges, params, img_idx, bins)
    H = np.mean(entropy)
    nbins = np.mean(nbins)
    return discretized_coeffs, H, nbins

def lloyd_max_v1(coeffs, bin_edges, bin_centers, max_iter=1000):
    for i in range(max_iter+1):
        init_bin_edges = copy.deepcopy(bin_edges)
        for j, c in enumerate(bin_centers):
            vals = np.array([c for c in coeffs if c >= bin_edges[j] and c < bin_edges[j+1]])
            if len(vals) == 0:
                bin_centers[j] = (bin_edges[j] + bin_edges[j+1]) / 2
            else:
                bin_centers[j] = np.mean(vals)
        for j, e in enumerate(bin_edges):
            if e != bin_edges[0] and e != bin_edges[-1]:
                bin_edges[j] = (bin_centers[j] + bin_centers[j-1]) / 2
        if np.mean((bin_edges -  init_bin_edges)**2) < 1e-6:
            break
        print(i)
    if i == max_iter:
        print("max iter reached")
    bin_centers = np.insert(bin_centers, 0, bin_edges[0])
    bin_centers = np.append(bin_centers, bin_edges[-1])
    bin_idxs = np.digitize(coeffs, bin_edges)
    discretized_coeffs = np.array([bin_centers[i] for i in bin_idxs])
    return discretized_coeffs, bin_edges, bin_centers

def lloyd_max(coeffs, init_assignments_pts, epsilon=1e-4, EC=False, lam=.5):
    assert np.all(np.diff(init_assignments_pts) > 0)  # monotonically increasing
    assignment_pts = np.copy(init_assignments_pts)
#     MSE = 0.0
#     old_MSE = np.inf
#     H = np.inf
    cost = 0.0
    old_cost = np.inf
#     old_cost = old_MSE + lam * H if EC else MSE
    while np.abs(old_cost - cost) > epsilon:
        #^ this algorithm provably reduces MSE or leaves it unchanged at each
        #  iteration so the boundedness of MSE means this is a valid stopping
        #  criterion
        old_MSE = np.copy(cost)  
        bin_edges = np.hstack(
            [-np.inf, (assignment_pts[:-1] + assignment_pts[1:]) / 2, np.inf])
        if EC:
            hist, bin_edges = np.histogram(coeffs, bin_edges)
            dist = hist / np.sum(hist)
            bin_edges = [-np.inf, bin_edges[1:-1] - (lam *( np.log2(dist[:-1]) - np.log2(dist[1:]))) / (2 * (assignment_pts[:-1] - assignment_pts[1:])), np.inf]
#             bin_edges = np.array(bin_edges) - np.array(ec_terms)
        binned_vals = [[] for _ in range(len(init_assignments_pts))]
        for coeff in coeffs:
            bin_assignment = int(np.digitize(coeff, bin_edges, right=True) - 1)
            binned_vals[bin_assignment].append(coeff)
        for bin_idx in range(len(binned_vals)):
            if len(binned_vals[bin_idx]) != 0:   # can happen for low sample count
                assignment_pts[bin_idx] = np.mean(binned_vals[bin_idx])
              # otherwise don't update the assignment point
            quantized_coeffs = quantize(coeffs, bin_edges, assignment_pts)
            cost = np.mean(np.square(quantized_coeffs - coeffs))
            if EC:
                hist, bin_edges = np.histogram(coeffs, bin_edges)
                dist = hist / np.sum(hist)
                H = scipy.stats.entropy(dist)
                cost += lam * H      
    return quantize(coeffs, bin_edges, assignment_pts), bin_edges, assignment_pts

def quantize(raw_scalar_vals, edges, assignments):
    return assignments[np.digitize(raw_scalar_vals, edges, right=True) - 1]
    #^ using convention that intervals are open on the LHS and closed on the RHS

def unwhiten_bases(weights, method="pad_bf", filter_edge=1024):
    weight_edge = int(np.sqrt(weights.shape[0]))
    unwhite_weights = np.reshape(np.zeros(weights.T.shape), (weights.shape[1], weight_edge, weight_edge))
    min_coord = int(filter_edge/2 - weight_edge/2)
    max_coord = int(filter_edge/2 + weight_edge/2)
    lengths = []
    nyq = np.int32(np.floor(1024/2))
    freqs = np.linspace(-nyq, nyq-1, num=1024)
    fspace = np.meshgrid(freqs, freqs)
    rho = np.sqrt(np.square(fspace[0]) + np.square(fspace[1]))
    invw_filter = np.zeros(rho.shape)
    for i, a in enumerate(rho):
        for j, b in enumerate(a):
            invw_filter[i, j] = b ** -1 if b != 0 else 0
    if method == "crop_filter":
        invw_filter = invw_filter[min_coord:max_coord, min_coord:max_coord]
    for idx, w in enumerate(weights.T):
        if method == "pad_bf":
            weight = copy.deepcopy(w)
            bf = np.zeros((filter_edge, filter_edge)) 
            bf[min_coord:max_coord, min_coord:max_coord] = np.reshape(weight, (weight_edge, weight_edge))
        else:
            bf = np.reshape(copy.deepcopy(w), (weight_edge, weight_edge))
        unwhite = dp.unwhiten_data(bf[None, ..., None], invw_filter=invw_filter).squeeze()
        if method == "pad_bf":
            unwhite = unwhite[min_coord:max_coord, min_coord:max_coord]
        length = np.linalg.norm(unwhite)
        lengths.append(length)
        unwhite_weights[idx] = unwhite
    unwhite_weights = np.reshape(unwhite_weights, weights.shape).T
    invbflengths = 1/np.array(lengths)
    length_factors = .9 *(invbflengths - np.min(invbflengths)) / (np.max(invbflengths)-np.min(invbflengths)) + .1
    return unwhite_weights, lengths, length_factors, invw_filter
    
def discretize_and_recon(coeffs, weights, data, data_mean, params, m, b, data_range, bf_length_factors=None, crop_idxs=None, img_idx="unspecified"):
    discretized_coeffs, h, num_bins = discretize_coeffs(coeffs, b, params, bf_length_factors, params["plot_coeff_hists"], crop_idxs, img_idx)
    if m == "pca":
        reconstructions = np.matmul(discretized_coeffs, np.linalg.inv(weights))
    else:
        reconstructions = np.matmul(discretized_coeffs, weights.T)
    reconstructions = dp.reshape_data(reconstructions, flatten=False)[0]
    if params["white"] == True:
        reconstructions = dp.unwhiten_data(reconstructions, data_mean[img_idx], method=params["whiten_method"])  
    else: 
        reconstructions = dp.patches_to_image(reconstructions, [1, params['im_edge_size'], params['im_edge_size'], 1])
        reconstructions += data_mean[img_idx]
    if "crop_border" in params and params["crop_border"]:
        reconstructions, crop_idxs = crop_border(reconstructions, params)
    mse = np.mean((data[img_idx] - reconstructions) ** 2)
    mse_sd = np.std(mse)
    psnr = data_range[img_idx] ** 2 / mse
    psnr_db = 10 * np.log10(psnr)
    struc_sim = ssim(data[img_idx].squeeze(), reconstructions.squeeze(), data_range=data_range[img_idx])
    return mse, mse_sd, psnr, psnr_db, struc_sim, h, num_bins

def crop_border(data, params):
    patches_per_side, patches_per_img = img_patch_count(params)
    crop_idxs = [(patches_per_side-1) + patches_per_side * x for x in range(patches_per_side-1)] + list(range(patches_per_img-patches_per_side, patches_per_img))
    cropped_size = params['im_edge_size'] - params["patch_edge_size"]
    data = dp.reshape_data(data, flatten=False)[0]
    data = data[:, :cropped_size, :cropped_size, None]
    return data, crop_idxs

def img_patch_count(params):
    patches_per_side = int(params['im_edge_size'] / params["patch_edge_size"])
    patches_per_img = int(params["im_edge_size"] ** 2 / params["patch_edge_size"] ** 2)
    return patches_per_side, patches_per_img

def num_active(coeffs):
    n_active = np.mean(np.count_nonzero(coeffs, axis=1))
    p_active = n_active/coeffs.shape[1]
    return n_active, p_active
    
def rate_dist(params, w_filter=None, crop_idxs=None, bf_length_factors=None):
    df_headers = ["model", "n_neurons", "overcompleteness", "cost", "lambda", "p_active", "n_active", "mse", "mse_sd", "log_mse", "psnr", "psnr_db", "ssim", "entropy", "transmission_rate", "bin_width", "num_bins", "bin_ratio"]
    rate_distortion = []
    patches_per_side, patches_per_img = img_patch_count(params)
    with np.load(params["input_dir"]) as d:
        D = d['arr_0'].item()['train']
        data = D.images
    if params["white"] == True:
        data = dp.unwhiten_data(data, D.data_mean, method=params["whiten_method"])
    else:
        data = dp.reshape_data(data, flatten=False)[0]
        data = dp.patches_to_image(data, [params["num_images"], params['im_edge_size'], params['im_edge_size'], 1])
        data += D.data_mean
    if "crop_border" in params and params["crop_border"]:
        data, crop_idxs = crop_border(data, params)
    data_range = np.max(data, axis=(1,2)) - np.min(data, axis=(1,2))
    for m in params["mod_names"]:
        for n in params["n_neurons"]:
            for c in params["costs"]:
                for idx, l in enumerate(params["lams"]):
                    name = m+'_'+str(n)+'_'+c+'_'+str(l)+'_'+params['version'] if m == 'lca' else m
                    if name == "ica":
                        name += "_" + params["version"]
                    with np.load(params["out_dir"]+name+'_coeffs.npz') as d:
                        coeffs = d['arr_0']
                    with np.load(params["out_dir"]+name+'_weights.npz') as d:
                        weights = d['arr_0']
                    if params["adjust_bf_length"] == True:
                        unwhite_weights, lengths, bf_length_factors, invw_filter = unwhiten_bases(weights)
                    print((np.min(coeffs), np.max(coeffs)))
                    n_active, p_active = num_active(coeffs)
                    overcompleteness = params["patch_edge_size"]**2 / coeffs.shape[1]
                    for b in params["bins"]:
                        H = []; MSE = []; SSIM = []; MSE_SD = []; PSNR = []; PSNR_DB = []
                        for i in range(params["num_images"]):
                            img_coeffs = coeffs[i * patches_per_img:(i+1) * patches_per_img]
                            mean_coeff_range = np.mean(np.max(img_coeffs, axis=0) - np.min(img_coeffs, axis=0))
                            mse, mse_sd, psnr, psnr_db, struc_sim, h, num_bins = discretize_and_recon(img_coeffs, weights, data, D.data_mean, params, m, b, data_range, bf_length_factors, crop_idxs, i)
                            SSIM.append(float(struc_sim)); H.append(float(h)); MSE.append(float(mse)); PSNR.append(float(psnr)); PSNR_DB.append(float(psnr_db)); MSE_SD = float(np.std(MSE)); LOG_MSE = [np.log(m) for m in MSE]
                        rate_distortion.append([name, coeffs.shape[1], overcompleteness, c, l, p_active, n_active, MSE, MSE_SD, LOG_MSE, PSNR, PSNR_DB, SSIM, H, float(np.mean(H) * overcompleteness), b, num_bins, b/mean_coeff_range])
                        df = pd.DataFrame(rate_distortion, columns=df_headers)
                        if params["print"]:
                            print(df) 
                        df.to_pickle(params["save_name"])  
    return df


#                         mean_coeff_range = np.mean(np.max(coeffs, axis=0) - np.min(coeffs, axis=0))

# def rate_dist(params):
#     w_filter = None
#     bf_length_factors = None
#     crop_idxs = None
#     patches_per_side = int(params['im_edge_size'] / params["patch_edge_size"])
#     patches_per_img = int(params["im_edge_size"] ** 2 / params["patch_edge_size"] ** 2)
#     df_headers = ["model", "n_neurons", "overcompleteness", "cost", "lambda", "p_active", "n_active", "mse", "mse_sd", "log_mse", "psnr", "psnr_db", "ssim", "entropy", "transmission_rate", "bin_width", "num_bins", "bin_ratio"]
#     rate_distortion = []
#     with np.load(params["input_dir"]) as d:
#         D = d['arr_0'].item()['train']
#         data = D.images
#     if params["white"] == True:
#         data = dp.unwhiten_data(data, D.data_mean, method=params["whiten_method"])
#     else:
#         data = dp.reshape_data(data, flatten=False)[0]
#         data = dp.patches_to_image(data, [params["num_images"], params['im_edge_size'], params['im_edge_size'], 1])
#         data += D.data_mean
#     if "crop_border" in params and params["crop_border"] is not None:
#         crop_idxs = [(patches_per_side-1) + patches_per_side * x for x in range(patches_per_side-1)] + list(range(patches_per_img-patches_per_side, patches_per_img))
#         cropped_size = params['im_edge_size'] - params["patch_edge_size"]
#         data = data[:, :cropped_size, :cropped_size, None]
#     data_range = np.max(data, axis=(1,2)) - np.min(data, axis=(1,2))
#     whole_data_range = np.max(data) - np.min(data)
#     for m in params["mod_names"]:
#         for n in params["n_neurons"]:
#             for c in params["costs"]:
#                 for idx, l in enumerate(params["lams"]):
#                     name = m+'_'+str(n)+'_'+c+'_'+str(l)+'_'+params['version'] if m == 'lca' else m
#                     if name == "ica":
#                         name += "_" + params["version"]
#                     with np.load(params["out_dir"]+name+'_coeffs.npz') as d:
#                         coeffs = d['arr_0']
#                         n_coeffs = coeffs.shape[1]
#                     with np.load(params["out_dir"]+name+'_weights.npz') as d:
#                         weights = d['arr_0']
#                     if params["adjust_bf_length"] == True:
#                         bf_length_factors = unwhiten_bases(weights)
#                     print(np.min(coeffs))
#                     print(np.max(coeffs))
#                     n_active = np.mean(np.count_nonzero(coeffs, axis=1))
#                     p_active = np.mean(np.count_nonzero(coeffs, axis=1))/coeffs.shape[1]
#                     over = coeffs.shape[1] / n_coeffs
#                     for b in params["bins"]:
#                         H = []; MSE = []; SSIM = []; MSE_SD = []; PSNR = []; PSNR_DB = []
#                         for i in range(params["num_images"]):
#                             img_coeffs = coeffs[i * patches_per_img:(i+1) * patches_per_img]
#                             discretized_coeffs, h, num_bins = discretize_coeffs(img_coeffs, b, params, bf_length_factors, params["plot_coeff_hists"], crop_idxs, i)
#                             if m == "ica":
#                                 reconstructions = np.matmul(discretized_coeffs, weights.T)
#                             elif m == "pca":
#                                 reconstructions = np.matmul(discretized_coeffs, np.linalg.inv(weights))
#                             else:
#                                 reconstructions = np.matmul(discretized_coeffs, weights.T)
#                             discretized_coeffs = dp.reshape_data(discretized_coeffs, flatten=False)[0]
#                             reconstructions = dp.reshape_data(reconstructions, flatten=False)[0]
#                             if params["white"] == True:
#                                 reconstructions = dp.unwhiten_data(reconstructions, D.data_mean[i], method=params["whiten_method"])  
#                             else: 
#                                 reconstructions = dp.patches_to_image(reconstructions, [1, params['im_edge_size'], params['im_edge_size'], 1])
#                                 reconstructions += D.data_mean[i]
#                             if "crop_border" in params and params["crop_border"] is not None:
#                                 reconstructions = reconstructions[:, :cropped_size, :cropped_size, :]
#                             mse = np.mean((data[i] - reconstructions) ** 2)
#                             mse_sd = np.std(mse)
#                             psnr = data_range[i] ** 2 / mse
#                             psnr_db = 10 * np.log10(psnr)
#                             SSIM.append(ssim(data[i].squeeze(), reconstructions.squeeze(), data_range=data_range[i]))
#                             H.append(float(h))
#                             MSE.append(float(mse))
#                             PSNR.append(float(psnr))
#                             PSNR_DB.append(float(psnr_db))
#                         MSE_SD = float(np.std(MSE))
#                         LOG_MSE = [np.log(m) for m in MSE]
#                         mean_coeff_range = np.mean(np.max(coeffs, axis=0) - np.min(coeffs, axis=0))
#                         rate_distortion.append([name, coeffs.shape[1], over, c, l, p_active, n_active, MSE, MSE_SD,  LOG_MSE, PSNR, PSNR_DB, SSIM, H, float(np.mean(H) * over), b, num_bins, b/mean_coeff_range])
#                         df = pd.DataFrame(rate_distortion, columns=df_headers)
#                         if params["print"]:
#                             print(df) 
#                         df.to_pickle(params["save_name"])  
#     return df


# def rate_dist(params):
#     w_filter = None
#     rate_distortion = np.array(["model", "n_neurons", "overcompleteness", "cost", "lambda", "p_active", "n_active", "mse", "mse_sd", "log_mse", "entropy", "transmission_rate", "bin_width", "num_bins", "bin_ratio"], dtype="object")
#     with np.load(params["input_dir"]) as d:
#         D = d['arr_0'].item()['train']
#     if params["white"] == True:
#         data = dp.unwhiten_data(D.images, D.data_mean, D.w_filter, method=params["whiten_method"])
#         w_filter = D.w_filter
# #         data, _, _ = dp.standardize_data(data)
#     else: 
#         data = D.images
#     for m in params["mod_names"]:
#         for n in params["n_neurons"]:
#             for c in params["costs"]:
#                 for idx, l in enumerate(params["lams"]):
#                     name = m+'_'+str(n)+'_'+c+'_'+l+'_'+params['version'] if m == 'lca' else m        
#                     if m == "rg":
#                         logs = pickle.load(open(params["out_dir"], 'rb'))
#                         coeffs = logs['coded_patches'].T #num_components x num_samples 
#                         radial_scalings = logs['radial_scalings']
#                         p_whitening = logs['p_whitening']
#                     else:
#                         with np.load(params["out_dir"]+name+'_coeffs.npz') as d:
#                             coeffs = d['arr_0']
#                         with np.load(params["out_dir"]+name+'_weights.npz') as d:
#                             weights = d['arr_0']
#                     n_active = np.mean(np.count_nonzero(coeffs, axis=1))
#                     p_active = np.mean(np.count_nonzero(coeffs, axis=1))/coeffs.shape[1]
#                     over = coeffs.shape[1] / data.shape[1]
#                     for b in params["bins"]:
#                         discretized_coeffs, H, num_bins = discretize_coeffs(coeffs, b, params, w_filter)
#                         if m == "rg":
#                             reconstructions = invert_rg(discretized_coeffs.T, radial_scalings, p_whitening).T
#                         elif m == "ica_v1.0":
#                             reconstructions = np.matmul(discretized_coeffs, weights.T)
#                         elif m == "pca":
#                             with np.load(params["out_dir"]+"patch_means.npz") as d:
#                                 patch_means = d["arr_0"]
#                             reconstructions = np.matmul(discretized_coeffs, np.linalg.inv(weights)) + patch_means
#                         else:
#                             reconstructions = np.matmul(discretized_coeffs, weights.T)
#                         if params["white"] == True:
#                             reconstructions = dp.unwhiten_data(reconstructions, D.data_mean, D.w_filter, method=params["whiten_method"])
# #                             reconstructions, _, _ = dp.standardize_data(data)
#                         mse_per_img = np.mean(((data - reconstructions) ** 2), axis=1)
#                         error = np.mean(mse_per_img)
#                         error_sd = np.std(mse_per_img)
#                         mean_coeff_range = np.mean(np.max(coeffs, axis=0) - np.min(coeffs, axis=0))
#                         rate_distortion = np.vstack((rate_distortion, np.array([name, coeffs.shape[1], over, c, l, p_active, n_active, float(error), error_sd,  np.log(error), float(H), float(H * over), b, num_bins, b/mean_coeff_range])))
#                         df = pd.DataFrame(rate_distortion[1:], columns=rate_distortion[0])
#                         if params["print"]:
#                             print(df) 
#                         df.to_pickle(params["save_name"])  
#     return df



#     with np.load(params["input_dir"]) as d:
#         D = d['arr_0'].item()['train']
#         data = D.images
#     orig_img = dp.unwhiten_data(data, D.data_mean, D.w_filter, method=params["whiten_method"])
#     reshape = (data.shape[0], params['patch_edge_size'], params['patch_edge_size'], 1)
# #     orig_img = np.reshape(orig_img, reshape) 
# #     orig_img = dp.patches_to_image(orig_img, [25, params['im_edge_size'], params['im_edge_size'], 1])
#     orig_img = orig_img[params['img_idx']]
#     data_range = np.max(orig_img) - np.min(orig_img)
#     ## Load in model coeffs and weights
#     name = params['model_type']+'_'+str(params['n_neurons'])+'_'+params['cost']+'_'+str(params['lam'])+'_'+params['version']
#     with np.load(params["out_dir"]+name+'_coeffs.npz') as d:
#         coeffs = d["arr_0"]
#     with np.load(params["out_dir"]+name+'_weights.npz') as d:
#         weights = d["arr_0"]
#     p_active =  str(round((np.count_nonzero(coeffs) / (coeffs.shape[0] * coeffs.shape[1])), 4)*100) + '%'
#     if params["adjust_bf_length"] == True:
#         bf_lengths = rd.unwhiten_bases(weights)
#     else:
#         bf_lengths = None
#     ## Create plot
#     plt.rcParams["figure.figsize"] = params["fig_size"]
#     fig, axes = plt.subplots(nrows=len(params["bins"])+1)
#     axes[0].imshow(orig_img.squeeze(), cmap="Greys_r")
#     axes[0].set_title(name + '\n p active: ' + p_active + '\n \n ' + 'original image')
#     recon = np.matmul(coeffs, weights.T)
#     recon = np.reshape(recon, reshape)
# #     recon = dp.patches_to_image(recon, [25, params['im_edge_size'], params['im_edge_size'], 1])
# #     recon_mean = np.mean(recon, axis=1)[:, None]
# #     recon -= recon_mean
# #     recon = dp.extract_patches(recon, reshape)
#     recon = dp.unwhiten_data(recon, D.data_mean, D.w_filter, method=params["whiten_method"])
# #     recon += recon_mean
# #     recon = np.reshape(recon, reshape)
# #     recon = dp.patches_to_image(recon, [25, params['im_edge_size'], params['im_edge_size'], 1])
#     recon = recon[params['img_idx']]  
#     axes[1].imshow(recon.squeeze(), cmap="Greys_r")
#     # Generate reconstructions for quantized coeffs and plot
#     MSE = []; PSNR = []; MSSIM = []
#     for i, b in enumerate(params['bins']):
#         disc_coeffs, H, num_bins = rd.discretize_coeffs(coeffs, b, params, bf_lengths)
#         recon = np.matmul(disc_coeffs, weights.T)
#         recon = np.reshape(recon, reshape)
# #         recon = dp.patches_to_image(recon, [25, params['im_edge_size'], params['im_edge_size'], 1])
# #         recon_mean = np.mean(recon, axis=1)[:, None]
# #         recon -= recon_mean
# #         recon = dp.extract_patches(recon, reshape)
#         recon = dp.unwhiten_data(recon, D.data_mean, method=params["whiten_method"])
# #         recon = np.reshape(recon, reshape)
# #         recon = dp.patches_to_image(recon, [25, params['im_edge_size'], params['im_edge_size'], 1])
# #         recon += recon_mean
#         recon = recon[params['img_idx']]
#         axes[i+1].imshow(recon.squeeze(), cmap="Greys_r")
#         mse = np.mean((orig_img - recon)**2)
#         psnr = 'na'
#         struc_sim = ssim(orig_img.squeeze(), recon.squeeze(), data_range=data_range)
# #         psnr = np.mean(10 * np.log10(np.max(orig_img) ** 2 / mse)) 
#         MSE.append(mse); PSNR.append(psnr); MSSIM.append(struc_sim)
