import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import utils.data_processing as dp
import utils.rate_distortion as rd
import utils.log_parser as lp
import pandas as pd
import seaborn as sns
import h5py
import pickle
import sys
from skimage.measure import compare_ssim as ssim
rg_install_dir = '/home/sanborn/packages/rg-toolbox/'
sys.path.append(rg_install_dir)
from rg_toolbox.core.invert_rg import invert_rg

"""""""""
RD ANALYSIS PLOTS
"""""""""
def alt_models_plots(data, x='entropy', y='mse', title=None):
    p = sns.lmplot(x, y, data=data, hue='model', fit_reg=False, scatter_kws={"s": 5})
    p.map(plt.plot, x, y, marker="o", ms=5)
    axes = p.axes.flatten()
    axes[0].set_xlim(0,)
    if y == 'mse':
        axes[0].set_ylim(0,)
    if title is not None:
        p.fig.suptitle(title, size=30)
        
def make_legend_labels(rd_table):
    p_active = [round(x, 2) for x in pd.unique(rd_table.p_active)]
    labels = [p_active[n*5:(n+1)*10] for n in range(0,6)]
    labels = {x: labels[x] for x in range(0,6)}
    return labels

def full_comparison_plots(lca_table, alt_1=None, alt_2=None, alt_3=None, alt_4=None, x='entropy', y='mse', ylim=.4, xlim=5, row='n_neurons', col='cost', hue='lambda', title=None):
    if hue == 'lambda':
        labels_key = make_legend_labels(lca_table)
    p = sns.lmplot(x, y, data=lca_table.where(lca_table.n_bins>1), row=row, col=col,  hue=hue, fit_reg=False, scatter_kws={"s": 3})
    p.map(plt.plot, x, y, marker="o", ms=2)
    if title is not None:
        plt.subplots_adjust(top=0.9)
        p.fig.suptitle(title, size=30)
    axes = p.axes.flatten()
    for idx, a in enumerate(axes):
        if y == 'mse':
            a.set_ylim(0, ylim)
            a.set_xlim(0, xlim)
        if hue == 'lambda':
            handles, labels = a.get_legend_handles_labels()
            labels = ['p_active: ' + str(labels_key[idx][i]) for i, x in enumerate(labels[0:5])]     
            a.legend(handles, labels)
        if alt_1 is not None:
            a.plot(alt_1[x], alt_1[y].where(alt_1.n_bins>1), marker=".", color="black", linestyle="dashed")
        if alt_2 is not None:
            a.plot(alt_2[x], alt_2[y].where(alt_2.n_bins>1), marker="+", color="black", linestyle="dotted") 
        if alt_3 is not None:
            a.plot(alt_3[x], alt_3[y].where(alt_3.n_bins>1), marker="^", color="black", linestyle=":", ms=2) 
        if alt_4 is not None:
            a.plot(alt_4[x], alt_4[y].where(alt_4.n_bins>1), marker="s", color="black", linestyle="-.", ms=2) 
            
def plot_coeff_hists(params):
    fig, axes = plt.subplots(nrows=len(params["lams"]), ncols=len(params["costs"]), squeeze=False, figsize=params["figsize"])
    for m in params["mod_names"]:
        for n in params["n_neurons"]:
            for j, c in enumerate(params["costs"]):
                for i, l in enumerate(params["lams"]):
                    if m == "lca":
                        mod_label = m+'_'+str(n)+'_'+c+'_'+l+'_'+params['version']
                    else: 
                        mod_label = m+'_'+params["version"]
                    if m == "rg":
                        logs = pickle.load(open(params['out_dir'], 'rb'))
                        coeffs = logs['coded_patches']
                    else:
                        with np.load(params["out_dir"]+'coeffs/'+mod_label+'_coeffs.npz') as d:  
                            coeffs = d['arr_0']
                    axes[i][j].hist(coeffs[0:1000].ravel(), bins=100)
                    axes[i][j].set_yticks([])
                    axes[i][j].set_xlim(params["xlim"])
                    if c is not None:
                        axes[i][j].set_title('cost: ' + c + ' | lambda: ' + l)
                    
def batch_plot_model_grads(params):
    global_batch_index = []; gradients = []; key = []; model = []; lam = []; cost = []
    m = params["model_type"]
    for n in params["n_neurons"]:
        for c in params["costs"]:
            for idx, l in enumerate(params["lams"]):
                mod_label = m+'_'+str(n)+'_'+c+'_'+l+'_'+params["version"] if m == 'lca' else m
                log = lp.read_stats(lp.load_file(params["out_dir"]+mod_label+'/logfiles/'+mod_label+'.log')) 
                global_batch_index.extend(log['global_batch_index'])
                global_batch_index.extend(log['global_batch_index'])
                if m == "lca":
                    weights_max = log['phi_max_grad'] 
                    weights_min = log['phi_min_grad']
                elif m == "ica":
                    weights_max = log['a_max_grad'] 
                    weights_min = log['a_min_grad']
                gradients.extend(weights_max + weights_min)
                model.extend([mod_label] * 2 * len(weights_max))
                key.extend(['max'] * len(weights_max) + ['min'] * len(weights_min))
                cost.extend([c] * 2 * len(weights_max))
                lam.extend([l] * 2 * len(weights_max))
    stats = pd.DataFrame({'model': model, 'global_batch_index': global_batch_index, 'gradients': gradients, 'key': key, 'cost': cost, 'lambda': lam})
    sns.set_style("darkgrid")
    if m == "lca":
        p = sns.lmplot('global_batch_index', 'gradients', data=stats, row='lambda', col='cost',  hue='key', lowess=True, scatter_kws={"s": 3})
    else: 
        p = sns.lmplot('global_batch_index', 'gradients', data=stats, hue='key', lowess=True, scatter_kws={"s": 3})
    p.map(plt.plot, 'global_batch_index', 'gradients', marker="o", ms=2)
    return global_batch_index[-1]
    
def batch_plot_model_loss(params):
    global_batch_index = []; activity = []; loss = []; key = []; model = []; lam = []; cost = []
    m = params["model_type"]
    for n in params["n_neurons"]:
        for c in params["costs"]:
            for idx, l in enumerate(params["lams"]):
                mod_label = m+'_'+str(n)+'_'+c+'_'+l+'_'+params["version"] if m == 'lca' else m
                log = lp.read_stats(lp.load_file(params["out_dir"]+mod_label+'/logfiles/'+mod_label+'.log')) 
                global_batch_index.extend(list(log['global_batch_index'])*3)
                loss.extend(log['total_loss'] + log['recon_loss'] + log['sparse_loss'])
                model.extend([mod_label] * 3 * len(log['global_batch_index']))
                key.extend(['total'] * len(log['global_batch_index']) + ['recon'] * len(log['global_batch_index']) + ['sparse'] * len(log['global_batch_index']))
                cost.extend([c] * 3 * len(log['global_batch_index']))
                lam.extend([l] * 3 * len(log['global_batch_index']))
    stats = pd.DataFrame({'model': model, 'global_batch_index': global_batch_index, 'loss': loss, 'key': key, 'cost': cost, 'lambda': lam})
    sns.set_style("darkgrid")
    p = sns.lmplot('global_batch_index', 'loss', data=stats, row='lambda', col='cost',  hue='key', lowess=True, scatter_kws={"s": 3})
    p.map(plt.plot, 'global_batch_index', 'loss', marker="o", ms=2) 
    return global_batch_index[-1]
    
def print_model_sched(params, print_params=False):
    m = params["model_type"]
    for n in params["n_neurons"]:
        for c in params["costs"]:
            for l in params["lams"]:
                model_name = m+'_'+str(n)+'_'+c+'_'+l+'_'+params["version"] if m == "lca" else m
                print(model_name+':') 
                print(lp.read_schedule(lp.load_file(params['out_dir']+model_name+'/logfiles/'+model_name+'.log')))
                if print_params == True:
                    print(lp.read_params(lp.load_file(params['out_dir']+model_name+'/logfiles/'+model_name+'.log')))

def draw_image(*args, **kwargs):
    data = kwargs.pop('data')
    sns.heatmap(np.array(data['reconstruction'])[0], xticklabels=False, yticklabels=False, cmap="Greys_r", **kwargs)
    
def extract_images(filename, num_images=50,
    rand_state=np.random.RandomState()):
    with h5py.File(filename, "r") as f:
        full_img_data = np.array(f["van_hateren_good"], dtype=np.float32)
        im_keep_idx = rand_state.choice(full_img_data.shape[0], num_images,
            replace=False)
        full_img_data = full_img_data[im_keep_idx, ...]
    return full_img_data
    
def plot_reconstructions(params, whole_img=True, patch_idx=None):
    sns.set_style('white')
    img_dim = (params["patch_edge_size"], params["patch_edge_size"])
    with np.load(params["input_dir"]) as d:
        if whole_img == True:
            orig_img = reconstruct_img(d["arr_0"], params)
        else:   
            orig_img = np.reshape(d["arr_0"][patch_idx], img_dim)
    fig, axes = plt.subplots(nrows=len(params["lambdas"])+1, ncols=len(params["costs"]),figsize=(8, 20))
    orig = axes[0][0].imshow(orig_img, cmap="Greys_r")
    axes[0][0].set_title(params["costs"][0]+'\n original image')
    orig2 = axes[0][1].imshow(orig_img, cmap="Greys_r")
    axes[0][1].set_title(params["costs"][1]+'\n original image')
    fig.colorbar(orig, ax=axes[0][0]) 
    fig.colorbar(orig2, ax=axes[0][1])
    mse = {}
    for m in params["model_names"]:
        for n in params["n_neurons"]:
            for i, c in enumerate(params["costs"]):
                for j, l in enumerate(params["lambdas"]):
                    name = m+'_'+str(n)+'_'+c+'_'+str(l) if m == 'lca' else m
                    with np.load(params["out_dir"]+name+'_coeffs.npz') as d:
                        coeffs = d["arr_0"]
                    with np.load(params["out_dir"]+name+'_weights.npz') as d:
                        weights = d["arr_0"]
                    if whole_img == True:
                        recon = reconstruct_img(np.matmul(coeffs, weights.T), params)
                    else:
                        recon = np.reshape(np.matmul(coeffs[patch_idx], weights.T), img_dim)
                    mse[c+str(l)] = np.mean((orig_img - recon)**2)
                    im2 = axes[j+1][i].imshow(recon, cmap="Greys_r")#, vmin=np.min(orig_img), vmax=np.max(orig_img))
                    axes[j+1][i].set_title('lambda: '+str(l)+'\n mse: '+str(mse[c+str(l)]))
                    fig.colorbar(im2, ax=axes[j+1][i])
#     axes[-1].set_xlabel(model_name+' '+str(cost_function), fontsize=30)
    fig.tight_layout()

def reconstruct_img(img_patches, params):
    img_patches = np.reshape(img_patches, (img_patches.shape[0], params["patch_edge_size"], params["patch_edge_size"]))
    reconstruction = np.zeros((1024, 1024))
    p = 0
    for i in range(int(reconstruction.shape[0] / params["patch_edge_size"])):
        for j in range(int(reconstruction.shape[1] / params["patch_edge_size"])):
            reconstruction[i*params["patch_edge_size"]:(i+1)*params["patch_edge_size"], j*params["patch_edge_size"]:(j+1)*params["patch_edge_size"]] = img_patches[p]
            p += 1
    return reconstruction

def print_mse(params):
    for m in params["model_names"]:
        for n in params["n_neurons"]:
            for c in params["costs"]:
                for l in params["lambdas"]:
                    name = m+'_'+str(n)+'_'+c+'_'+str(l) if m == 'lca' else m
                    with np.load(params["out_dir"]+name+'_coeffs.npz') as d:
                        coeffs = d["arr_0"]
                    with np.load(params["out_dir"]+name+'_weights.npz') as d:
                        weights = d["arr_0"]
                    with np.load(params["input_dir"]) as d:
                        orig_img = d["arr_0"]
                    recon = reconstruct_img(np.matmul(coeffs, weights.T), params)
                    orig_img = reconstruct_img(orig_img, params)
                    print(name + ': '+ str(np.mean((orig_img - recon)**2)))

def plot_bases(weights, padding=None):
    """
    Plot all bases:
    weights: [np.ndarray] with shape [num_inputs, num_outputs]
      num_inputs must have even square root.
    """
    num_inputs, num_outputs = weights.shape
    assert np.sqrt(num_inputs) == np.floor(np.sqrt(num_inputs)), (
    "weights.shape[0] must have an even square root.")
    patch_edge_size = int(np.sqrt(num_inputs))
    fig, ax = plt.subplots(figsize=(64,64))
    plot_data = pad_data(weights.T.reshape((num_outputs, patch_edge_size,
    patch_edge_size)))
    bf_axis_image = ax.imshow(plot_data, cmap="Greys_r",
    interpolation="nearest")
    ax.tick_params(axis="both", bottom="off", top="off", left="off",
    right="off")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Basis Functions", fontsize=32)

def discretize_and_recon(params):
    ## Load in data and reconstruct whole image
    with np.load(params["input_dir"]) as d:
        data = d['arr_0'].item()['train']
        orig_img = data.images
    if params["white"] == True:
        orig_img = np.matmul(orig_img, np.linalg.inv(data.w_filter))
        orig_img += data.patch_means
    reshape = (orig_img.shape[0], params['patch_edge_size'], params['patch_edge_size'])
    orig_img = np.reshape(orig_img, reshape) 
    orig_img = dp.patches_to_image(orig_img, params['num_im'], params['im_edge_size'])
    data_range = np.max(orig_img) - np.min(orig_img)
    
    ## Load in model coeffs and weights
    if params['model_type'] == 'rg':
        logs = pickle.load(open(params['out_dir'], 'rb'))
        coeffs = logs['coded_patches']
        radial_scalings = logs['radial_scalings']
        p_whitening = logs['p_whitening']
        name = params['model_type']
    else:
        if params['model_type'] == 'lca':
            name = params['model_type']+'_'+str(params['n_neurons'])+'_'+params['cost']+'_'+str(params['lam'])+'_'+params['version']
        elif params['model_type'] == 'ica': 
            name = params['model_type'] + '_' + params['version']
        else: 
            name = params['model_type']
        with np.load(params["out_dir"]+name+'_coeffs.npz') as d:
            coeffs = d["arr_0"]
        with np.load(params["out_dir"]+name+'_weights.npz') as d:
            weights = d["arr_0"]
    p_active =  str(round((np.count_nonzero(coeffs) / (coeffs.shape[0] * coeffs.shape[1])), 4)*100) + '%'   
    
    ## Create plot
    plt.rcParams["figure.figsize"] = params["fig_size"]
    fig, axes = plt.subplots(nrows=len(params["n_bins"])+1)
    axes[0].imshow(orig_img, cmap="Greys_r")
    axes[0].set_title(name + '\n p active: ' + p_active + '\n \n ' + 'original image')
    
    ## Generate reconstructions for quantized coeffs and plot
    MSE = []; PSNR = []; MSSIM = []
    for i, n_bins in enumerate(params['n_bins']):
        disc_coeffs, H = rd.discretize_coeffs(coeffs, n_bins, params['disc_type'])
        if params['model_type'] == 'rg':
            recon = invert_rg(disc_coeffs, radial_scalings, p_whitening).T
        else:
            recon = np.matmul(disc_coeffs, weights.T)
        if params["white"] == True:
            recon = np.matmul(recon, np.linalg.inv(data.w_filter))
            recon += data.patch_means
        recon = np.reshape(recon, reshape)
        recon = dp.patches_to_image(recon, params['num_im'], params['im_edge_size'])
        axes[i+1].imshow(recon, cmap="Greys_r")
        mse = np.mean((orig_img - recon)**2)
        struc_sim = ssim(orig_img, recon, data_range=data_range)
        psnr = np.mean(10 * np.log10(np.max(orig_img) ** 2 / mse)) 
        MSE.append(mse); PSNR.append(psnr); MSSIM.append(struc_sim)
        axes[i+1].set_title('n_bins: ' + str(n_bins) + '\n mse: ' + str(mse) + '\n psnr: ' + str(psnr) + '\n mssim: ' + str(struc_sim) + '\n transmission rate: ' + str(H  * params['n_neurons'] / 256))
    stats = {'mse': MSE, 'psnr': PSNR, 'mssim': MSSIM}
    return stats